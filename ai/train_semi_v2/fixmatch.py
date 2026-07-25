"""FixMatch training mechanics for semi-supervised DR grading.

The public seam is intentionally small: parse the five grade thresholds,
maintain an EMA teacher, and train one epoch from separate labeled and
unlabeled loaders. Dataset discovery, checkpoint policy, and validation remain
owned by :mod:`ai.train_semi_v2.train`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from ai.grading.train import IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES


def parse_grade_thresholds(
    raw: Optional[str | Sequence[float]],
    *,
    fallback: float,
) -> tuple[float, ...]:
    """Return one confidence threshold for each ICDR grade 0..4."""
    if raw is None:
        values = [float(fallback)] * NUM_CLASSES
    elif isinstance(raw, str):
        values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    else:
        values = [float(value) for value in raw]
    if len(values) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} comma-separated grade thresholds for grades "
            f"0..{NUM_CLASSES - 1}; received {values}"
        )
    if any(value < 0.5 or value > 1.0 for value in values):
        raise ValueError("Every grade threshold must be between 0.5 and 1.0")
    return tuple(values)


def build_fixmatch_transforms(
    image_size: int,
) -> tuple[transforms.Compose, transforms.Compose]:
    """Build lesion-preserving weak and strong views of one fundus image."""
    weak = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5, interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    strong = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(
                degrees=15,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.ColorJitter(
                brightness=0.20,
                contrast=0.20,
                saturation=0.10,
                hue=0.02,
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
                p=0.20,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return weak, strong


class EMATeacher:
    """Frozen exponential-moving-average copy of a trainable student."""

    def __init__(self, student: nn.Module, *, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be between 0 and 1")
        self.decay = float(decay)
        self.model = copy.deepcopy(student)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, student: nn.Module) -> None:
        student_state = student.state_dict()
        teacher_state = self.model.state_dict()
        for name, teacher_value in teacher_state.items():
            student_value = student_state[name].detach()
            if torch.is_floating_point(teacher_value):
                teacher_value.mul_(self.decay).add_(
                    student_value.to(teacher_value.device),
                    alpha=1.0 - self.decay,
                )
            else:
                teacher_value.copy_(student_value.to(teacher_value.device))

    def state_dict(self) -> Mapping[str, Any]:
        return self.model.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.model.load_state_dict(state, strict=True)
        self.model.eval()


@dataclass
class FixMatchEpochStats:
    total_loss: float
    supervised_loss: float
    unsupervised_loss: float
    accepted: int
    seen_unlabeled: int
    accepted_per_grade: dict[int, int] = field(default_factory=dict)
    predicted_per_grade: dict[int, int] = field(default_factory=dict)
    mean_confidence: float = 0.0
    final_unsupervised_weight: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / max(self.seen_unlabeled, 1)


def _next_or_restart(
    iterator: Iterable[Any],
    loader: DataLoader,
) -> tuple[Any, Iterable[Any]]:
    try:
        return next(iterator), iterator
    except StopIteration:
        restarted = iter(loader)
        return next(restarted), restarted


def _ramped_weight(
    maximum: float,
    *,
    epoch_index: int,
    step_index: int,
    steps_per_epoch: int,
    warmup_epochs: int,
) -> float:
    if warmup_epochs <= 0:
        return float(maximum)
    completed_steps = epoch_index * steps_per_epoch + step_index + 1
    warmup_steps = warmup_epochs * steps_per_epoch
    return float(maximum) * min(1.0, completed_steps / max(warmup_steps, 1))


def train_fixmatch_epoch(
    student: nn.Module,
    teacher: EMATeacher,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    *,
    grade_thresholds: Sequence[float],
    max_unsupervised_weight: float,
    unsupervised_warmup_epochs: int,
    epoch_index: int,
    accum_steps: int,
    amp_enabled: bool,
    log_every: int = 0,
    log_fn: Optional[Callable[[str], None]] = None,
) -> FixMatchEpochStats:
    """Train one student epoch while generating online EMA pseudo-labels."""
    thresholds = parse_grade_thresholds(
        grade_thresholds, fallback=float(grade_thresholds[0])
    )
    threshold_tensor = torch.tensor(thresholds, dtype=torch.float32, device=device)
    steps_per_epoch = len(labeled_loader)
    if steps_per_epoch == 0 or len(unlabeled_loader) == 0:
        raise ValueError("FixMatch requires non-empty labeled and unlabeled loaders")

    student.train()
    teacher.model.eval()
    optimizer.zero_grad(set_to_none=True)
    labeled_iterator = iter(labeled_loader)
    unlabeled_iterator = iter(unlabeled_loader)

    total_loss_sum = 0.0
    supervised_loss_sum = 0.0
    unsupervised_loss_sum = 0.0
    confidence_sum = 0.0
    seen_unlabeled = 0
    accepted = 0
    accepted_per_grade = {grade: 0 for grade in range(NUM_CLASSES)}
    predicted_per_grade = {grade: 0 for grade in range(NUM_CLASSES)}
    final_weight = 0.0

    for step in range(steps_per_epoch):
        labeled_batch, labeled_iterator = _next_or_restart(
            labeled_iterator, labeled_loader
        )
        unlabeled_batch, unlabeled_iterator = _next_or_restart(
            unlabeled_iterator, unlabeled_loader
        )
        labeled_images, labeled_targets = labeled_batch[0], labeled_batch[1]
        weak_images, strong_images = unlabeled_batch[0], unlabeled_batch[1]
        labeled_images = labeled_images.to(device, non_blocking=True)
        labeled_targets = labeled_targets.to(device, non_blocking=True)
        weak_images = weak_images.to(device, non_blocking=True)
        strong_images = strong_images.to(device, non_blocking=True)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp_enabled):
            teacher_probabilities = torch.softmax(teacher.model(weak_images), dim=1)
            confidence, pseudo_targets = teacher_probabilities.max(dim=1)
            mask = confidence >= threshold_tensor[pseudo_targets]

        unsupervised_weight = _ramped_weight(
            max_unsupervised_weight,
            epoch_index=epoch_index,
            step_index=step,
            steps_per_epoch=steps_per_epoch,
            warmup_epochs=unsupervised_warmup_epochs,
        )
        final_weight = unsupervised_weight
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            supervised_logits = student(labeled_images)
            strong_logits = student(strong_images)
            supervised_loss = F.cross_entropy(
                supervised_logits,
                labeled_targets,
                label_smoothing=0.05,
            )
            per_unlabeled = F.cross_entropy(
                strong_logits,
                pseudo_targets,
                reduction="none",
            )
            accepted_count = int(mask.sum().item())
            if accepted_count:
                unsupervised_loss = per_unlabeled[mask].mean()
            else:
                unsupervised_loss = strong_logits.sum() * 0.0
            loss = supervised_loss + unsupervised_weight * unsupervised_loss

        scaler.scale(loss / accum_steps).backward()
        should_step = (step + 1) % accum_steps == 0 or step + 1 == steps_per_epoch
        if should_step:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            teacher.update(student)

        batch_size = int(pseudo_targets.numel())
        seen_unlabeled += batch_size
        accepted += accepted_count
        confidence_sum += float(confidence.sum().item())
        for grade in range(NUM_CLASSES):
            predicted_per_grade[grade] += int((pseudo_targets == grade).sum().item())
            accepted_per_grade[grade] += int(
                ((pseudo_targets == grade) & mask).sum().item()
            )
        total_loss_sum += float(loss.detach().item())
        supervised_loss_sum += float(supervised_loss.detach().item())
        unsupervised_loss_sum += float(unsupervised_loss.detach().item())

        if log_fn is not None and log_every > 0 and (
            step == 0 or step + 1 == steps_per_epoch or (step + 1) % log_every == 0
        ):
            log_fn(
                "FixMatch train: "
                f"step={step + 1}/{steps_per_epoch}, "
                f"loss={total_loss_sum / (step + 1):.6f}, "
                f"sup={supervised_loss_sum / (step + 1):.6f}, "
                f"unsup={unsupervised_loss_sum / (step + 1):.6f}, "
                f"lambda_u={unsupervised_weight:.4f}, "
                f"accepted={accepted}/{seen_unlabeled} "
                f"({100.0 * accepted / max(seen_unlabeled, 1):.1f}%)"
            )

    denominator = max(steps_per_epoch, 1)
    return FixMatchEpochStats(
        total_loss=total_loss_sum / denominator,
        supervised_loss=supervised_loss_sum / denominator,
        unsupervised_loss=unsupervised_loss_sum / denominator,
        accepted=accepted,
        seen_unlabeled=seen_unlabeled,
        accepted_per_grade=accepted_per_grade,
        predicted_per_grade=predicted_per_grade,
        mean_confidence=confidence_sum / max(seen_unlabeled, 1),
        final_unsupervised_weight=final_weight,
    )
