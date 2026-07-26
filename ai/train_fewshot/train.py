"""Fixed-support few-shot adaptation connected to a grading checkpoint.

The source grading checkpoint must not have been trained on the target domain.
Exactly K labeled target images per DR grade are selected once from the target
training split and embedded in each checkpoint. Adaptation only sees that fixed
support set. The target test split is read only in explicit ``--eval-only``
mode, never for training, early stopping, or checkpoint selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path
import time
from typing import Any, Collection, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from ai.grading.train import (
    FundusDataset,
    NUM_CLASSES,
    build_transforms,
    calculate_metrics,
    create_scaler,
)
from ai.train_fewshot.runtime import (
    load_grading_checkpoint,
    load_split_frames,
    prepare_fresh_output_dir,
    seed_everything,
)


def log(message: str) -> None:
    print(message, flush=True)


METRIC_FIELDS = (
    "stage",
    "epoch",
    "loss",
    "best_loss",
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "qwk",
)


def save_metrics(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    replace_stages: Collection[str] = (),
) -> None:
    """Keep epoch history and before/after scores in one small CSV file."""
    existing: List[Dict[str, Any]] = []
    if path.is_file():
        with path.open("r", encoding="utf-8", newline="") as handle:
            existing = [
                row
                for row in csv.DictReader(handle)
                if row.get("stage") not in replace_stages
            ]
    existing.extend(
        {field: row.get(field, "") for field in METRIC_FIELDS}
        for row in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(existing)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--target-dataset-dir",
        "--dataset-dir",
        dest="target_dataset_dir",
        required=True,
        type=Path,
        help="Unseen target-domain dataset with fixed train/validation/test splits",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--train-episodes", type=int, default=20)
    parser.add_argument(
        "--shots",
        type=int,
        default=5,
        help="Total labeled target images per class available to adaptation",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=1,
        help="Support images per class temporarily held out as query in each episode",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=0,
        help="0 keeps the pretrained embedding; a positive value adds a projection",
    )
    parser.add_argument("--encoder-lr", type=float, default=1e-6)
    parser.add_argument("--projection-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--unfreeze-last-blocks", type=int, default=1)
    parser.add_argument(
        "--forward-batch-size",
        type=int,
        default=2,
        help="Chunk size used by the large RETFound encoder",
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enhance", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from last.pth inside the same output directory",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate an adapted --resume checkpoint on held-out target test",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "epochs",
        "patience",
        "train_episodes",
        "shots",
        "queries",
        "forward_batch_size",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.queries > args.shots:
        raise ValueError("--queries cannot exceed --shots")
    if args.shots > 1 and args.queries >= args.shots:
        raise ValueError("--queries must leave at least one prototype image per class")
    if args.embedding_dim < 0:
        raise ValueError("--embedding-dim cannot be negative")
    if args.unfreeze_last_blocks < 0:
        raise ValueError("--unfreeze-last-blocks cannot be negative")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.eval_only and args.resume is None:
        raise ValueError("--eval-only requires --resume CHECKPOINT")


def prepare_output_dir(output_dir: Path, resume: Optional[Path]) -> Path:
    output_dir = output_dir.expanduser().resolve()
    if resume is None:
        return prepare_fresh_output_dir(output_dir)
    resume = resume.expanduser().resolve()
    if not resume.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
    if resume.parent != output_dir:
        raise ValueError("--resume must be a checkpoint inside --output-dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def select_fixed_support(frame: pd.DataFrame, shots: int, seed: int) -> pd.DataFrame:
    """Select exactly K target images per class once for the whole run."""
    rng = random.Random(seed)
    selected: List[pd.DataFrame] = []
    for grade in range(NUM_CLASSES):
        indices = frame.index[frame["diagnosis"] == grade].tolist()
        if len(indices) < shots:
            raise ValueError(
                f"Target train grade {grade} has {len(indices)} images but {shots}-shot "
                "adaptation needs at least that many"
            )
        chosen = rng.sample(indices, shots)
        class_frame = frame.loc[chosen].copy()
        class_frame["support_rank"] = list(range(1, shots + 1))
        selected.append(class_frame)
    support = pd.concat(selected, ignore_index=True)
    support["support_seed"] = seed
    return support


class FixedSupportEpisodeSampler:
    """Create leave-out episodes using only one previously selected support set."""

    def __init__(self, support_frame: pd.DataFrame, seed: int) -> None:
        self.rng = random.Random(seed)
        self.by_class: Dict[int, List[int]] = {
            grade: support_frame.index[support_frame["diagnosis"] == grade].tolist()
            for grade in range(NUM_CLASSES)
        }

    def sample(self, queries: int) -> Tuple[List[int], List[int], List[int], List[int]]:
        support_indices: List[int] = []
        support_labels: List[int] = []
        query_indices: List[int] = []
        query_labels: List[int] = []
        for grade in range(NUM_CLASSES):
            available = self.by_class[grade]
            if not available:
                raise ValueError(f"Fixed support is missing grade {grade}")
            if len(available) == 1:
                if queries != 1:
                    raise ValueError("One-shot adaptation supports exactly one query view")
                chosen_queries = available
                chosen_support = available
            else:
                if queries >= len(available):
                    raise ValueError(
                        f"Grade {grade} has {len(available)} fixed shots; queries must be smaller"
                    )
                chosen_queries = self.rng.sample(available, queries)
                query_set = set(chosen_queries)
                chosen_support = [index for index in available if index not in query_set]
            support_indices.extend(chosen_support)
            support_labels.extend([grade] * len(chosen_support))
            query_indices.extend(chosen_queries)
            query_labels.extend([grade] * len(chosen_queries))
        return support_indices, support_labels, query_indices, query_labels


def load_episode_images(dataset: Dataset, indices: Sequence[int]) -> torch.Tensor:
    return torch.stack([dataset[index][0] for index in indices])


class RetfoundProtoNet(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        embedding_dim: int,
        *,
        temperature: float,
        forward_batch_size: int,
    ) -> None:
        super().__init__()
        feature_dim = int(getattr(encoder, "num_features", 0))
        if feature_dim <= 0:
            raise ValueError("Encoder does not expose a valid num_features value")
        self.encoder = encoder
        if embedding_dim == 0:
            self.projection: nn.Module = nn.Identity()
            self.embedding_dim = feature_dim
        else:
            projection = nn.Linear(feature_dim, embedding_dim, bias=False)
            nn.init.orthogonal_(projection.weight)
            self.projection = projection
            self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.forward_batch_size = forward_batch_size

    def _encode_chunk(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder.forward_features(images)
        embeddings = self.encoder.forward_head(features, pre_logits=True)
        if embeddings.ndim != 2:
            embeddings = embeddings.flatten(1)
        return F.normalize(self.projection(embeddings), dim=1)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        chunks = [
            self._encode_chunk(images[start : start + self.forward_batch_size])
            for start in range(0, images.size(0), self.forward_batch_size)
        ]
        return torch.cat(chunks, dim=0)

    def logits_from_prototypes(
        self, query_images: torch.Tensor, prototypes: torch.Tensor
    ) -> torch.Tensor:
        distances = torch.cdist(self.encode(query_images), prototypes).pow(2)
        return -distances / self.temperature

    def episode_logits(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        support_embeddings = self.encode(support_images)
        class_ids = torch.unique(support_labels, sorted=True)
        prototypes = torch.stack(
            [support_embeddings[support_labels == class_id].mean(dim=0) for class_id in class_ids]
        )
        return self.logits_from_prototypes(query_images, prototypes), class_ids


def configure_encoder_trainability(encoder: nn.Module, last_blocks: int) -> int:
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    if last_blocks == 0:
        return 0
    blocks = getattr(encoder, "blocks", None)
    if blocks is None:
        raise ValueError("Encoder does not expose transformer blocks")
    if last_blocks > len(blocks):
        raise ValueError(f"Requested {last_blocks} blocks but encoder only has {len(blocks)}")
    for block in blocks[-last_blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    for norm_name in ("norm", "fc_norm"):
        norm = getattr(encoder, norm_name, None)
        if norm is not None:
            for parameter in norm.parameters():
                parameter.requires_grad = True
    return sum(parameter.numel() for parameter in encoder.parameters() if parameter.requires_grad)


def _local_targets(global_targets: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
    matches = global_targets.unsqueeze(1) == class_ids.unsqueeze(0)
    if not bool(matches.any(dim=1).all()):
        raise ValueError("Query label is missing from the support set")
    return matches.to(torch.int64).argmax(dim=1)


def run_episode(
    model: RetfoundProtoNet,
    dataset: Dataset,
    episode: Tuple[List[int], List[int], List[int], List[int]],
    device: torch.device,
    *,
    amp_enabled: bool,
) -> Tuple[torch.Tensor, List[int], List[int]]:
    support_indices, support_labels, query_indices, query_labels = episode
    support_images = load_episode_images(dataset, support_indices).to(device)
    query_images = load_episode_images(dataset, query_indices).to(device)
    support_targets = torch.tensor(support_labels, device=device)
    query_targets = torch.tensor(query_labels, device=device)
    with torch.amp.autocast("cuda", enabled=amp_enabled):
        logits, class_ids = model.episode_logits(support_images, support_targets, query_images)
        local_targets = _local_targets(query_targets, class_ids)
        loss = F.cross_entropy(logits, local_targets)
    predictions = class_ids[logits.argmax(dim=1)]
    return (
        loss,
        query_targets.detach().cpu().tolist(),
        predictions.detach().cpu().tolist(),
    )


@torch.no_grad()
def compute_prototypes(
    model: RetfoundProtoNet,
    support_dataset: Dataset,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    images = load_episode_images(support_dataset, list(range(len(support_dataset)))).to(device)
    labels = torch.tensor(
        [int(support_dataset[index][1]) for index in range(len(support_dataset))],
        device=device,
    )
    embeddings = model.encode(images)
    class_ids = torch.arange(NUM_CLASSES, device=device)
    prototypes = torch.stack(
        [embeddings[labels == class_id].mean(dim=0) for class_id in class_ids]
    )
    return prototypes, class_ids


@torch.no_grad()
def evaluate_source_classifier(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    *,
    batch_size: int,
    amp_enabled: bool,
) -> Dict[str, Any]:
    model.eval()
    targets: List[int] = []
    predictions: List[int] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for images, labels, _ in tqdm(loader, desc="source-target-test", leave=False):
        images = images.to(device)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(images)
        targets.extend(labels.tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return calculate_metrics(targets, predictions)


@torch.no_grad()
def evaluate_protonet(
    model: RetfoundProtoNet,
    support_dataset: Dataset,
    evaluation_dataset: Dataset,
    device: torch.device,
    *,
    batch_size: int,
    amp_enabled: bool,
) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]:
    model.eval()
    prototypes, class_ids = compute_prototypes(model, support_dataset, device)
    targets: List[int] = []
    predictions: List[int] = []
    loader = DataLoader(evaluation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for images, labels, _ in tqdm(loader, desc="few-shot-target-test", leave=False):
        images = images.to(device)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model.logits_from_prototypes(images, prototypes)
        targets.extend(labels.tolist())
        predictions.extend(class_ids[logits.argmax(dim=1)].cpu().tolist())
    return calculate_metrics(targets, predictions), prototypes, class_ids


def _dataset_args(saved_args: argparse.Namespace, args: argparse.Namespace) -> argparse.Namespace:
    values = dict(vars(saved_args))
    values.update(
        {
            "dataset_dir": args.target_dataset_dir,
            "images_dir": None,
            "labels_csv": None,
            "image_column": "id_code",
            "label_column": "diagnosis",
            "image_extension": ".png",
            "enhance": args.enhance,
        }
    )
    return argparse.Namespace(**values)


def _assert_target_path_differs_from_recorded_source(
    saved_args: argparse.Namespace, target_dataset_dir: Path
) -> None:
    recorded = getattr(saved_args, "dataset_dir", None)
    if recorded is None:
        return
    source = Path(recorded).expanduser().resolve()
    target = target_dataset_dir.expanduser().resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError(
            "Target dataset path overlaps the dataset recorded in the source checkpoint. "
            "Few-shot adaptation requires a domain unseen during source training."
        )


def save_adapted_checkpoint(
    path: Path,
    model: RetfoundProtoNet,
    prototypes: torch.Tensor,
    class_ids: torch.Tensor,
    args: argparse.Namespace,
    saved_args: argparse.Namespace,
    support_frame: pd.DataFrame,
    *,
    epoch: int,
    best_support_loss: float,
    stale_epochs: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Any = None,
    sampler_state: Any = None,
) -> None:
    state: Dict[str, Any] = {
        "method": "fixed_support_target_domain_protonet",
        "encoder_model": model.encoder.state_dict(),
        "projection": model.projection.state_dict(),
        "embedding_dim": model.embedding_dim,
        "prototypes": prototypes.detach().cpu(),
        "class_ids": class_ids.detach().cpu(),
        "temperature": model.temperature,
        "base_model_args": dict(vars(saved_args)),
        "few_shot_args": {
            key: os.fspath(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "parent_checkpoint": os.fspath(args.checkpoint.resolve()),
        "support_rows": support_frame.to_dict(orient="records"),
        "requires_support_set": False,
        "target_test_used_for_selection": False,
        "epoch": epoch,
        "best_support_loss": best_support_loss,
        "stale_epochs": stale_epochs,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    if sampler_state is not None:
        state["sampler_state"] = sampler_state
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def _metric_delta(after: Dict[str, Any], before: Dict[str, Any]) -> Dict[str, float]:
    keys = ("accuracy", "macro_f1", "balanced_accuracy", "qwk")
    return {key: float(after[key]) - float(before[key]) for key in keys}


def run(args: argparse.Namespace) -> None:
    run_started = time.perf_counter()
    validate_args(args)
    args.output_dir = prepare_output_dir(args.output_dir, args.resume)
    metrics_path = args.output_dir / "metrics.csv"
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for RETFound few-shot adaptation")
    device = torch.device("cuda")
    amp_enabled = not args.no_amp
    log(
        f"GPU={torch.cuda.get_device_name(0)}, AMP={amp_enabled}, "
        f"mode={'eval' if args.eval_only else 'resume' if args.resume else 'train-new'}"
    )

    bundle = load_grading_checkpoint(args.checkpoint, device, require_ce=True)
    encoder, saved_args = bundle.model, bundle.saved_args
    _assert_target_path_differs_from_recorded_source(saved_args, args.target_dataset_dir)
    frames, _ = load_split_frames(args.target_dataset_dir)
    resume_state = (
        torch.load(args.resume, map_location=device, weights_only=False)
        if args.resume is not None
        else None
    )
    if resume_state is not None:
        saved_few_shot_args = resume_state.get("few_shot_args", {})
        contract_keys = (
            "shots",
            "queries",
            "embedding_dim",
            "unfreeze_last_blocks",
            "temperature",
            "seed",
        )
        mismatches = {
            key: (saved_few_shot_args.get(key), getattr(args, key))
            for key in contract_keys
            if saved_few_shot_args.get(key) != getattr(args, key)
        }
        if mismatches:
            raise ValueError(
                "Resume configuration differs from checkpoint: "
                + json.dumps(mismatches, default=str)
            )
    if resume_state is None:
        support_frame = select_fixed_support(frames["train"], args.shots, args.seed)
    else:
        support_rows = resume_state.get("support_rows")
        if support_rows is not None:
            support_frame = pd.DataFrame(support_rows)
        else:
            legacy_support_path = args.output_dir / "support_manifest.csv"
            if not legacy_support_path.is_file():
                raise FileNotFoundError(
                    "Resume checkpoint does not embed its fixed support set and "
                    f"the legacy manifest is missing: {legacy_support_path}"
                )
            support_frame = pd.read_csv(legacy_support_path)
        expected_shots = int(resume_state["few_shot_args"]["shots"])
        actual_counts = support_frame["diagnosis"].value_counts()
        if not all(int(actual_counts.get(grade, 0)) == expected_shots for grade in range(NUM_CLASSES)):
            raise ValueError("Support manifest no longer matches the resume checkpoint")
    log(
        f"Fixed target support={len(support_frame)} ({len(support_frame) // NUM_CLASSES}/class); "
        f"target_train_total={len(frames['train'])}, target_validation_held_out={len(frames['val'])}, "
        f"target_test_held_out={len(frames['test'])}."
    )

    image_size = int(getattr(saved_args, "image_size", 224))
    train_transform, eval_transform = build_transforms(image_size)
    data_args = _dataset_args(saved_args, args)
    train_support_dataset = FundusDataset(support_frame, data_args, train_transform)
    eval_support_dataset = FundusDataset(support_frame, data_args, eval_transform)
    target_test_dataset = FundusDataset(frames["test"], data_args, eval_transform)

    if args.eval_only:
        source_metrics = evaluate_source_classifier(
            encoder,
            target_test_dataset,
            device,
            batch_size=args.forward_batch_size,
            amp_enabled=amp_enabled,
        )

    trainable_encoder = configure_encoder_trainability(encoder, args.unfreeze_last_blocks)
    model = RetfoundProtoNet(
        encoder,
        args.embedding_dim,
        temperature=args.temperature,
        forward_batch_size=args.forward_batch_size,
    ).to(device)
    log(f"Trainable encoder parameters={trainable_encoder / 1e6:.2f}M")

    if resume_state is not None:
        model.encoder.load_state_dict(resume_state["encoder_model"])
        model.projection.load_state_dict(resume_state["projection"])
        log(
            f"Loaded adapted checkpoint epoch={resume_state.get('epoch', -1) + 1}, "
            f"best_support_loss={resume_state.get('best_support_loss')}."
        )

    if args.eval_only:
        adapted_metrics, prototypes, class_ids = evaluate_protonet(
            model,
            eval_support_dataset,
            target_test_dataset,
            device,
            batch_size=args.forward_batch_size,
            amp_enabled=amp_enabled,
        )
        comparison = {
            "source_classifier_without_target_adaptation": source_metrics,
            "few_shot_adapted_protonet": adapted_metrics,
            "delta": _metric_delta(adapted_metrics, source_metrics),
            "target_test_used_for_training": False,
            "target_test_used_for_model_selection": False,
        }
        score_rows = []
        for stage, values in (
            ("before", source_metrics),
            ("after", adapted_metrics),
            ("delta", comparison["delta"]),
        ):
            score_rows.append(
                {
                    "stage": stage,
                    **{
                        key: values[key]
                        for key in (
                            "accuracy",
                            "macro_f1",
                            "balanced_accuracy",
                            "qwk",
                        )
                    },
                }
            )
        save_metrics(
            metrics_path,
            score_rows,
            replace_stages={"before", "after", "delta"},
        )
        log("Held-out target test evaluation:\n" + json.dumps(comparison, indent=2))
        return

    encoder_parameters = [
        parameter for parameter in model.encoder.parameters() if parameter.requires_grad
    ]
    projection_parameters = [
        parameter for parameter in model.projection.parameters() if parameter.requires_grad
    ]
    groups: List[Dict[str, Any]] = []
    if projection_parameters:
        groups.append({"params": projection_parameters, "lr": args.projection_lr})
    if encoder_parameters:
        groups.append({"params": encoder_parameters, "lr": args.encoder_lr})
    if not groups:
        raise ValueError(
            "No trainable parameters. Set --unfreeze-last-blocks above 0 or use a "
            "positive --embedding-dim."
        )
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scaler = create_scaler(amp_enabled)
    sampler = FixedSupportEpisodeSampler(support_frame, args.seed)
    start_epoch = 0
    best_support_loss = math.inf
    stale_epochs = 0
    if resume_state is not None:
        required = {"optimizer", "epoch", "best_support_loss", "stale_epochs"}
        missing = required.difference(resume_state)
        if missing:
            raise ValueError(f"Resume checkpoint missing state: {sorted(missing)}")
        optimizer.load_state_dict(resume_state["optimizer"])
        if resume_state.get("scaler"):
            scaler.load_state_dict(resume_state["scaler"])
        if resume_state.get("sampler_state") is not None:
            sampler.rng.setstate(resume_state["sampler_state"])
        start_epoch = int(resume_state["epoch"]) + 1
        best_support_loss = float(resume_state["best_support_loss"])
        stale_epochs = int(resume_state["stale_epochs"])
        log(
            f"Resume restored: next_epoch={start_epoch + 1}/{args.epochs}, "
            f"best_support_loss={best_support_loss:.6f}, "
            f"patience={stale_epochs}/{args.patience}."
        )

    for epoch in range(start_epoch, args.epochs):
        epoch_started = time.perf_counter()
        model.train()
        losses: List[float] = []
        epoch_targets: List[int] = []
        epoch_predictions: List[int] = []
        for _ in tqdm(
            range(args.train_episodes),
            desc=f"few-shot epoch {epoch + 1}/{args.epochs}",
            leave=True,
        ):
            optimizer.zero_grad(set_to_none=True)
            loss, targets, predictions = run_episode(
                model,
                train_support_dataset,
                sampler.sample(args.queries),
                device,
                amp_enabled=amp_enabled,
            )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
            epoch_targets.extend(targets)
            epoch_predictions.extend(predictions)
        support_loss = sum(losses) / len(losses)
        epoch_metrics = calculate_metrics(epoch_targets, epoch_predictions)
        improved = support_loss < best_support_loss
        if improved:
            best_support_loss = support_loss
            stale_epochs = 0
        else:
            stale_epochs += 1
        record = {
            "stage": "epoch",
            "epoch": epoch + 1,
            "loss": support_loss,
            "best_loss": best_support_loss,
            **{
                key: epoch_metrics[key]
                for key in (
                    "accuracy",
                    "macro_f1",
                    "balanced_accuracy",
                    "qwk",
                )
            },
        }
        log(
            f"Epoch {epoch + 1}/{args.epochs} complete in "
            f"{time.perf_counter() - epoch_started:.1f}s: "
            + json.dumps(record, ensure_ascii=False)
        )
        save_metrics(metrics_path, [record])
        prototypes, class_ids = compute_prototypes(
            model, eval_support_dataset, device
        )
        checkpoint_kwargs = {
            "epoch": epoch,
            "best_support_loss": best_support_loss,
            "stale_epochs": stale_epochs,
            "optimizer": optimizer,
            "scaler": scaler,
            "sampler_state": sampler.rng.getstate(),
        }
        save_adapted_checkpoint(
            args.output_dir / "last.pth",
            model,
            prototypes,
            class_ids,
            args,
            saved_args,
            support_frame,
            **checkpoint_kwargs,
        )
        if improved:
            save_adapted_checkpoint(
                args.output_dir / "best.pth",
                model,
                prototypes,
                class_ids,
                args,
                saved_args,
                support_frame,
                **checkpoint_kwargs,
            )
        if stale_epochs >= args.patience:
            log(f"Early stopping: no support-loss improvement for {stale_epochs} epochs.")
            break

    summary = {
        "method": "fixed_support_target_domain_protonet",
        "shots_per_class": args.shots,
        "unique_labeled_target_images": len(support_frame),
        "support_seed": args.seed,
        "target_validation_used": False,
        "target_test_used_for_training": False,
        "target_test_used_for_model_selection": False,
        "target_test_evaluated_during_training": False,
        "requires_support_set_at_inference": False,
        "best_support_loss": best_support_loss,
        "best_checkpoint": "best.pth",
        "last_checkpoint": "last.pth",
        "selection_signal": "fixed-support episodic loss",
    }
    log(
        f"Few-shot training finished in {time.perf_counter() - run_started:.1f}s.\n"
        + json.dumps(summary, indent=2, ensure_ascii=False)
    )


def main() -> None:
    try:
        run(parse_args())
    except KeyboardInterrupt:
        log(
            "Training interrupted. last.pth from the most recent "
            "completed epoch remains safe; rerun with --resume."
        )
        raise SystemExit(130)


if __name__ == "__main__":
    main()
