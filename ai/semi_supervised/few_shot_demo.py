"""Fixed-support few-shot adaptation for a new retinal-image domain.

The source grading checkpoint must not have been trained on the target domain.
Exactly K labeled target images per DR grade are selected once from the target
training split and saved as a manifest. Adaptation only sees that fixed support
set. The target test split is used for reporting before/after metrics, never for
training, early stopping, or checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from ai.semi_supervised.research_utils import (
    load_grading_checkpoint,
    load_split_frames,
    prepare_fresh_output_dir,
    seed_everything,
)


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
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    for name in ("epochs", "train_episodes", "shots", "queries", "forward_batch_size"):
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
) -> torch.Tensor:
    support_indices, support_labels, query_indices, query_labels = episode
    support_images = load_episode_images(dataset, support_indices).to(device)
    query_images = load_episode_images(dataset, query_indices).to(device)
    support_targets = torch.tensor(support_labels, device=device)
    query_targets = torch.tensor(query_labels, device=device)
    with torch.amp.autocast("cuda", enabled=amp_enabled):
        logits, class_ids = model.episode_logits(support_images, support_targets, query_images)
        loss = F.cross_entropy(logits, _local_targets(query_targets, class_ids))
    return loss


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
        "support_manifest": "support_manifest.csv",
        "requires_support_set": False,
        "target_test_used_for_selection": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def _metric_delta(after: Dict[str, Any], before: Dict[str, Any]) -> Dict[str, float]:
    keys = ("accuracy", "macro_f1", "balanced_accuracy", "qwk")
    return {key: float(after[key]) - float(before[key]) for key in keys}


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    args.output_dir = prepare_fresh_output_dir(args.output_dir)
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for RETFound few-shot adaptation")
    device = torch.device("cuda")
    amp_enabled = not args.no_amp

    bundle = load_grading_checkpoint(args.checkpoint, device, require_ce=True)
    encoder, saved_args = bundle.model, bundle.saved_args
    _assert_target_path_differs_from_recorded_source(saved_args, args.target_dataset_dir)
    frames, _ = load_split_frames(args.target_dataset_dir)

    support_frame = select_fixed_support(frames["train"], args.shots, args.seed)
    support_path = args.output_dir / "support_manifest.csv"
    support_frame.to_csv(support_path, index=False)
    print(f"Fixed target support: {len(support_frame)} images ({args.shots}/class)")
    print(f"Target test images: {len(frames['test'])}; validation split is not used")

    image_size = int(getattr(saved_args, "image_size", 224))
    train_transform, eval_transform = build_transforms(image_size)
    data_args = _dataset_args(saved_args, args)
    train_support_dataset = FundusDataset(support_frame, data_args, train_transform)
    eval_support_dataset = FundusDataset(support_frame, data_args, eval_transform)
    target_test_dataset = FundusDataset(frames["test"], data_args, eval_transform)

    source_metrics = evaluate_source_classifier(
        encoder,
        target_test_dataset,
        device,
        batch_size=args.forward_batch_size,
        amp_enabled=amp_enabled,
    )
    trainable_encoder = configure_encoder_trainability(encoder, args.unfreeze_last_blocks)
    print(f"Trainable encoder parameters: {trainable_encoder / 1e6:.2f}M")
    model = RetfoundProtoNet(
        encoder,
        args.embedding_dim,
        temperature=args.temperature,
        forward_batch_size=args.forward_batch_size,
    ).to(device)
    before_metrics, _, _ = evaluate_protonet(
        model,
        eval_support_dataset,
        target_test_dataset,
        device,
        batch_size=args.forward_batch_size,
        amp_enabled=amp_enabled,
    )

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
    history_path = args.output_dir / "history.jsonl"

    for epoch in range(args.epochs):
        model.train()
        losses: List[float] = []
        for _ in tqdm(range(args.train_episodes), desc="target-support-adapt", leave=False):
            optimizer.zero_grad(set_to_none=True)
            loss = run_episode(
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
        record = {
            "epoch": epoch,
            "support_episode_loss": sum(losses) / len(losses),
            "unique_labeled_target_images": len(support_frame),
            "target_test_evaluated": False,
        }
        print(json.dumps(record, ensure_ascii=False))
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    after_metrics, prototypes, class_ids = evaluate_protonet(
        model,
        eval_support_dataset,
        target_test_dataset,
        device,
        batch_size=args.forward_batch_size,
        amp_enabled=amp_enabled,
    )
    save_adapted_checkpoint(
        args.output_dir / "checkpoint-adapted-protonet.pth",
        model,
        prototypes,
        class_ids,
        args,
        saved_args,
    )

    comparison = {
        "source_classifier_without_target_adaptation": source_metrics,
        "prototype_before_gradient_adaptation": before_metrics,
        "prototype_after_gradient_adaptation": after_metrics,
        "adaptation_delta_vs_prototype_before": _metric_delta(after_metrics, before_metrics),
        "adaptation_delta_vs_source_classifier": _metric_delta(after_metrics, source_metrics),
    }
    with (args.output_dir / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2, ensure_ascii=False)

    qwk_delta = float(after_metrics["qwk"]) - float(before_metrics["qwk"])
    qwk_delta_vs_source = float(after_metrics["qwk"]) - float(source_metrics["qwk"])
    if not math.isfinite(qwk_delta) or not math.isfinite(qwk_delta_vs_source):
        raise RuntimeError("Target evaluation produced a non-finite QWK delta")
    summary = {
        "method": "fixed_support_target_domain_protonet",
        "shots_per_class": args.shots,
        "unique_labeled_target_images": len(support_frame),
        "support_seed": args.seed,
        "target_validation_used": False,
        "target_test_used_for_training": False,
        "target_test_used_for_model_selection": False,
        "target_test_evaluated_only_before_and_after": True,
        "requires_support_set_at_inference": False,
        "adapted_checkpoint": "checkpoint-adapted-protonet.pth",
        "source_classifier_qwk": float(source_metrics["qwk"]),
        "qwk_before_adaptation": float(before_metrics["qwk"]),
        "qwk_after_adaptation": float(after_metrics["qwk"]),
        "qwk_delta": qwk_delta,
        "qwk_delta_vs_source_classifier": qwk_delta_vs_source,
        "adapted_beats_source_on_this_run": qwk_delta_vs_source > 0.0,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
