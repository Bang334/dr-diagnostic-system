"""Real-data episodic few-shot training initialized from a RETFound checkpoint.

Episodes are sampled from the fixed training split and model selection uses
episodes from the validation split. The held-out test split remains untouched.
The resulting artifact is a research ProtoNet checkpoint, not a drop-in
replacement for the five-class grading classifier.
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
from torch.utils.data import Dataset
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
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-episodes", type=int, default=40)
    parser.add_argument("--val-episodes", type=int, default=20)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--queries", type=int, default=3)
    parser.add_argument("--embedding-dim", type=int, default=128)
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
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enhance", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "epochs",
        "train_episodes",
        "val_episodes",
        "shots",
        "queries",
        "embedding_dim",
        "forward_batch_size",
        "patience",
    )
    for name in positive:
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.unfreeze_last_blocks < 0:
        raise ValueError("--unfreeze-last-blocks cannot be negative")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")


class EpisodeSampler:
    """Sample balanced five-way support/query episodes from a fixed split."""

    def __init__(self, frame: pd.DataFrame, seed: int) -> None:
        self.rng = random.Random(seed)
        self.by_class: Dict[int, List[int]] = {
            grade: frame.index[frame["diagnosis"] == grade].tolist()
            for grade in range(NUM_CLASSES)
        }

    def sample(self, shots: int, queries: int) -> Tuple[List[int], List[int], List[int], List[int]]:
        required = shots + queries
        support_indices: List[int] = []
        support_labels: List[int] = []
        query_indices: List[int] = []
        query_labels: List[int] = []
        for grade in range(NUM_CLASSES):
            available = self.by_class[grade]
            if len(available) < required:
                raise ValueError(
                    f"Grade {grade} has {len(available)} images but an episode needs {required}"
                )
            chosen = self.rng.sample(available, required)
            support_indices.extend(chosen[:shots])
            support_labels.extend([grade] * shots)
            query_indices.extend(chosen[shots:])
            query_labels.extend([grade] * queries)
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
        self.projection = nn.Linear(feature_dim, embedding_dim)
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

    def episode_logits(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        support_embeddings = self.encode(support_images)
        query_embeddings = self.encode(query_images)
        class_ids = torch.unique(support_labels, sorted=True)
        prototypes = torch.stack(
            [support_embeddings[support_labels == class_id].mean(dim=0) for class_id in class_ids]
        )
        distances = torch.cdist(query_embeddings, prototypes).pow(2)
        return -distances / self.temperature, class_ids


def configure_encoder_trainability(encoder: nn.Module, last_blocks: int) -> int:
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    if last_blocks == 0:
        return 0
    blocks = getattr(encoder, "blocks", None)
    if blocks is None:
        raise ValueError("Encoder does not expose transformer blocks")
    if last_blocks > len(blocks):
        raise ValueError(
            f"Requested {last_blocks} blocks but encoder only has {len(blocks)}"
        )
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
        logits, class_ids = model.episode_logits(
            support_images, support_targets, query_images
        )
        local_targets = _local_targets(query_targets, class_ids)
        loss = F.cross_entropy(logits, local_targets)
    predictions = class_ids[logits.argmax(dim=1)]
    return loss, query_targets.cpu().tolist(), predictions.detach().cpu().tolist()


@torch.no_grad()
def evaluate_episodes(
    model: RetfoundProtoNet,
    dataset: Dataset,
    frame: pd.DataFrame,
    *,
    episodes: int,
    shots: int,
    queries: int,
    seed: int,
    device: torch.device,
    amp_enabled: bool,
) -> Tuple[float, Dict[str, Any]]:
    model.eval()
    sampler = EpisodeSampler(frame, seed)
    losses: List[float] = []
    targets: List[int] = []
    predictions: List[int] = []
    for _ in tqdm(range(episodes), desc="few-shot-val", leave=False):
        loss, episode_targets, episode_predictions = run_episode(
            model,
            dataset,
            sampler.sample(shots, queries),
            device,
            amp_enabled=amp_enabled,
        )
        losses.append(float(loss))
        targets.extend(episode_targets)
        predictions.extend(episode_predictions)
    return sum(losses) / len(losses), calculate_metrics(targets, predictions)


def _dataset_args(saved_args: argparse.Namespace, args: argparse.Namespace) -> argparse.Namespace:
    values = dict(vars(saved_args))
    values.update(
        {
            "dataset_dir": args.dataset_dir,
            "images_dir": None,
            "labels_csv": None,
            "image_column": "id_code",
            "label_column": "diagnosis",
            "image_extension": ".png",
            "enhance": args.enhance,
        }
    )
    return argparse.Namespace(**values)


def save_few_shot_checkpoint(
    path: Path,
    model: RetfoundProtoNet,
    args: argparse.Namespace,
    *,
    epoch: int,
    best_qwk: float,
    saved_args: argparse.Namespace,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    state: Dict[str, Any] = {
        "method": "prototypical_network",
        "encoder_model": model.encoder.state_dict(),
        "projection": model.projection.state_dict(),
        "base_model_args": dict(vars(saved_args)),
        "few_shot_args": {
            key: os.fspath(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "epoch": epoch,
        "best_qwk": float(best_qwk),
        "parent_checkpoint": os.fspath(args.checkpoint.resolve()),
        "requires_support_set": True,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    args.output_dir = prepare_fresh_output_dir(args.output_dir)
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for RETFound episodic training")
    device = torch.device("cuda")
    amp_enabled = not args.no_amp

    bundle = load_grading_checkpoint(args.checkpoint, device, require_ce=True)
    encoder, saved_args = bundle.model, bundle.saved_args
    trainable_encoder = configure_encoder_trainability(
        encoder, args.unfreeze_last_blocks
    )
    print(f"Trainable encoder parameters: {trainable_encoder / 1e6:.2f}M")

    image_size = int(getattr(saved_args, "image_size", 224))
    frames, _ = load_split_frames(args.dataset_dir)
    train_transform, eval_transform = build_transforms(image_size)
    data_args = _dataset_args(saved_args, args)
    train_dataset = FundusDataset(frames["train"], data_args, train_transform)
    val_dataset = FundusDataset(frames["val"], data_args, eval_transform)
    print(f"Train images: {len(train_dataset):,}; validation images: {len(val_dataset):,}")
    print("Held-out test split is not loaded into an episode sampler.")

    model = RetfoundProtoNet(
        encoder,
        args.embedding_dim,
        temperature=args.temperature,
        forward_batch_size=args.forward_batch_size,
    ).to(device)
    encoder_parameters = [
        parameter for parameter in model.encoder.parameters() if parameter.requires_grad
    ]
    groups = [{"params": model.projection.parameters(), "lr": args.projection_lr}]
    if encoder_parameters:
        groups.append({"params": encoder_parameters, "lr": args.encoder_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scaler = create_scaler(amp_enabled)
    train_sampler = EpisodeSampler(frames["train"], args.seed)

    best_qwk = -1.0
    best_epoch = -1
    stale_epochs = 0
    history_path = args.output_dir / "history.jsonl"
    for epoch in range(args.epochs):
        model.train()
        epoch_losses: List[float] = []
        for _ in tqdm(range(args.train_episodes), desc="few-shot-train", leave=False):
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = run_episode(
                model,
                train_dataset,
                train_sampler.sample(args.shots, args.queries),
                device,
                amp_enabled=amp_enabled,
            )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(float(loss.detach()))

        val_loss, metrics = evaluate_episodes(
            model,
            val_dataset,
            frames["val"],
            episodes=args.val_episodes,
            shots=args.shots,
            queries=args.queries,
            seed=args.seed + 10_000,
            device=device,
            amp_enabled=amp_enabled,
        )
        record = {
            "epoch": epoch,
            "train_episode_loss": sum(epoch_losses) / len(epoch_losses),
            "val_episode_loss": val_loss,
            **metrics,
        }
        print(json.dumps(record, ensure_ascii=False))
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        current_qwk = float(metrics["qwk"])
        if not math.isfinite(current_qwk):
            raise RuntimeError("Few-shot validation produced a non-finite QWK")
        if current_qwk > best_qwk:
            best_qwk = current_qwk
            best_epoch = epoch
            stale_epochs = 0
            save_few_shot_checkpoint(
                args.output_dir / "checkpoint-best-protonet.pth",
                model,
                args,
                epoch=epoch,
                best_qwk=best_qwk,
                saved_args=saved_args,
            )
        else:
            stale_epochs += 1
        save_few_shot_checkpoint(
            args.output_dir / "checkpoint-last-protonet.pth",
            model,
            args,
            epoch=epoch,
            best_qwk=best_qwk,
            saved_args=saved_args,
            optimizer=optimizer,
        )
        if stale_epochs >= args.patience:
            print(f"Early stopping after {stale_epochs} epochs without QWK improvement")
            break

    summary = {
        "best_validation_qwk": best_qwk,
        "best_epoch": best_epoch,
        "shots": args.shots,
        "queries_per_class": args.queries,
        "test_split_used": False,
        "requires_support_set_at_inference": True,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
