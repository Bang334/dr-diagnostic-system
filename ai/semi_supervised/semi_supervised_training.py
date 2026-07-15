"""Continue a RETFound grading checkpoint with confidence-filtered pseudo-labels.

Only the fixed training split and an external unlabeled image directory are
used for optimization. The validation split selects checkpoints; the test
split is discovered only to enforce separation and is never evaluated here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm

from ai.grading.train import (
    FundusDataset,
    build_transforms,
    calculate_metrics,
    create_scaler,
    evaluate,
    is_head_parameter,
)
from ai.semi_supervised.research_utils import (
    PseudoLabeledFundusDataset,
    UnlabeledFundusDataset,
    WeightedLabeledDataset,
    assert_unlabeled_is_external,
    checkpoint_args_with_metadata,
    discover_images,
    load_grading_checkpoint,
    load_split_frames,
    prepare_fresh_output_dir,
    save_classifier_checkpoint,
    seed_everything,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=Path,
        help="Labeled dataset containing fixed train/val/test directories",
    )
    parser.add_argument(
        "--unlabeled-dir",
        required=True,
        type=Path,
        help="External directory containing genuinely unlabeled fundus images",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=8)
    parser.add_argument("--head-lr", type=float, default=1e-5)
    parser.add_argument("--backbone-lr", type=float, default=1e-6)
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--pseudo-weight", type=float, default=0.25)
    parser.add_argument(
        "--max-pseudo-per-class",
        type=int,
        default=2000,
        help="Keep the most confident N images per class; use 0 for no cap",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enhance", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not 0.5 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0.5 and 1.0")
    if not 0.0 < args.pseudo_weight <= 1.0:
        raise ValueError("--pseudo-weight must be in (0, 1]")
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("--epochs and --patience must be positive")
    if args.batch_size < 1 or args.accum_steps < 1:
        raise ValueError("--batch-size and --accum-steps must be positive")
    if args.max_pseudo_per_class < 0:
        raise ValueError("--max-pseudo-per-class cannot be negative")
    output_checkpoint = (args.output_dir / "checkpoint-best.pth").resolve()
    if output_checkpoint == args.checkpoint.expanduser().resolve():
        raise ValueError(
            "--output-dir would overwrite the parent checkpoint; choose a new run directory"
        )


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


@torch.no_grad()
def generate_pseudo_labels(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    threshold: float,
    max_per_class: int,
    amp_enabled: bool,
) -> pd.DataFrame:
    model.eval()
    records: List[Dict[str, Any]] = []
    for images, paths in tqdm(loader, desc="pseudo-label", leave=False):
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            probabilities = torch.softmax(model(images), dim=1)
        confidence, labels = probabilities.max(dim=1)
        for path, label, score in zip(paths, labels.cpu(), confidence.cpu()):
            if float(score) >= threshold:
                records.append(
                    {
                        "image_path": os.fspath(Path(path).resolve()),
                        "pseudo_label": int(label),
                        "confidence": float(score),
                    }
                )

    frame = pd.DataFrame.from_records(
        records, columns=["image_path", "pseudo_label", "confidence"]
    )
    if frame.empty:
        return frame
    frame = frame.sort_values("confidence", ascending=False)
    if max_per_class > 0:
        frame = frame.groupby("pseudo_label", group_keys=False).head(max_per_class)
    return frame.sort_values(["pseudo_label", "confidence"], ascending=[True, False]).reset_index(
        drop=True
    )


def build_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    head, backbone = [], []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = True
        (head if is_head_parameter(name) else backbone).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": args.backbone_lr},
            {"params": head, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    *,
    accum_steps: int,
    amp_enabled: bool,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_weighted_loss = 0.0
    total_weight = 0.0
    for step, (images, labels, _, sample_weights) in enumerate(
        tqdm(loader, desc="semi-train", leave=False)
    ):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        sample_weights = sample_weights.to(device, dtype=torch.float32, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(images)
            per_sample = F.cross_entropy(
                logits,
                labels,
                reduction="none",
                label_smoothing=0.05,
            )
            loss = (per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)
        scaler.scale(loss / accum_steps).backward()
        should_step = (step + 1) % accum_steps == 0 or step + 1 == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_weighted_loss += float((per_sample.detach() * sample_weights).sum())
        total_weight += float(sample_weights.sum())
    return total_weighted_loss / max(total_weight, 1e-8)


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    args.output_dir = prepare_fresh_output_dir(args.output_dir)
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for RETFound semi-supervised training")
    device = torch.device("cuda")
    amp_enabled = not args.no_amp

    bundle = load_grading_checkpoint(args.checkpoint, device, require_ce=True)
    model, saved_args = bundle.model, bundle.saved_args
    image_size = int(getattr(saved_args, "image_size", 224))
    frames, _ = load_split_frames(args.dataset_dir)
    unlabeled_paths = discover_images(args.unlabeled_dir)
    assert_unlabeled_is_external(unlabeled_paths, frames)
    train_transform, eval_transform = build_transforms(image_size)
    data_args = _dataset_args(saved_args, args)

    train_dataset = FundusDataset(frames["train"], data_args, train_transform)
    val_dataset = FundusDataset(frames["val"], data_args, eval_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    unlabeled_dataset = UnlabeledFundusDataset(
        unlabeled_paths,
        image_size,
        eval_transform,
        enhance=args.enhance,
    )
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Labeled train images: {len(train_dataset):,}")
    print(f"Validation images: {len(val_dataset):,}")
    print(f"External unlabeled images: {len(unlabeled_dataset):,}")
    print("Held-out test split is not loaded into a DataLoader.")

    pseudo_frame = generate_pseudo_labels(
        model,
        unlabeled_loader,
        device,
        threshold=args.threshold,
        max_per_class=args.max_pseudo_per_class,
        amp_enabled=amp_enabled,
    )
    if pseudo_frame.empty:
        raise RuntimeError(
            "No pseudo-label passed the confidence threshold; lower --threshold only "
            "after inspecting model calibration and the unlabeled domain"
        )
    pseudo_frame.to_csv(args.output_dir / "pseudo_labels.csv", index=False)
    counts = pseudo_frame["pseudo_label"].value_counts().sort_index().to_dict()
    print(f"Accepted pseudo-labels: {len(pseudo_frame):,}; per class: {counts}")

    pseudo_dataset = PseudoLabeledFundusDataset(
        pseudo_frame,
        image_size,
        train_transform,
        pseudo_weight=args.pseudo_weight,
        enhance=args.enhance,
    )
    combined_dataset = ConcatDataset(
        [WeightedLabeledDataset(train_dataset), pseudo_dataset]
    )
    train_loader = DataLoader(
        combined_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    baseline_loss, targets, predictions, _ = evaluate(
        model, val_loader, criterion, device, "ce", amp_enabled
    )
    baseline_metrics = calculate_metrics(targets, predictions)
    best_qwk = float(baseline_metrics["qwk"])
    if not math.isfinite(best_qwk):
        raise RuntimeError("Parent checkpoint produced a non-finite validation QWK")
    print(f"Parent checkpoint validation QWK: {best_qwk:.6f}")

    metadata = {
        "research_method": "pseudo_labeling",
        "parent_checkpoint": os.fspath(args.checkpoint.resolve()),
        "pseudo_threshold": args.threshold,
        "pseudo_weight": args.pseudo_weight,
        "unlabeled_dir": os.fspath(args.unlabeled_dir.resolve()),
        "semi_supervised_args": {
            key: os.fspath(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    checkpoint_args = checkpoint_args_with_metadata(saved_args, metadata)
    save_classifier_checkpoint(
        args.output_dir / "checkpoint-best.pth",
        model,
        checkpoint_args,
        epoch=-1,
        best_qwk=best_qwk,
        parent_checkpoint=args.checkpoint,
    )

    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs - 1, 1),
        eta_min=args.min_lr,
    )
    scaler = create_scaler(amp_enabled)
    history_path = args.output_dir / "history.jsonl"
    stale_epochs = 0
    best_epoch = -1

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            accum_steps=args.accum_steps,
            amp_enabled=amp_enabled,
        )
        val_loss, targets, predictions, _ = evaluate(
            model, val_loader, criterion, device, "ce", amp_enabled
        )
        metrics = calculate_metrics(targets, predictions)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "baseline_val_loss": baseline_loss,
            **metrics,
        }
        print(json.dumps(record, ensure_ascii=False))
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        current_qwk = float(metrics["qwk"])
        if not math.isfinite(current_qwk):
            raise RuntimeError("Semi-supervised model produced a non-finite validation QWK")
        improved = current_qwk > best_qwk
        if improved:
            best_qwk = current_qwk
            best_epoch = epoch
            stale_epochs = 0
            save_classifier_checkpoint(
                args.output_dir / "checkpoint-best.pth",
                model,
                checkpoint_args,
                epoch=epoch,
                best_qwk=best_qwk,
                parent_checkpoint=args.checkpoint,
            )
        else:
            stale_epochs += 1
        save_classifier_checkpoint(
            args.output_dir / "checkpoint-last.pth",
            model,
            checkpoint_args,
            epoch=epoch,
            best_qwk=best_qwk,
            parent_checkpoint=args.checkpoint,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        scheduler.step()
        if stale_epochs >= args.patience:
            print(f"Early stopping after {stale_epochs} epochs without QWK improvement")
            break

    summary = {
        "parent_qwk": float(baseline_metrics["qwk"]),
        "best_qwk": best_qwk,
        "best_epoch": best_epoch,
        "accepted_pseudo_labels": len(pseudo_frame),
        "pseudo_labels_per_class": {str(k): int(v) for k, v in counts.items()},
        "mean_pseudo_confidence": float(pseudo_frame["confidence"].mean()),
        "test_split_used": False,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
