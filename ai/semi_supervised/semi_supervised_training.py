"""Continue a RETFound grading checkpoint with confidence-filtered pseudo-labels.

Only the fixed training split and an external unlabeled image directory are
used for optimization. The validation split selects checkpoints; the test
split is discovered only to enforce separation and is never evaluated here.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import sys
import time
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


def log(message: str) -> None:
    """Emit a timestamped line immediately, including through Colab subprocesses."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def gpu_memory_summary() -> str:
    if not torch.cuda.is_available():
        return "GPU memory unavailable"
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    return f"GPU memory allocated={allocated:.2f} GiB, reserved={reserved:.2f} GiB"


def should_log_progress(current: int, total: int, updates: int = 20) -> bool:
    if total <= 0:
        return current == 1
    interval = max(1, math.ceil(total / updates))
    return current == 1 or current == total or current % interval == 0


def limit_labeled_replay(
    frame: pd.DataFrame, *, max_per_class: int, seed: int
) -> pd.DataFrame:
    """Build a deterministic, class-balanced rehearsal subset."""
    if max_per_class == 0:
        return frame.reset_index(drop=True).copy()
    selected = []
    for label, class_frame in frame.groupby("diagnosis", sort=True):
        selected.append(
            class_frame.sample(
                n=min(max_per_class, len(class_frame)),
                random_state=seed + int(label),
            )
        )
    if not selected:
        return frame.iloc[0:0].copy()
    return (
        pd.concat(selected, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def limit_unlabeled_paths(
    paths: List[Path], *, max_images: int, seed: int
) -> List[Path]:
    """Select a deterministic subset without depending on filesystem order."""
    ordered = sorted((Path(path) for path in paths), key=lambda path: os.fspath(path))
    if max_images == 0 or len(ordered) <= max_images:
        return ordered
    selected = random.Random(seed).sample(ordered, max_images)
    return sorted(selected, key=lambda path: os.fspath(path))


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
        "--max-unlabeled-images",
        type=int,
        default=20_000,
        help="Deterministically scan at most N unlabeled images; use 0 for all",
    )
    parser.add_argument(
        "--max-labeled-per-class",
        type=int,
        default=1_000,
        help="Replay at most N labeled train images per class; use 0 for all",
    )
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
    if args.max_unlabeled_images < 0:
        raise ValueError("--max-unlabeled-images cannot be negative")
    if args.max_labeled_per_class < 0:
        raise ValueError("--max-labeled-per-class cannot be negative")
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
    total_images = len(loader.dataset)
    processed_images = 0
    started_at = time.perf_counter()
    progress = tqdm(
        total=total_images,
        desc="pseudo-label images",
        unit="image",
        mininterval=0.5,
        dynamic_ncols=True,
        file=sys.stdout,
        leave=True,
    )
    try:
        for images, paths in loader:
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
                processed_images += 1
                progress.update(1)
                progress.set_postfix(accepted=len(records), refresh=False)
                if should_log_progress(processed_images, total_images, updates=100):
                    elapsed = format_duration(time.perf_counter() - started_at)
                    percent = 100.0 * processed_images / max(total_images, 1)
                    log(
                        "Pseudo-label progress: "
                        f"image {processed_images}/{total_images} ({percent:.1f}%), "
                        f"accepted_so_far={len(records):,}, elapsed={elapsed}, "
                        f"{gpu_memory_summary()}"
                    )
    finally:
        progress.close()

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
    epoch_number: int,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_weighted_loss = 0.0
    total_weight = 0.0
    total_batches = len(loader)
    started_at = time.perf_counter()
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
        batch_number = step + 1
        if should_log_progress(batch_number, total_batches):
            elapsed = format_duration(time.perf_counter() - started_at)
            average_loss = total_weighted_loss / max(total_weight, 1e-8)
            backbone_lr = optimizer.param_groups[0]["lr"]
            head_lr = optimizer.param_groups[1]["lr"]
            log(
                f"Epoch {epoch_number} train progress: "
                f"batch {batch_number}/{total_batches} "
                f"({100.0 * batch_number / max(total_batches, 1):.1f}%), "
                f"weighted_loss={average_loss:.6f}, "
                f"lr(backbone/head)={backbone_lr:.2e}/{head_lr:.2e}, "
                f"elapsed={elapsed}, {gpu_memory_summary()}"
            )
    return total_weighted_loss / max(total_weight, 1e-8)


def run(args: argparse.Namespace) -> None:
    run_started_at = time.perf_counter()
    log("[1/7] Validating arguments and preparing output directory...")
    log(
        "Configuration: "
        + json.dumps(
            {
                key: os.fspath(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    validate_args(args)
    args.output_dir = prepare_fresh_output_dir(args.output_dir)
    seed_everything(args.seed)
    log(f"Output directory ready: {args.output_dir}")

    log("[2/7] Checking CUDA availability...")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for RETFound semi-supervised training")
    device = torch.device("cuda")
    amp_enabled = not args.no_amp
    gpu_properties = torch.cuda.get_device_properties(0)
    log(
        f"GPU: {torch.cuda.get_device_name(0)}, "
        f"VRAM={gpu_properties.total_memory / 1024**3:.2f} GiB, AMP={amp_enabled}"
    )

    checkpoint_size = args.checkpoint.expanduser().stat().st_size / 1024**3
    checkpoint_started_at = time.perf_counter()
    log(
        f"[3/7] Loading checkpoint: {args.checkpoint} "
        f"({checkpoint_size:.2f} GiB)"
    )
    bundle = load_grading_checkpoint(args.checkpoint, device, require_ce=True)
    model, saved_args = bundle.model, bundle.saved_args
    image_size = int(getattr(saved_args, "image_size", 224))
    log(
        f"Checkpoint loaded in {format_duration(time.perf_counter() - checkpoint_started_at)}; "
        f"image_size={image_size}, {gpu_memory_summary()}"
    )

    dataset_started_at = time.perf_counter()
    log("[4/7] Loading dataset splits and discovering unlabeled images...")
    frames, _ = load_split_frames(args.dataset_dir)
    discovered_unlabeled_paths = discover_images(args.unlabeled_dir)
    unlabeled_paths = limit_unlabeled_paths(
        discovered_unlabeled_paths,
        max_images=args.max_unlabeled_images,
        seed=args.seed,
    )
    replay_frame = limit_labeled_replay(
        frames["train"],
        max_per_class=args.max_labeled_per_class,
        seed=args.seed,
    )
    assert_unlabeled_is_external(unlabeled_paths, frames)
    replay_manifest_path = args.output_dir / "labeled_replay_manifest.csv"
    unlabeled_manifest_path = args.output_dir / "unlabeled_scan_manifest.csv"
    replay_frame.to_csv(replay_manifest_path, index=False)
    pd.DataFrame(
        {"image_path": [os.fspath(path.resolve()) for path in unlabeled_paths]}
    ).to_csv(unlabeled_manifest_path, index=False)
    train_transform, eval_transform = build_transforms(image_size)
    data_args = _dataset_args(saved_args, args)

    train_dataset = FundusDataset(replay_frame, data_args, train_transform)
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

    log(
        "Dataset ready in "
        f"{format_duration(time.perf_counter() - dataset_started_at)}: "
        f"labeled_discovered={len(frames['train']):,}, "
        f"labeled_replay={len(train_dataset):,} "
        f"({replay_frame['diagnosis'].value_counts().sort_index().to_dict()}), "
        f"validation={len(val_dataset):,} images/{len(val_loader):,} batches, "
        f"unlabeled_discovered={len(discovered_unlabeled_paths):,}, "
        f"unlabeled_selected={len(unlabeled_dataset):,} images/{len(unlabeled_loader):,} batches"
    )
    log(
        f"Sampling manifests saved: labeled={replay_manifest_path}, "
        f"unlabeled={unlabeled_manifest_path}"
    )
    log("Held-out test split is not loaded into a DataLoader.")

    pseudo_started_at = time.perf_counter()
    log(
        f"[5/7] Generating pseudo-labels: threshold={args.threshold}, "
        f"max_per_class={args.max_pseudo_per_class}, batches={len(unlabeled_loader):,}"
    )
    log(
        f"Running inference on {len(unlabeled_dataset):,} unlabeled images; "
        "this can take from minutes to hours depending on dataset size."
    )
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
    log(
        f"Pseudo-labeling completed in {format_duration(time.perf_counter() - pseudo_started_at)}; "
        f"accepted={len(pseudo_frame):,}, per_class={counts}, "
        f"csv={args.output_dir / 'pseudo_labels.csv'}"
    )

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

    log(
        f"Combined training set: labeled={len(train_dataset):,}, "
        f"pseudo={len(pseudo_dataset):,}, total={len(combined_dataset):,}, "
        f"batches_per_epoch={len(train_loader):,}"
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    baseline_started_at = time.perf_counter()
    log(
        f"[6/7] Evaluating parent checkpoint on validation set "
        f"({len(val_loader):,} batches)..."
    )
    baseline_loss, targets, predictions, _ = evaluate(
        model, val_loader, criterion, device, "ce", amp_enabled
    )
    baseline_metrics = calculate_metrics(targets, predictions)
    best_qwk = float(baseline_metrics["qwk"])
    if not math.isfinite(best_qwk):
        raise RuntimeError("Parent checkpoint produced a non-finite validation QWK")
    log(
        f"Parent validation completed in "
        f"{format_duration(time.perf_counter() - baseline_started_at)}: "
        f"loss={baseline_loss:.6f}, accuracy={baseline_metrics['accuracy']:.6f}, "
        f"macro_f1={baseline_metrics['macro_f1']:.6f}, qwk={best_qwk:.6f}"
    )

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
    initial_checkpoint_path = args.output_dir / "checkpoint-best.pth"
    checkpoint_save_started_at = time.perf_counter()
    log(f"Saving parent baseline checkpoint to {initial_checkpoint_path}...")
    save_classifier_checkpoint(
        initial_checkpoint_path,
        model,
        checkpoint_args,
        epoch=-1,
        best_qwk=best_qwk,
        parent_checkpoint=args.checkpoint,
    )
    log(
        f"Baseline checkpoint saved in "
        f"{format_duration(time.perf_counter() - checkpoint_save_started_at)} "
        f"({initial_checkpoint_path.stat().st_size / 1024**3:.2f} GiB)."
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

    log(
        f"[7/7] Starting semi-supervised training: epochs={args.epochs}, "
        f"patience={args.patience}, batches_per_epoch={len(train_loader):,}, "
        f"effective_batch_size={args.batch_size * args.accum_steps}"
    )
    for epoch in range(args.epochs):
        epoch_number = epoch + 1
        epoch_started_at = time.perf_counter()
        log(f"--- Epoch {epoch_number}/{args.epochs} started ---")
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            accum_steps=args.accum_steps,
            amp_enabled=amp_enabled,
            epoch_number=epoch_number,
        )
        train_duration = time.perf_counter() - epoch_started_at
        log(
            f"Epoch {epoch_number} training completed in {format_duration(train_duration)}; "
            f"train_loss={train_loss:.6f}. Starting validation ({len(val_loader):,} batches)..."
        )
        validation_started_at = time.perf_counter()
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
        log(
            f"Epoch {epoch_number} validation completed in "
            f"{format_duration(time.perf_counter() - validation_started_at)}; "
            f"metrics={json.dumps(record, ensure_ascii=False)}"
        )
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        log(f"Epoch {epoch_number} history appended to {history_path}.")

        current_qwk = float(metrics["qwk"])
        if not math.isfinite(current_qwk):
            raise RuntimeError("Semi-supervised model produced a non-finite validation QWK")
        improved = current_qwk > best_qwk
        if improved:
            best_qwk = current_qwk
            best_epoch = epoch
            stale_epochs = 0
            best_checkpoint_path = args.output_dir / "checkpoint-best.pth"
            best_save_started_at = time.perf_counter()
            log(
                f"Epoch {epoch_number} improved QWK to {best_qwk:.6f}; "
                f"saving best checkpoint to {best_checkpoint_path}..."
            )
            save_classifier_checkpoint(
                best_checkpoint_path,
                model,
                checkpoint_args,
                epoch=epoch,
                best_qwk=best_qwk,
                parent_checkpoint=args.checkpoint,
            )
            log(
                f"Best checkpoint saved in "
                f"{format_duration(time.perf_counter() - best_save_started_at)} "
                f"({best_checkpoint_path.stat().st_size / 1024**3:.2f} GiB)."
            )
        else:
            stale_epochs += 1
            log(
                f"Epoch {epoch_number} did not improve QWK "
                f"({current_qwk:.6f} <= {best_qwk:.6f}); "
                f"patience={stale_epochs}/{args.patience}."
            )
        last_checkpoint_path = args.output_dir / "checkpoint-last.pth"
        last_save_started_at = time.perf_counter()
        log(f"Saving resumable checkpoint to {last_checkpoint_path}...")
        save_classifier_checkpoint(
            last_checkpoint_path,
            model,
            checkpoint_args,
            epoch=epoch,
            best_qwk=best_qwk,
            parent_checkpoint=args.checkpoint,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        log(
            f"Last checkpoint saved in "
            f"{format_duration(time.perf_counter() - last_save_started_at)} "
            f"({last_checkpoint_path.stat().st_size / 1024**3:.2f} GiB)."
        )
        scheduler.step()
        log(
            f"--- Epoch {epoch_number}/{args.epochs} finished in "
            f"{format_duration(time.perf_counter() - epoch_started_at)}; "
            f"best_qwk={best_qwk:.6f} ---"
        )
        if stale_epochs >= args.patience:
            log(f"Early stopping after {stale_epochs} epochs without QWK improvement.")
            break

    summary = {
        "parent_qwk": float(baseline_metrics["qwk"]),
        "best_qwk": best_qwk,
        "best_epoch": best_epoch,
        "labeled_train_images_discovered": len(frames["train"]),
        "labeled_replay_images": len(train_dataset),
        "unlabeled_images_discovered": len(discovered_unlabeled_paths),
        "unlabeled_images_scanned": len(unlabeled_dataset),
        "accepted_pseudo_labels": len(pseudo_frame),
        "pseudo_labels_per_class": {str(k): int(v) for k, v in counts.items()},
        "mean_pseudo_confidence": float(pseudo_frame["confidence"].mean()),
        "test_split_used": False,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    log(f"Summary saved to {args.output_dir / 'summary.json'}.")
    log(
        f"Semi-supervised run completed in "
        f"{format_duration(time.perf_counter() - run_started_at)}."
    )
    log("Final summary:\n" + json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
