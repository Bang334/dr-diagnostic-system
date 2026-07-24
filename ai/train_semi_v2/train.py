"""Train semi-supervised v2 from a canonical grading checkpoint.

Only the fixed training split and an external unlabeled image directory are
used for optimization. The validation split selects checkpoints. The test
split is evaluated only in explicit ``--eval-only`` mode after training.
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
    save_evaluation_artifacts,
)
from ai.train_semi_v2.runtime import (
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
from ai.train_semi_v2.pseudo_cache import PseudoLabelCache, PseudoLabelCacheSpec


_LOG_PATH: Optional[Path] = None


def configure_log_file(path: Optional[Path]) -> None:
    """Mirror structured progress messages to a durable UTF-8 log."""
    global _LOG_PATH
    _LOG_PATH = path
    if _LOG_PATH is not None:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    if _LOG_PATH is not None:
        with _LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


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


def should_log_progress(current: int, total: int, updates: int = 100) -> bool:
    if total <= 0:
        return current == 1
    interval = max(1, math.ceil(total / updates))
    return current == 1 or current == total or current % interval == 0


def limit_labeled_replay(
    frame: pd.DataFrame, *, max_per_class: int, seed: int
) -> pd.DataFrame:
    """Build a deterministic class-balanced rehearsal subset."""
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


def cap_pseudo_grade_zero(
    frame: pd.DataFrame, *, max_grade_zero: int
) -> pd.DataFrame:
    """Keep the most confident grade-0 predictions without limiting grades 1..4."""
    if max_grade_zero == 0 or frame.empty:
        return frame.reset_index(drop=True).copy()
    grade_zero = (
        frame.loc[frame["pseudo_label"] == 0]
        .sort_values("confidence", ascending=False)
        .head(max_grade_zero)
    )
    diseased = frame.loc[frame["pseudo_label"] != 0]
    return (
        pd.concat([grade_zero, diseased], ignore_index=True)
        .sort_values(["pseudo_label", "confidence"], ascending=[True, False])
        .reset_index(drop=True)
    )


def limit_unlabeled_paths(
    paths: List[Path], *, max_images: int, seed: int
) -> List[Path]:
    """Select a deterministic subset independent of filesystem ordering."""
    ordered = sorted((Path(path) for path in paths), key=os.fspath)
    if max_images == 0 or len(ordered) <= max_images:
        return ordered
    selected = random.Random(seed).sample(ordered, max_images)
    return sorted(selected, key=os.fspath)


def prepare_output_dir(output_dir: Path, resume: Optional[Path]) -> Path:
    """Allow a non-empty run directory only for a checkpoint inside that run."""
    output_dir = output_dir.expanduser().resolve()
    if resume is None:
        restartable = {
            "train.log",
            "replay.csv",
            "scan.csv",
            "cache.json",
            "pseudo.csv",
            "baseline.json",
            "best.pth",
            "history.jsonl",
        }
        if output_dir.is_dir():
            existing = {path.name for path in output_dir.iterdir()}
            if existing and existing.issubset(restartable):
                return output_dir
        return prepare_fresh_output_dir(output_dir)
    resume = resume.expanduser().resolve()
    if not resume.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
    if resume.parent != output_dir:
        raise ValueError(
            f"--resume must be inside --output-dir ({output_dir}); received {resume}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resume_training_state(
    state: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
) -> tuple[int, float, int, int]:
    required = {"epoch", "best_qwk", "optimizer", "scheduler"}
    missing = required.difference(state)
    if missing:
        raise ValueError(f"Resume checkpoint missing training state: {sorted(missing)}")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler_state = state.get("scaler")
    if scaler_state:
        scaler.load_state_dict(scaler_state)
    return (
        int(state["epoch"]) + 1,
        float(state["best_qwk"]),
        int(state.get("best_epoch", state["epoch"])),
        int(state.get("stale_epochs", 0)),
    )


def trim_history_for_resume(history_path: Path, start_epoch: int) -> None:
    """Keep one record per completed epoch and remove partial future records."""
    if not history_path.is_file():
        return
    records_by_epoch: Dict[int, Dict[str, Any]] = {}
    for raw_line in history_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        epoch = int(record["epoch"])
        if epoch < start_epoch:
            records_by_epoch[epoch] = record
    content = "".join(
        json.dumps(records_by_epoch[epoch], ensure_ascii=False) + "\n"
        for epoch in sorted(records_by_epoch)
    )
    history_path.write_text(content, encoding="utf-8")


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
    parser.add_argument(
        "--pseudo-cache-dir",
        type=Path,
        default=None,
        help=(
            "Persistent pseudo-label cache shared by runs. Matching teacher, "
            "unlabeled manifest, preprocessing and threshold skip inference."
        ),
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--head-lr", type=float, default=1e-5)
    parser.add_argument("--backbone-lr", type=float, default=1e-6)
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--pseudo-weight", type=float, default=0.25)
    parser.add_argument(
        "--max-unlabeled-images",
        type=int,
        default=0,
        help="Deterministically scan at most N unlabeled images; use 0 for all",
    )
    parser.add_argument(
        "--max-labeled-per-class",
        type=int,
        default=0,
        help="Replay at most N labeled train images per class; use 0 for all",
    )
    parser.add_argument(
        "--max-pseudo-per-class",
        type=int,
        default=0,
        help="Keep the most confident N images per class; use 0 for no cap",
    )
    parser.add_argument(
        "--max-pseudo-grade-zero",
        type=int,
        default=500,
        help=(
            "After loading or generating predictions, keep only the most confident "
            "N grade-0 pseudo-labels while retaining every grade 1..4; use 0 for all"
        ),
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Continue from checkpoint-last.pth in the same output directory",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate --resume checkpoint once on the held-out test split",
    )
    parser.add_argument("--enhance", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args(argv)
    if args.pseudo_cache_dir is None:
        args.pseudo_cache_dir = args.output_dir.parent / "cache"
    return args


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
    if args.max_pseudo_grade_zero < 0:
        raise ValueError("--max-pseudo-grade-zero cannot be negative")
    if args.max_unlabeled_images < 0:
        raise ValueError("--max-unlabeled-images cannot be negative")
    if args.max_labeled_per_class < 0:
        raise ValueError("--max-labeled-per-class cannot be negative")
    if args.eval_only and args.resume is None:
        raise ValueError("--eval-only requires --resume CHECKPOINT")
    output_checkpoint = (args.output_dir / "best.pth").resolve()
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
    progress_path: Optional[Path] = None,
) -> pd.DataFrame:
    model.eval()
    progress_columns = ["image_path", "pseudo_label", "confidence", "accepted"]
    progress_records: List[Dict[str, Any]] = []
    if progress_path is not None and progress_path.is_file():
        saved_progress = pd.read_csv(progress_path, on_bad_lines="skip")
        if set(progress_columns).issubset(saved_progress.columns):
            saved_progress = saved_progress.drop_duplicates("image_path", keep="last")
            progress_records = saved_progress[progress_columns].to_dict("records")
            log(f"Resuming pseudo-label inference: already processed={len(progress_records):,}.")
    processed_paths = {str(record["image_path"]) for record in progress_records}
    accepted_count = sum(
        str(record["accepted"]).strip().lower() == "true"
        for record in progress_records
    )
    total_images = len(loader.dataset)
    processed_images = len(processed_paths)
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
    progress.update(min(processed_images, total_images))
    try:
        for images, paths in loader:
            resolved_paths = [os.fspath(Path(path).resolve()) for path in paths]
            pending_indices = [
                index for index, path in enumerate(resolved_paths)
                if path not in processed_paths
            ]
            if not pending_indices:
                continue
            images = images[pending_indices].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                probabilities = torch.softmax(model(images), dim=1)
            confidence, labels = probabilities.max(dim=1)
            batch_records = []
            pending_paths = [resolved_paths[index] for index in pending_indices]
            for path, label, score in zip(
                pending_paths, labels.cpu(), confidence.cpu()
            ):
                accepted = float(score) >= threshold
                batch_records.append(
                    {
                        "image_path": path,
                        "pseudo_label": int(label),
                        "confidence": float(score),
                        "accepted": accepted,
                    }
                )
                processed_paths.add(path)
                processed_images += 1
                accepted_count += int(accepted)
                progress.update(1)
                progress.set_postfix(accepted=accepted_count, refresh=False)
                if should_log_progress(processed_images, total_images):
                    elapsed = format_duration(time.perf_counter() - started_at)
                    percent = 100.0 * processed_images / max(total_images, 1)
                    log(
                        "Pseudo-label progress: "
                        f"image {processed_images}/{total_images} ({percent:.1f}%), "
                        f"accepted_so_far={accepted_count:,}, elapsed={elapsed}, "
                        f"{gpu_memory_summary()}"
                    )
            progress_records.extend(batch_records)
            if progress_path is not None:
                pd.DataFrame(batch_records, columns=progress_columns).to_csv(
                    progress_path,
                    mode="a",
                    header=not progress_path.exists(),
                    index=False,
                )
    finally:
        progress.close()

    all_predictions = pd.DataFrame.from_records(
        progress_records, columns=progress_columns
    ).drop_duplicates("image_path", keep="last")
    accepted_values = all_predictions["accepted"]
    accepted_mask = (
        accepted_values
        if accepted_values.dtype == bool
        else accepted_values.astype(str).str.lower().eq("true")
    )
    frame = all_predictions.loc[
        accepted_mask,
        ["image_path", "pseudo_label", "confidence"],
    ].copy()
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
    total_samples = 0
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
            loss = (per_sample * sample_weights).mean()
        scaler.scale(loss / accum_steps).backward()
        should_step = (step + 1) % accum_steps == 0 or step + 1 == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_weighted_loss += float((per_sample.detach() * sample_weights).sum())
        total_samples += int(per_sample.numel())
        batch_number = step + 1
        if should_log_progress(batch_number, total_batches, updates=20):
            elapsed = format_duration(time.perf_counter() - started_at)
            average_loss = total_weighted_loss / max(total_samples, 1)
            log(
                f"Epoch {epoch_number} train: batch {batch_number}/{total_batches} "
                f"({100.0 * batch_number / max(total_batches, 1):.1f}%), "
                f"weighted_loss={average_loss:.6f}, "
                f"lr={optimizer.param_groups[0]['lr']:.2e}/"
                f"{optimizer.param_groups[-1]['lr']:.2e}, elapsed={elapsed}, "
                f"{gpu_memory_summary()}"
            )
    return total_weighted_loss / max(total_samples, 1)


def evaluate_held_out_test(
    args: argparse.Namespace,
    model: nn.Module,
    saved_args: argparse.Namespace,
    frames: Dict[str, pd.DataFrame],
    device: torch.device,
    amp_enabled: bool,
) -> Dict[str, Any]:
    """Evaluate once on test; never use these metrics for training or selection."""
    image_size = int(getattr(saved_args, "image_size", 224))
    _, eval_transform = build_transforms(image_size)
    data_args = _dataset_args(saved_args, args)
    test_dataset = FundusDataset(frames["test"], data_args, eval_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    started_at = time.perf_counter()
    log(
        f"Final held-out test: images={len(test_dataset):,}, "
        f"batches={len(test_loader):,}, checkpoint={args.resume}"
    )
    test_loss, targets, predictions, image_ids = evaluate(
        model, test_loader, criterion, device, "ce", amp_enabled
    )
    metrics = {"loss": test_loss, **calculate_metrics(targets, predictions)}
    save_evaluation_artifacts(
        args.output_dir, targets, predictions, image_ids, metrics
    )
    summary_path = args.output_dir / "summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else {}
    )
    summary.update(
        {
            "test_split_used_for_training": False,
            "test_split_used_for_model_selection": False,
            "test_split_evaluated_after_training": True,
            "test_checkpoint": os.fspath(args.resume.resolve()),
            "test_metrics": metrics,
        }
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log(
        f"Held-out test completed in "
        f"{format_duration(time.perf_counter() - started_at)}: "
        f"accuracy={metrics['accuracy']:.6f}, macro_f1={metrics['macro_f1']:.6f}, "
        f"qwk={metrics['qwk']:.6f}. Artifacts saved to {args.output_dir}."
    )
    return metrics


def run(args: argparse.Namespace) -> None:
    run_started_at = time.perf_counter()
    validate_args(args)
    args.output_dir = prepare_output_dir(args.output_dir, args.resume)
    configure_log_file(args.output_dir / "train.log")
    log("[1/7] Arguments validated and run directory prepared.")
    seed_everything(args.seed)

    log("[2/7] Checking CUDA availability...")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for RETFound semi-supervised training")
    device = torch.device("cuda")
    amp_enabled = not args.no_amp
    gpu = torch.cuda.get_device_properties(0)
    log(
        f"GPU={torch.cuda.get_device_name(0)}, "
        f"VRAM={gpu.total_memory / 1024**3:.2f} GiB, AMP={amp_enabled}"
    )

    checkpoint_to_load = args.resume or args.checkpoint
    checkpoint_started_at = time.perf_counter()
    log(
        f"[3/7] Loading {'resume' if args.resume else 'parent'} checkpoint: "
        f"{checkpoint_to_load} ({checkpoint_to_load.stat().st_size / 1024**3:.2f} GiB)"
    )
    bundle = load_grading_checkpoint(checkpoint_to_load, device, require_ce=True)
    model, saved_args = bundle.model, bundle.saved_args
    resume_state = bundle.state if args.resume else None
    image_size = int(getattr(saved_args, "image_size", 224))
    log(
        f"Checkpoint loaded in {format_duration(time.perf_counter() - checkpoint_started_at)}; "
        f"image_size={image_size}, {gpu_memory_summary()}"
    )

    log("[4/7] Loading fixed train/validation/test splits...")
    frames, _ = load_split_frames(args.dataset_dir)
    if args.eval_only:
        evaluate_held_out_test(
            args, model, saved_args, frames, device, amp_enabled
        )
        return

    train_transform, eval_transform = build_transforms(image_size)
    data_args = _dataset_args(saved_args, args)
    replay_manifest_path = args.output_dir / "replay.csv"
    unlabeled_manifest_path = args.output_dir / "scan.csv"
    pseudo_path = args.output_dir / "pseudo.csv"
    pseudo_cache_record_path = args.output_dir / "cache.json"
    baseline_path = args.output_dir / "baseline.json"

    if resume_state is not None:
        if not pseudo_path.is_file():
            raise FileNotFoundError(
                f"Cannot resume safely without saved pseudo-labels: {pseudo_path}"
            )
        pseudo_frame = pd.read_csv(pseudo_path)
        saved_run_args = resume_state.get("args", {}).get(
            "semi_supervised_args", {}
        )
        if replay_manifest_path.is_file():
            replay_frame = pd.read_csv(replay_manifest_path)
        else:
            replay_frame = limit_labeled_replay(
                frames["train"],
                max_per_class=int(saved_run_args.get("max_labeled_per_class", 0)),
                seed=int(saved_run_args.get("seed", args.seed)),
            )
            replay_frame.to_csv(replay_manifest_path, index=False)
            log("Legacy resume: reconstructed the labeled replay manifest.")
        if unlabeled_manifest_path.is_file():
            unlabeled_manifest = pd.read_csv(unlabeled_manifest_path)
            discovered_unlabeled_count = len(unlabeled_manifest)
        else:
            discovered_paths = discover_images(args.unlabeled_dir)
            discovered_unlabeled_count = len(discovered_paths)
            selected_paths = limit_unlabeled_paths(
                discovered_paths,
                max_images=int(saved_run_args.get("max_unlabeled_images", 0)),
                seed=int(saved_run_args.get("seed", args.seed)),
            )
            unlabeled_manifest = pd.DataFrame(
                {"image_path": [os.fspath(path.resolve()) for path in selected_paths]}
            )
            unlabeled_manifest.to_csv(unlabeled_manifest_path, index=False)
            log("Legacy resume: reconstructed the unlabeled scan manifest.")
        if baseline_path.is_file():
            baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        else:
            history_path = args.output_dir / "history.jsonl"
            if not history_path.is_file():
                raise FileNotFoundError(
                    "Legacy resume needs history.jsonl to recover baseline_val_loss"
                )
            first_record = json.loads(
                next(
                    line
                    for line in history_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            )
            baseline_payload = {
                "loss": float(first_record["baseline_val_loss"]),
                "qwk": None,
                "unlabeled_images_discovered": discovered_unlabeled_count,
                "unlabeled_images_scanned": len(unlabeled_manifest),
            }
            baseline_path.write_text(
                json.dumps(baseline_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            log("Legacy resume: recovered baseline loss from history.jsonl.")
        discovered_unlabeled_count = int(
            baseline_payload.get("unlabeled_images_discovered", len(unlabeled_manifest))
        )
        log(
            f"Resume artifacts loaded: labeled_replay={len(replay_frame):,}, "
            f"unlabeled_scan={len(unlabeled_manifest):,}, "
            f"pseudo_labels={len(pseudo_frame):,}. Pseudo-label inference is skipped."
        )
        unlabeled_loader = None
    else:
        discovered_unlabeled_paths = discover_images(args.unlabeled_dir)
        discovered_unlabeled_count = len(discovered_unlabeled_paths)
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
        replay_frame.to_csv(replay_manifest_path, index=False)
        pd.DataFrame(
            {"image_path": [os.fspath(path.resolve()) for path in unlabeled_paths]}
        ).to_csv(unlabeled_manifest_path, index=False)
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

    train_dataset = FundusDataset(replay_frame, data_args, train_transform)
    val_dataset = FundusDataset(frames["val"], data_args, eval_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    selected_unlabeled_count = (
        len(unlabeled_loader.dataset)
        if unlabeled_loader is not None
        else len(unlabeled_manifest)
    )
    log(
        f"Data prepared: labeled_discovered={len(frames['train']):,}, "
        f"labeled_replay={len(train_dataset):,} "
        f"{replay_frame['diagnosis'].value_counts().sort_index().to_dict()}, "
        f"validation={len(val_dataset):,}, test_held_out={len(frames['test']):,}, "
        f"unlabeled_discovered={discovered_unlabeled_count:,}, "
        f"unlabeled_selected={selected_unlabeled_count:,}."
    )

    if resume_state is None:
        pseudo_started_at = time.perf_counter()
        log(
            f"[5/7] Generating pseudo-labels: threshold={args.threshold}, "
            f"max_per_class={args.max_pseudo_per_class}, "
            f"images={selected_unlabeled_count:,}."
        )
        cache = (
            PseudoLabelCache(args.pseudo_cache_dir)
            if args.pseudo_cache_dir is not None
            else None
        )

        def predict_pseudo_labels() -> pd.DataFrame:
            return generate_pseudo_labels(
                model,
                unlabeled_loader,
                device,
                threshold=args.threshold,
                max_per_class=args.max_pseudo_per_class,
                amp_enabled=amp_enabled,
                progress_path=cache.progress_path if cache is not None else None,
            )

        if cache is not None:
            preprocessing = str(getattr(saved_args, "preprocessing", "rgb_crop"))
            cache_result = cache.load_or_generate(
                PseudoLabelCacheSpec(
                    threshold=args.threshold,
                    teacher_checkpoint=args.checkpoint,
                    unlabeled_paths=tuple(unlabeled_paths),
                    preprocessing=preprocessing,
                    image_size=image_size,
                    max_pseudo_per_class=args.max_pseudo_per_class,
                    enhance=args.enhance,
                ),
                predict_pseudo_labels,
            )
            pseudo_frame = cache_result.frame
            pseudo_cache_record_path.write_text(
                json.dumps(
                    {
                        "cache_key": cache_result.cache_key,
                        "reused": cache_result.reused,
                        "csv_path": os.fspath(cache_result.csv_path),
                        "metadata_path": os.fspath(cache_result.metadata_path),
                        "threshold": args.threshold,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            action = "reused" if cache_result.reused else "created"
            log(
                f"Pseudo-label cache {action}: key={cache_result.cache_key[:20]}, "
                f"path={cache_result.csv_path}."
            )
        else:
            pseudo_frame = predict_pseudo_labels()
        unfiltered_counts = (
            pseudo_frame["pseudo_label"].value_counts().sort_index().to_dict()
        )
        pseudo_frame = cap_pseudo_grade_zero(
            pseudo_frame, max_grade_zero=args.max_pseudo_grade_zero
        )
        if args.max_pseudo_grade_zero > 0:
            log(
                "Applied post-cache grade-0 cap without repeating inference: "
                f"max_grade_zero={args.max_pseudo_grade_zero}, "
                f"before={unfiltered_counts}, "
                f"after={pseudo_frame['pseudo_label'].value_counts().sort_index().to_dict()}."
            )
        if pseudo_frame.empty:
            raise RuntimeError(
                "No pseudo-label passed the confidence threshold; lower --threshold "
                "only after inspecting model calibration and the unlabeled domain"
            )
        pseudo_frame.to_csv(pseudo_path, index=False)
        log(
            f"Pseudo-labeling completed in "
            f"{format_duration(time.perf_counter() - pseudo_started_at)}; "
            f"accepted={len(pseudo_frame):,}, manifest={pseudo_path}."
        )
    else:
        log("[5/7] Reusing saved pseudo-labels; inference is not repeated.")
    counts = pseudo_frame["pseudo_label"].value_counts().sort_index().to_dict()
    log(f"Accepted pseudo-labels={len(pseudo_frame):,}, per_class={counts}.")

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
        f"Combined train set: labeled={len(train_dataset):,}, "
        f"pseudo={len(pseudo_dataset):,}, total={len(combined_dataset):,}, "
        f"batches_per_epoch={len(train_loader):,}."
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    if resume_state is None:
        baseline_started_at = time.perf_counter()
        log(f"[6/7] Evaluating parent checkpoint on {len(val_loader):,} validation batches...")
        baseline_loss, targets, predictions, _ = evaluate(
            model, val_loader, criterion, device, "ce", amp_enabled
        )
        baseline_metrics = calculate_metrics(targets, predictions)
        baseline_payload = {
            "loss": baseline_loss,
            **baseline_metrics,
            "labeled_train_images_discovered": len(frames["train"]),
            "labeled_replay_images": len(train_dataset),
            "unlabeled_images_discovered": discovered_unlabeled_count,
            "unlabeled_images_scanned": selected_unlabeled_count,
        }
        baseline_path.write_text(
            json.dumps(baseline_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        best_qwk = float(baseline_metrics["qwk"])
        if not math.isfinite(best_qwk):
            raise RuntimeError("Parent checkpoint produced a non-finite validation QWK")
        log(
            f"Parent validation completed in "
            f"{format_duration(time.perf_counter() - baseline_started_at)}: "
            f"loss={baseline_loss:.6f}, accuracy={baseline_metrics['accuracy']:.6f}, "
            f"qwk={best_qwk:.6f}."
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
        save_classifier_checkpoint(
            args.output_dir / "best.pth",
            model,
            checkpoint_args,
            epoch=-1,
            best_qwk=best_qwk,
            best_epoch=-1,
            stale_epochs=0,
            parent_checkpoint=args.checkpoint,
        )
        parent_checkpoint = args.checkpoint
    else:
        baseline_loss = float(baseline_payload["loss"])
        baseline_metrics = {
            key: value
            for key, value in baseline_payload.items()
            if key
            in {"accuracy", "macro_f1", "balanced_accuracy", "qwk", "per_class_recall"}
        }
        checkpoint_args = dict(resume_state["args"])
        parent_checkpoint = Path(
            resume_state.get("parent_checkpoint", args.checkpoint)
        )
        baseline_qwk = baseline_metrics.get("qwk")
        log(
            f"[6/7] Resume keeps parent baseline: loss={baseline_loss:.6f}, "
            f"qwk={float(baseline_qwk):.6f}."
            if baseline_qwk is not None
            else f"[6/7] Resume keeps parent baseline loss={baseline_loss:.6f}; "
            "parent QWK was not stored by the legacy run."
        )

    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs - 1, 1),
        eta_min=args.min_lr,
    )
    scaler = create_scaler(amp_enabled)
    history_path = args.output_dir / "history.jsonl"
    initial_qwk = baseline_metrics.get("qwk")
    start_epoch, best_qwk, best_epoch, stale_epochs = (
        0,
        float(initial_qwk) if initial_qwk is not None else -1.0,
        -1,
        0,
    )
    if resume_state is not None:
        start_epoch, best_qwk, best_epoch, stale_epochs = resume_training_state(
            resume_state, optimizer, scheduler, scaler
        )
        trim_history_for_resume(history_path, start_epoch)
        if "best_epoch" not in resume_state and history_path.is_file():
            completed_records = [
                json.loads(line)
                for line in history_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if completed_records:
                best_record = max(completed_records, key=lambda record: float(record["qwk"]))
                best_epoch = int(best_record["epoch"])
                stale_epochs = max(0, start_epoch - best_epoch - 1)
                log(
                    f"Legacy resume: recovered best_epoch={best_epoch + 1} and "
                    f"patience={stale_epochs} from history.jsonl."
                )
        log(
            f"Resume state restored: next_epoch={start_epoch + 1}/{args.epochs}, "
            f"best_epoch={best_epoch + 1}, best_qwk={best_qwk:.6f}, "
            f"patience={stale_epochs}/{args.patience}."
        )
        if stale_epochs >= args.patience:
            log("Run had already reached early stopping; no additional epoch is needed.")
            start_epoch = args.epochs

    if start_epoch >= args.epochs:
        log(f"[7/7] No remaining epoch: checkpoint already reached {start_epoch} epochs.")
    else:
        log(
            f"[7/7] Training from epoch {start_epoch + 1} to {args.epochs}; "
            f"patience={args.patience}, batches_per_epoch={len(train_loader):,}, "
            f"effective_batch_size={args.batch_size * args.accum_steps}."
        )
    for epoch in range(start_epoch, args.epochs):
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
        log(
            f"Epoch {epoch_number} train complete: loss={train_loss:.6f}, "
            f"elapsed={format_duration(time.perf_counter() - epoch_started_at)}. "
            f"Starting validation..."
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
            f"Epoch {epoch_number} validation complete in "
            f"{format_duration(time.perf_counter() - validation_started_at)}: "
            f"{json.dumps(record, ensure_ascii=False)}"
        )
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
            best_path = args.output_dir / "best.pth"
            log(f"QWK improved to {best_qwk:.6f}; saving {best_path}...")
            save_classifier_checkpoint(
                best_path,
                model,
                checkpoint_args,
                epoch=epoch,
                best_qwk=best_qwk,
                best_epoch=best_epoch,
                stale_epochs=stale_epochs,
                parent_checkpoint=parent_checkpoint,
            )
        else:
            stale_epochs += 1
            log(
                f"QWK did not improve ({current_qwk:.6f} <= {best_qwk:.6f}); "
                f"patience={stale_epochs}/{args.patience}."
            )
        scheduler.step()
        last_path = args.output_dir / "last.pth"
        checkpoint_started_at = time.perf_counter()
        log(f"Saving resumable state to {last_path}...")
        checkpoint_size = save_classifier_checkpoint(
            last_path,
            model,
            checkpoint_args,
            epoch=epoch,
            best_qwk=best_qwk,
            best_epoch=best_epoch,
            stale_epochs=stale_epochs,
            parent_checkpoint=parent_checkpoint,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        log(
            f"Resume checkpoint saved in "
            f"{format_duration(time.perf_counter() - checkpoint_started_at)} "
            f"({checkpoint_size / 1024**3:.2f} GiB)."
        )
        if stale_epochs >= args.patience:
            log(f"Early stopping after {stale_epochs} epochs without QWK improvement.")
            break

    summary = {
        "parent_qwk": (
            float(baseline_metrics["qwk"])
            if baseline_metrics.get("qwk") is not None
            else None
        ),
        "best_qwk": best_qwk,
        "best_epoch": best_epoch,
        "labeled_train_images_discovered": len(frames["train"]),
        "labeled_replay_images": len(train_dataset),
        "unlabeled_images_discovered": discovered_unlabeled_count,
        "unlabeled_images_scanned": selected_unlabeled_count,
        "accepted_pseudo_labels": len(pseudo_frame),
        "pseudo_labels_per_class": {str(k): int(v) for k, v in counts.items()},
        "mean_pseudo_confidence": float(pseudo_frame["confidence"].mean()),
        "test_split_used_for_training": False,
        "test_split_used_for_model_selection": False,
        "test_split_evaluated_after_training": False,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    log(
        f"Training run finished in {format_duration(time.perf_counter() - run_started_at)}. "
        f"Run --eval-only with best.pth for the final held-out test."
    )
    log("Summary:\n" + json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    try:
        run(parse_args())
    except KeyboardInterrupt:
        log(
            "Training interrupted by user. last.pth from the most "
            "recent fully completed epoch remains safe; use --resume to continue."
        )
        raise SystemExit(130)


if __name__ == "__main__":
    main()
