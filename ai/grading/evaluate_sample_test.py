"""Evaluate a saved RETFound checkpoint on a limited sample from the test split.

Logs predictions for each sample and prints summary metrics at the end.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ai.grading.backbones import BACKBONE_PRESETS

# Ensure stdout is flushed immediately for subprocess environments
sys.stdout.reconfigure(line_buffering=True)

from ai.grading.train import (
    CLASS_NAMES,
    NUM_CLASSES,
    FundusDataset,
    build_transforms,
    calculate_metrics,
    find_predefined_splits,
    predictions_from_logits,
    scan_classification_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint-best.pth",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Root of the fundus dataset containing predefined train/val/test splits",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=30,
        help="Number of samples to evaluate per class (default: 30)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision (AMP)",
    )
    return parser.parse_args()


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, argparse.Namespace]:
    print(f"Loading checkpoint from: {checkpoint_path}")

    # Fix PosixPath issue when loading checkpoint on Windows
    import pathlib

    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    finally:
        pathlib.PosixPath = temp

    saved_args = argparse.Namespace(**state["args"])

    import timm

    output_dim = NUM_CLASSES if saved_args.loss == "ce" else NUM_CLASSES - 1
    if saved_args.model_source == "retfound":
        model = timm.create_model(
            "vit_large_patch14_dinov2.lvd142m",
            pretrained=False,
            img_size=saved_args.image_size,
            num_classes=output_dim,
        )
    else:
        model = timm.create_model(
            saved_args.model_name
            or BACKBONE_PRESETS[getattr(saved_args, "architecture", "convnext")],
            pretrained=False,
            num_classes=output_dim,
        )

    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()

    epoch = state.get("epoch", "?")
    best_qwk = state.get("best_qwk", float("nan"))
    print(f"Loaded checkpoint at epoch {epoch} (Best validation QWK: {best_qwk:.4f})")
    return model, saved_args


def get_sampled_test_df(dataset_dir: Path, limit_per_class: int) -> pd.DataFrame:
    split_paths = find_predefined_splits(dataset_dir)
    dataset_root = next(iter(split_paths.values())).parent
    test_df = scan_classification_split(split_paths["test"], dataset_root)

    # Stratified sampling of N samples per class
    sampled_df = (
        test_df.groupby("diagnosis")
        .apply(lambda x: x.sample(n=min(len(x), limit_per_class), random_state=42))
        .reset_index(drop=True)
    )

    counts = sampled_df["diagnosis"].value_counts().sort_index().to_dict()
    print(f"Sampled test split: {len(sampled_df)} total images (limit per class = {limit_per_class})")
    for grade, count in counts.items():
        print(f"  Class {grade} ({CLASS_NAMES[grade]}): {count} images")
    return sampled_df


@torch.no_grad()
def evaluate_samples(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    loss_name: str,
    amp_enabled: bool,
) -> tuple[float, list[int], list[int], list[float], list[str]]:
    model.eval()
    total_loss = 0.0
    sample_count = 0

    all_targets: list[int] = []
    all_predictions: list[int] = []
    all_confidences: list[float] = []
    all_ids: list[str] = []

    for images, targets, image_ids in tqdm(loader, desc="Evaluating samples", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)

        predictions = predictions_from_logits(logits, loss_name)

        # Calculate confidence scores (softmax probability for the predicted class)
        if loss_name == "coral":
            # For CORAL, logits represent cumulative logits
            probs = torch.sigmoid(logits)
            # Probability for class k is computed from difference between cumulative probabilities
            # For simplicity, we fallback to softmax over ordinal probability estimations
            # or treat logits as direct class probabilities after softmax
            probs_extended = torch.softmax(logits, dim=1)
            confidences, _ = probs_extended.max(dim=1)
        else:
            probs = torch.softmax(logits, dim=1)
            confidences, _ = probs.max(dim=1)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        sample_count += batch_size

        all_targets.extend(targets.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())
        all_confidences.extend(confidences.cpu().tolist())
        all_ids.extend(image_ids)

    avg_loss = total_loss / max(sample_count, 1)
    return avg_loss, all_targets, all_predictions, all_confidences, all_ids


def main() -> None:
    args = parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {torch.cuda.get_device_name(0)} ({mem_gb:.1f} GB)")

    amp_enabled = device.type == "cuda" and not args.no_amp

    # Load model and parameters
    model, saved_args = load_checkpoint(args.checkpoint, device)

    # Ensure metadata configuration matches training setup
    saved_args.enhance = getattr(saved_args, "enhance", False)
    saved_args.preprocessing = getattr(saved_args, "preprocessing", "rgb_crop")
    saved_args.image_size = getattr(saved_args, "image_size", 224)

    # Get test data subset
    test_df = get_sampled_test_df(args.dataset_dir, args.limit_per_class)
    _, eval_transform = build_transforms(saved_args.image_size)

    test_dataset = FundusDataset(test_df, saved_args, eval_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Standard CrossEntropyLoss for evaluation
    criterion = nn.CrossEntropyLoss()

    print("\n--- Running detailed evaluation on sampled images ---")
    avg_loss, targets, predictions, confidences, image_ids = evaluate_samples(
        model,
        test_loader,
        criterion,
        device,
        saved_args.loss,
        amp_enabled,
    )

    print("\n" + "=" * 80)
    print(f"{'DETAILED SAMPLE PREDICTIONS':^80}")
    print("=" * 80)
    print(f"{'ID / Image Name':<30} | {'True Label':<15} | {'Predicted Label':<15} | {'Confidence':<10} | {'Status'}")
    print("-" * 80)

    correct_count = 0
    for idx, (img_id, true_lbl, pred_lbl, conf) in enumerate(zip(image_ids, targets, predictions, confidences)):
        status = "✅ OK" if true_lbl == pred_lbl else "❌ WRONG"
        if true_lbl == pred_lbl:
            correct_count += 1
        
        true_name = CLASS_NAMES[true_lbl]
        pred_name = CLASS_NAMES[pred_lbl]
        
        # Keep path short for cleaner display
        display_id = Path(img_id).name
        print(f"{display_id:<30} | {true_name:<15} | {pred_name:<15} | {conf * 100:>8.2f}% | {status}")

    print("=" * 80)

    # Calculate metrics
    metrics = calculate_metrics(targets, predictions)

    # Display final statistics
    print("\n" + "=" * 55)
    print(f"{'FINAL SUMMARY METRICS (SAMPLES)':^55}")
    print("=" * 55)
    print(f"  Loss             : {avg_loss:.4f}")
    print(f"  Accuracy         : {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"  Macro F1         : {metrics['macro_f1']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  QWK              : {metrics['qwk']:.4f}")
    print("\n  Per-class Recall:")
    for cls_name in CLASS_NAMES:
        recall = metrics["per_class_recall"][cls_name]
        bar = "█" * int(recall * 20)
        print(f"    {cls_name:<15}: {recall:.4f}  {bar}")
    print("=" * 55)


if __name__ == "__main__":
    main()
