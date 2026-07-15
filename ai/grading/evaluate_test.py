"""Evaluate a saved RETFound checkpoint on the predefined test split.

Usage example:
    python -m ai.grading.evaluate_test \
        --checkpoint  path/to/checkpoint-best.pth \
        --dataset-dir path/to/fundus_merged \
        --output-dir  path/to/eval_results
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from ai.grading.train import (
    NUM_CLASSES,
    CLASS_NAMES,
    build_transforms,
    evaluate,
    calculate_metrics,
    find_predefined_splits,
    scan_classification_split,
    save_evaluation_artifacts,
    FundusDataset,
)
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint-best.pth downloaded from Colab/Drive",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Root of the fundus dataset containing predefined train/val/test splits",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_output"),
        help="Directory where test_metrics.json and confusion matrix will be saved",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision (AMP)",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help="Limit the number of samples per class for quick testing (e.g. 50)",
    )
    return parser.parse_args()


def _build_fake_args(real_args: argparse.Namespace) -> argparse.Namespace:
    """Build a minimal args namespace compatible with train.py helpers."""
    import argparse as _ap

    fake = _ap.Namespace()
    fake.dataset_dir = real_args.dataset_dir
    fake.images_dir = None
    fake.labels_csv = None
    fake.image_column = "id_code"
    fake.label_column = "diagnosis"
    fake.image_extension = ".png"
    fake.image_size = 224
    fake.enhance = False
    fake.loss = "ce"
    return fake


def load_test_split(args: argparse.Namespace, fake_args: argparse.Namespace):
    split_paths = find_predefined_splits(args.dataset_dir)
    dataset_root = next(iter(split_paths.values())).parent
    test_df = scan_classification_split(split_paths["test"], dataset_root)
    
    if getattr(args, "limit_per_class", None) is not None:
        limit = args.limit_per_class
        # Group by label (diagnosis) và lấy N dòng đầu tiên của mỗi nhóm
        test_df = test_df.groupby("diagnosis").head(limit).reset_index(drop=True)
        
    counts = test_df["diagnosis"].value_counts().sort_index().to_dict()
    print(f"Test split (limited): {len(test_df):,} samples → {counts}")
    return test_df


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Khắc phục lỗi PosixPath trên hệ điều hành Windows
    import pathlib
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    finally:
        pathlib.PosixPath = temp

    # Reconstruct model using the args that were saved inside the checkpoint
    saved_args = argparse.Namespace(**state["args"])

    import timm

    output_dim = NUM_CLASSES if saved_args.loss == "ce" else NUM_CLASSES - 1
    if saved_args.model_source == "retfound":
        model = timm.create_model(
            "vit_large_patch14_dinov2.lvd142m",
            pretrained=False,  # weights come from the checkpoint
            img_size=saved_args.image_size,
            num_classes=output_dim,
        )
    else:
        model = timm.create_model(
            saved_args.model_name,
            pretrained=False,
            num_classes=output_dim,
        )

    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()

    epoch = state.get("epoch", "?")
    best_qwk = state.get("best_qwk", float("nan"))
    print(f"Checkpoint info — saved at epoch {epoch}, best QWK during training: {best_qwk:.4f}")
    return model, saved_args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {torch.cuda.get_device_name(0)} ({mem_gb:.1f} GB)")

    amp_enabled = device.type == "cuda" and not args.no_amp

    # Load model từ checkpoint
    model, saved_args = load_checkpoint(args.checkpoint, device)

    # Patch một số field cần thiết từ args người dùng truyền vào
    saved_args.enhance = False
    saved_args.image_size = getattr(saved_args, "image_size", 224)

    # Build test dataset
    fake_args = _build_fake_args(args)
    test_df = load_test_split(args, fake_args)
    _, eval_transform = build_transforms(saved_args.image_size)

    test_dataset = FundusDataset(test_df, saved_args, eval_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Dùng CE loss không trọng số chỉ để tính val_loss tham khảo
    import torch.nn as nn
    criterion = nn.CrossEntropyLoss()

    print(f"\nRunning evaluation on {len(test_dataset):,} test images...")
    test_loss, targets, predictions, image_ids = evaluate(
        model,
        test_loader,
        criterion,
        device,
        saved_args.loss,
        amp_enabled,
    )

    metrics = {"loss": test_loss, **calculate_metrics(targets, predictions)}

    # In kết quả ra màn hình
    print("\n" + "=" * 55)
    print("TEST SET RESULTS")
    print("=" * 55)
    print(f"  Loss             : {test_loss:.4f}")
    print(f"  Accuracy         : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  Macro F1         : {metrics['macro_f1']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  QWK              : {metrics['qwk']:.4f}")
    print("\n  Per-class Recall:")
    for cls_name in CLASS_NAMES:
        recall = metrics["per_class_recall"][cls_name]
        bar = "█" * int(recall * 20)
        print(f"    {cls_name:<15}: {recall:.4f}  {bar}")
    print("=" * 55)

    # Lưu kết quả
    save_evaluation_artifacts(
        args.output_dir,
        targets,
        predictions,
        image_ids,
        metrics,
    )
    print(f"\nResults saved to: {args.output_dir.resolve()}")
    print(f"  - test_metrics.json")
    print(f"  - test_predictions.csv")
    print(f"  - confusion_matrix_normalized.png")


if __name__ == "__main__":
    main()
