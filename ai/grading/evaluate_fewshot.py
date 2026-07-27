"""Evaluate a saved RETFound Few-Shot (ProtoNet) checkpoint on a test split.

Usage example:
    python -m ai.grading.evaluate_fewshot \
        --checkpoint  E:/HocTap/DoAnTT/dr-diagnostic-system/ai/weights/grading/best-fewshot.pth \
        --dataset-dir E:/HocTap/DoAnTT/archive/split_dataset/test \
        --output-dir  eval_output_fewshot
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ai.grading.train import (
    CLASS_NAMES,
    NUM_CLASSES,
    FundusDataset,
    build_transforms,
    calculate_metrics,
    find_predefined_splits,
    save_evaluation_artifacts,
    scan_classification_split,
)


class FewShotProtoNetEvaluator(nn.Module):
    """Wrapper module for Prototypical Network inference using saved prototypes."""

    def __init__(
        self,
        encoder: nn.Module,
        projection: nn.Module,
        prototypes: torch.Tensor,
        class_ids: torch.Tensor,
        temperature: float = 0.1,
        forward_batch_size: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.projection = projection
        self.register_buffer("prototypes", prototypes)
        self.register_buffer("class_ids", class_ids)
        self.temperature = temperature
        self.forward_batch_size = max(int(forward_batch_size), 1)

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

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.encode(images)
        distances = torch.cdist(embeddings, self.prototypes).pow(2)
        logits = -distances / self.temperature
        predictions = self.class_ids[logits.argmax(dim=1)]
        return logits, predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RETFound Few-Shot (ProtoNet) checkpoint on DR test dataset."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            r"E:\HocTap\DoAnTT\dr-diagnostic-system\ai\weights\grading\best-fewshot.pth"
        ),
        help="Path to best-fewshot.pth checkpoint",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(r"E:\HocTap\DoAnTT\archive\split_dataset\test"),
        help="Root directory of the test split containing class folders 0, 1, 2, 3, 4",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_output_fewshot"),
        help="Directory where test metrics, predictions CSV, and confusion matrix will be saved",
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
        help="Limit number of samples per class for quick testing (e.g. 50)",
    )
    return parser.parse_args()


def load_fewshot_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> Tuple[FewShotProtoNetEvaluator, argparse.Namespace]:
    print(f"Loading few-shot checkpoint: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Handle PosixPath saved on Linux when loading on Windows
    temp_posix = pathlib.PosixPath
    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc,assignment]
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    finally:
        pathlib.PosixPath = temp_posix  # type: ignore[misc,assignment]

    if not isinstance(state, dict) or state.get("method") != "fixed_support_target_domain_protonet":
        raise ValueError(
            f"Checkpoint at {checkpoint_path} is not a valid fixed-support few-shot ProtoNet checkpoint."
        )

    base_args_raw = state.get("base_model_args", {})
    if isinstance(base_args_raw, dict):
        base_args = argparse.Namespace(**base_args_raw)
    else:
        base_args = base_args_raw

    image_size = int(getattr(base_args, "image_size", 224))
    model_source = getattr(base_args, "model_source", "retfound")

    if model_source == "retfound":
        encoder = timm.create_model(
            "vit_large_patch14_dinov2.lvd142m",
            pretrained=False,
            img_size=image_size,
            num_classes=NUM_CLASSES,
        )
    else:
        model_name = getattr(base_args, "model_name", "convnext_tiny.fb_in22k_ft_in1k")
        encoder = timm.create_model(model_name, pretrained=False, num_classes=NUM_CLASSES)

    encoder.load_state_dict(state["encoder_model"], strict=True)

    feature_dim = int(getattr(encoder, "num_features", 1024))
    embedding_dim = int(state.get("embedding_dim", feature_dim))

    if state.get("projection"):
        projection: nn.Module = nn.Linear(feature_dim, embedding_dim, bias=False)
        projection.load_state_dict(state["projection"], strict=True)
    else:
        projection = nn.Identity()

    prototypes = state["prototypes"].to(device)
    class_ids = state["class_ids"].to(device)
    temperature = float(state.get("temperature", 0.1))
    few_shot_args = state.get("few_shot_args", {})
    forward_batch_size = int(few_shot_args.get("forward_batch_size", 2))

    evaluator = FewShotProtoNetEvaluator(
        encoder=encoder,
        projection=projection,
        prototypes=prototypes,
        class_ids=class_ids,
        temperature=temperature,
        forward_batch_size=forward_batch_size,
    )
    evaluator.to(device)
    evaluator.eval()

    epoch = state.get("epoch", "?")
    best_loss = state.get("best_support_loss", float("nan"))
    print(
        f"Checkpoint info — epoch: {epoch}, best support loss: {best_loss:.4f}, "
        f"forward batch size: {forward_batch_size}"
    )
    return evaluator, base_args


def load_test_split(dataset_dir: Path, limit_per_class: int | None = None) -> pd.DataFrame:
    dataset_dir = dataset_dir.expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_dir}")

    # Check if dataset_dir contains 0..4 directly
    subdirs = {p.name for p in dataset_dir.iterdir() if p.is_dir()}
    if any(grade_str in subdirs for grade_str in ("0", "1", "2", "3", "4")):
        test_df = scan_classification_split(dataset_dir, dataset_dir)
    else:
        try:
            split_paths = find_predefined_splits(dataset_dir)
            dataset_root = next(iter(split_paths.values())).parent
            test_dir = split_paths["test"]
            test_df = scan_classification_split(test_dir, dataset_root)
        except ValueError:
            test_candidates = list(dataset_dir.rglob("test"))
            test_df = None
            for cand in test_candidates:
                if cand.is_dir():
                    try:
                        test_df = scan_classification_split(cand, dataset_dir)
                        break
                    except ValueError:
                        continue
            if test_df is None:
                raise ValueError(
                    f"Could not find a valid test split with subfolders 0..4 in {dataset_dir}"
                )

    if limit_per_class is not None and limit_per_class > 0:
        test_df = test_df.groupby("diagnosis").head(limit_per_class).reset_index(drop=True)

    counts = test_df["diagnosis"].value_counts().sort_index().to_dict()
    suffix = " (limited)" if limit_per_class else ""
    print(f"Test dataset{suffix}: {len(test_df):,} samples -> {counts}")
    return test_df


def _build_fake_args(real_args: argparse.Namespace) -> argparse.Namespace:
    fake = argparse.Namespace()
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


@torch.no_grad()
def run_evaluation(
    evaluator: FewShotProtoNetEvaluator,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> Tuple[float, list[int], list[int], list[str]]:
    evaluator.eval()
    total_loss = 0.0
    sample_count = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []
    all_ids: list[str] = []

    progress = tqdm(loader, desc="Evaluating Few-Shot Model", leave=False)
    for images, targets, image_ids in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits, predictions = evaluator(images)
            loss = F.cross_entropy(logits, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        sample_count += batch_size

        all_targets.extend(targets.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())
        all_ids.extend(image_ids)

    average_loss = total_loss / max(sample_count, 1)
    return average_loss, all_targets, all_predictions, all_ids


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {torch.cuda.get_device_name(0)} ({mem_gb:.1f} GB)")

    amp_enabled = device.type == "cuda" and not args.no_amp

    # Load Few-Shot Model Evaluator
    evaluator, saved_args = load_fewshot_checkpoint(args.checkpoint, device)

    # Dataset Preprocessing configurations
    image_size = getattr(saved_args, "image_size", 224)
    data_args = _build_fake_args(args)
    data_args.image_size = image_size
    data_args.preprocessing = getattr(saved_args, "preprocessing", "rgb_crop")
    data_args.enhance = getattr(saved_args, "enhance", False)

    # Load Test Split
    test_df = load_test_split(args.dataset_dir, limit_per_class=args.limit_per_class)
    _, eval_transform = build_transforms(image_size)

    test_dataset = FundusDataset(test_df, data_args, eval_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers if os.name != "nt" else 0,  # Safety for Windows multiprocessing
        pin_memory=(device.type == "cuda"),
    )

    print(f"\nRunning evaluation on {len(test_dataset):,} test images...")
    test_loss, targets, predictions, image_ids = run_evaluation(
        evaluator, test_loader, device, amp_enabled
    )

    metrics = {"loss": test_loss, **calculate_metrics(targets, predictions)}

    # Display Test Results
    print("\n" + "=" * 55)
    print("FEW-SHOT MODEL TEST SET RESULTS")
    print("=" * 55)
    print(f"  Loss             : {test_loss:.4f}")
    print(f"  Accuracy         : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  Macro F1         : {metrics['macro_f1']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  QWK              : {metrics['qwk']:.4f}")
    print("\n  Per-class Recall:")
    for cls_name in CLASS_NAMES:
        recall = metrics["per_class_recall"][cls_name]
        bar = "#" * int(recall * 20)
        print(f"    {cls_name:<15}: {recall:.4f}  {bar}")
    print("=" * 55)

    # Save artifacts
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
