"""Colab-friendly fine-tuning pipeline for five-grade diabetic retinopathy.

The script supports the retina-specific RETFound-DINOv2 checkpoint and lighter
``timm`` backbones. It consumes either predefined folder splits or a legacy
flat image directory plus CSV, keeps the test set untouched until training
ends, and reports macro-F1, balanced accuracy, per-class recall and QWK.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import timm
import torch
import torch.nn as nn
from huggingface_hub import get_token, hf_hub_download
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from timm.layers.pos_embed import resample_abs_pos_embed
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

from ai.grading.artifacts import (
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    ArtifactMetadata,
    build_metadata,
    checkpoint_metadata,
    dump_metadata_json,
    load_torch_checkpoint,
    validate_resume_metadata,
)
from ai.grading.data_audit import audit_splits
from ai.preprocessing.fundus_prep import preprocess_fundus_array


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLIT_ALIASES = {
    "train": ("train", "training"),
    "val": ("val", "valid", "validation"),
    "test": ("test", "testing"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help=(
            "Root of a folder-classification dataset with predefined "
            "train/validation/test splits (for example split_dataset from the "
            "merged Kaggle fundus dataset)"
        ),
    )
    parser.add_argument("--images-dir", type=Path, help="Legacy flat image directory")
    parser.add_argument("--labels-csv", type=Path, help="Legacy CSV label file")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split-dir", type=Path, default=None)
    parser.add_argument("--image-column", default="id_code")
    parser.add_argument("--label-column", default="diagnosis")
    parser.add_argument("--image-extension", default=".png")
    parser.add_argument(
        "--max-images-per-grade",
        type=int,
        default=None,
        help=(
            "Deprecated compatibility option. Prefer separate train/eval limits."
        ),
    )
    parser.add_argument("--max-train-images-per-grade", type=int, default=None)
    parser.add_argument(
        "--max-eval-images-per-grade",
        type=int,
        default=0,
        help="Maximum per grade in each validation/test split; 0 uses all images",
    )

    parser.add_argument(
        "--model-source",
        choices=("retfound", "timm"),
        default="retfound",
    )
    parser.add_argument(
        "--retfound-id",
        default="RETFound_dinov2_meh",
        help="Hugging Face checkpoint under YukunZhou/<id>",
    )
    parser.add_argument(
        "--model-name",
        default="convnext_tiny.fb_in22k_ft_in1k",
        help="timm model used when --model-source=timm",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--loss",
        choices=("ce",),
        default="ce",
        help="Five-class cross-entropy baseline",
    )
    parser.add_argument(
        "--balance",
        choices=("none", "effective", "sampler"),
        default="none",
        help="Use only one rebalancing method per experiment",
    )
    parser.add_argument("--effective-beta", type=float, default=0.9999)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--adaptation",
        choices=("linear_probe", "last_n_blocks", "full_finetune"),
        default="last_n_blocks",
    )
    parser.add_argument("--last-n-blocks", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=8)
    parser.add_argument("--peak-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--backbone-lr", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--layer-decay", type=float, default=0.75)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--crop-tolerance", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--enhance", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--leakage-policy",
        choices=("error", "warn", "off"),
        default="error",
    )
    parser.add_argument("--phash-distance", type=int, default=4)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    using_predefined = args.dataset_dir is not None
    using_legacy = args.images_dir is not None or args.labels_csv is not None
    if using_predefined == using_legacy:
        raise ValueError(
            "Choose exactly one data input: --dataset-dir, or both "
            "--images-dir and --labels-csv"
        )
    if using_legacy and (args.images_dir is None or args.labels_csv is None):
        raise ValueError("--images-dir and --labels-csv must be provided together")
    if args.model_source == "retfound" and args.image_size != 224:
        raise ValueError("RETFound-DINOv2 currently requires --image-size 224")
    if args.val_size <= 0 or args.test_size <= 0:
        raise ValueError("Validation and test sizes must be positive")
    if args.val_size + args.test_size >= 1:
        raise ValueError("val-size + test-size must be below 1")
    if args.accum_steps < 1:
        raise ValueError("accum-steps must be at least 1")
    if args.freeze_epochs is not None:
        warnings.warn(
            "--freeze-epochs is deprecated and ignored; use --adaptation",
            DeprecationWarning,
        )
    if args.head_lr is not None:
        warnings.warn("--head-lr is deprecated; using its value as --peak-lr")
        args.peak_lr = args.head_lr
    if args.backbone_lr is not None:
        warnings.warn(
            "--backbone-lr is ignored; layer-wise rates are derived from "
            "--peak-lr and --layer-decay"
        )
    if args.peak_lr <= 0 or args.min_lr < 0 or args.min_lr > args.peak_lr:
        raise ValueError("Require 0 <= min-lr <= peak-lr")
    if not 0 < args.layer_decay <= 1:
        raise ValueError("layer-decay must be in (0, 1]")
    if args.warmup_epochs < 0 or args.warmup_epochs >= args.epochs:
        raise ValueError("warmup-epochs must be non-negative and below epochs")
    if args.last_n_blocks < 1:
        raise ValueError("last-n-blocks must be at least 1")
    if not 0 <= args.label_smoothing < 1:
        raise ValueError("label-smoothing must be in [0, 1)")
    if args.max_images_per_grade is not None and args.max_images_per_grade < 0:
        raise ValueError("max-images-per-grade cannot be negative")
    if args.max_train_images_per_grade is None:
        if args.max_images_per_grade is None:
            args.max_train_images_per_grade = 700
        else:
            legacy_limits = split_grade_limits(
                args.max_images_per_grade, args.val_size, args.test_size
            )
            args.max_train_images_per_grade = legacy_limits["train"]
    if args.max_train_images_per_grade < 0 or args.max_eval_images_per_grade < 0:
        raise ValueError("Per-grade image limits cannot be negative")
    if args.phash_distance < 0 or args.phash_distance > 16:
        raise ValueError("phash-distance must be between 0 and 16")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_or_create_csv_splits(
    args: argparse.Namespace, *, apply_limits: bool = True
) -> dict[str, pd.DataFrame]:
    split_dir = args.split_dir or (args.output_dir / "splits")
    split_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: split_dir / f"{name}.csv" for name in ("train", "val", "test")}
    if all(path.exists() for path in paths.values()):
        print(f"Reusing fixed splits from {split_dir}")
        splits = {name: pd.read_csv(path) for name, path in paths.items()}
        if apply_limits:
            splits = limit_splits_for_experiment(splits, args)
        for name, split in splits.items():
            split.to_csv(paths[name], index=False)
        return splits

    frame = pd.read_csv(args.labels_csv)
    required = {args.image_column, args.label_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    frame = frame[[args.image_column, args.label_column]].copy()
    frame[args.label_column] = frame[args.label_column].astype(int)
    invalid = sorted(set(frame[args.label_column]) - set(range(NUM_CLASSES)))
    if invalid:
        raise ValueError(f"Labels outside 0..4: {invalid}")

    train_val, test = train_test_split(
        frame,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=frame[args.label_column],
    )
    relative_val_size = args.val_size / (1.0 - args.test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        random_state=args.seed,
        stratify=train_val[args.label_column],
    )
    splits = {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }
    if apply_limits:
        splits = limit_splits_for_experiment(splits, args)
    for name, split in splits.items():
        split.to_csv(paths[name], index=False)
        print(f"{name}: {len(split)} samples; {split[args.label_column].value_counts().sort_index().to_dict()}")
    return splits


def _class_label_from_dir(name: str) -> int:
    """Map the merged dataset's class directory name to a DR grade."""
    stripped = name.strip().lower()
    numeric_prefix = re.match(r"^([0-4])(?:\D|$)", stripped)
    if numeric_prefix:
        return int(numeric_prefix.group(1))

    normalized = re.sub(r"[^a-z0-9]+", "", stripped)
    aliases = {
        "nodr": 0,
        "normal": 0,
        "healthy": 0,
        "mild": 1,
        "milddr": 1,
        "moderate": 2,
        "moderatedr": 2,
        "severe": 3,
        "severedr": 3,
        "proliferative": 4,
        "proliferativedr": 4,
        "pdr": 4,
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported class directory {name!r}; expected 0..4 or a standard DR class name"
        )
    return aliases[normalized]


def _split_directories(candidate: Path) -> dict[str, Path] | None:
    if not candidate.is_dir():
        return None
    children = {child.name.lower(): child for child in candidate.iterdir() if child.is_dir()}
    resolved: dict[str, Path] = {}
    for canonical, aliases in SPLIT_ALIASES.items():
        match = next((children[alias] for alias in aliases if alias in children), None)
        if match is None:
            return None
        resolved[canonical] = match
    return resolved


def find_predefined_splits(dataset_dir: Path) -> dict[str, Path]:
    """Locate train/validation/test, including Kaggle's split_dataset wrapper."""
    dataset_dir = dataset_dir.expanduser().resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    candidates = [dataset_dir]
    candidates.extend(child for child in dataset_dir.iterdir() if child.is_dir())
    for child in list(candidates[1:]):
        candidates.extend(grandchild for grandchild in child.iterdir() if grandchild.is_dir())
    for candidate in candidates:
        splits = _split_directories(candidate)
        if splits:
            print(f"Using predefined splits from {candidate}")
            return splits
    raise ValueError(
        f"Could not find train/validation/test below {dataset_dir}. "
        "Point --dataset-dir at the extracted dataset or its split_dataset directory."
    )


def scan_classification_split(split_dir: Path, dataset_root: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    seen_labels: set[int] = set()
    for class_dir in sorted((path for path in split_dir.iterdir() if path.is_dir())):
        label = _class_label_from_dir(class_dir.name)
        if label in seen_labels:
            raise ValueError(f"Duplicate directories for grade {label} under {split_dir}")
        seen_labels.add(label)
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append(
                    {
                        "image_path": os.fspath(image_path.resolve()),
                        "image_id": image_path.relative_to(dataset_root).as_posix(),
                        "diagnosis": label,
                    }
                )
    if not records:
        raise ValueError(f"No supported images found under {split_dir}")
    missing_labels = sorted(set(range(NUM_CLASSES)) - seen_labels)
    if missing_labels:
        raise ValueError(f"Split {split_dir.name} is missing class directories: {missing_labels}")
    return pd.DataFrame.from_records(records)


def limit_samples_per_grade(
    frame: pd.DataFrame,
    label_column: str,
    max_per_grade: int,
    seed: int,
) -> pd.DataFrame:
    """Deterministically cap each grade while retaining every smaller class."""
    if max_per_grade == 0:
        return frame.reset_index(drop=True)

    sampled_groups: list[pd.DataFrame] = []
    for grade in range(NUM_CLASSES):
        group = frame[frame[label_column] == grade]
        if group.empty:
            raise ValueError(f"Training split has no images for grade {grade}")
        if len(group) > max_per_grade:
            group = group.sample(n=max_per_grade, random_state=seed + grade)
        sampled_groups.append(group)

    sampled = pd.concat(sampled_groups, ignore_index=True)
    sort_column = "image_id" if "image_id" in sampled.columns else label_column
    return sampled.sort_values(sort_column, kind="stable").reset_index(drop=True)


def limit_splits_per_grade(
    splits: dict[str, pd.DataFrame],
    label_column: str,
    max_per_grade: int,
    val_size: float,
    test_size: float,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Divide one per-grade budget across train, validation and test."""
    limited: dict[str, pd.DataFrame] = {}
    seed_offsets = {"train": 0, "val": 1000, "test": 2000}
    limits = split_grade_limits(max_per_grade, val_size, test_size)
    if max_per_grade:
        print(
            "Per-grade image budget: "
            f"train={limits['train']}, val={limits['val']}, test={limits['test']}"
        )
    for name, split in splits.items():
        split_limit = limits[name]
        original_size = len(split)
        limited[name] = limit_samples_per_grade(
            split,
            label_column,
            split_limit,
            seed + seed_offsets.get(name, 0),
        )
        if split_limit:
            print(
                f"Limited {name} split from {original_size} to {len(limited[name])} "
                f"images (at most {split_limit} per grade; seed={seed})"
            )
    return limited


def limit_splits_for_experiment(
    splits: dict[str, pd.DataFrame], args: Any
) -> dict[str, pd.DataFrame]:
    """Limit training independently while keeping evaluation complete by default."""
    train_limit = int(getattr(args, "max_train_images_per_grade", 0) or 0)
    eval_limit = int(getattr(args, "max_eval_images_per_grade", 0) or 0)
    limits = {"train": train_limit, "val": eval_limit, "test": eval_limit}
    limited: dict[str, pd.DataFrame] = {}
    seed_offsets = {"train": 0, "val": 1000, "test": 2000}
    for name, split in splits.items():
        limit = limits[name]
        limited[name] = limit_samples_per_grade(
            split,
            args.label_column,
            limit,
            args.seed + seed_offsets[name],
        )
        if limit:
            print(
                f"Limited {name} from {len(split)} to {len(limited[name])} "
                f"(at most {limit} images per grade)"
            )
    return limited


def split_grade_limits(
    max_per_grade: int, val_size: float, test_size: float
) -> dict[str, int]:
    """Convert a total per-grade budget into train/val/test integer quotas."""
    if max_per_grade == 0:
        return {"train": 0, "val": 0, "test": 0}

    val_limit = int(max_per_grade * val_size)
    test_limit = int(max_per_grade * test_size)
    train_limit = max_per_grade - val_limit - test_limit
    limits = {"train": train_limit, "val": val_limit, "test": test_limit}
    if any(limit < 1 for limit in limits.values()):
        raise ValueError(
            "max-images-per-grade is too small for the requested train/val/test ratios"
        )
    return limits


def load_predefined_splits(
    args: argparse.Namespace, *, apply_limits: bool = True
) -> dict[str, pd.DataFrame]:
    split_paths = find_predefined_splits(args.dataset_dir)
    dataset_root = next(iter(split_paths.values())).parent
    splits = {
        name: scan_classification_split(path, dataset_root)
        for name, path in split_paths.items()
    }
    if args.label_column != "diagnosis":
        splits = {
            name: split.rename(columns={"diagnosis": args.label_column})
            for name, split in splits.items()
        }

    if apply_limits:
        if hasattr(args, "max_train_images_per_grade"):
            splits = limit_splits_for_experiment(splits, args)
        else:
            splits = limit_splits_per_grade(
                splits,
                args.label_column,
                args.max_images_per_grade,
                args.val_size,
                args.test_size,
                args.seed,
            )

    all_paths: list[str] = []
    manifest_dir = args.split_dir or (args.output_dir / "splits")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        paths = split["image_path"].tolist()
        all_paths.extend(paths)
        split.to_csv(manifest_dir / f"{name}.csv", index=False)
        counts = split[args.label_column].value_counts().sort_index().to_dict()
        print(f"{name}: {len(split)} samples; {counts}")
    if len(all_paths) != len(set(all_paths)):
        raise ValueError("The predefined train/validation/test splits contain duplicate paths")
    return splits


def load_splits(
    args: argparse.Namespace, *, apply_limits: bool = True
) -> dict[str, pd.DataFrame]:
    if args.dataset_dir is not None:
        return load_predefined_splits(args, apply_limits=apply_limits)
    return load_or_create_csv_splits(args, apply_limits=apply_limits)


def save_split_manifests(
    splits: dict[str, pd.DataFrame], manifest_dir: Path
) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        split.to_csv(manifest_dir / f"{name}.csv", index=False)


class FundusDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        args: argparse.Namespace,
        transform: transforms.Compose,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.args = args
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def _image_path(self, image_id: str) -> Path:
        if "image_path" in self.frame.columns:
            return Path(image_id)
        path = self.args.images_dir / image_id
        if not path.suffix:
            path = path.with_suffix(self.args.image_extension)
        return path

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        row = self.frame.iloc[index]
        if "image_path" in self.frame.columns:
            path = Path(str(row["image_path"]))
            image_id = str(row["image_id"])
        else:
            image_id = str(row[self.args.image_column])
            path = self._image_path(image_id)
        image_bgr = cv2.imread(os.fspath(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        image_bgr = preprocess_fundus_array(
            image_bgr,
            img_size=None,
            enhance=self.args.enhance,
            crop_tolerance=self.args.crop_tolerance,
        )
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = self.transform(Image.fromarray(image_rgb))
        return image, int(row[self.args.label_column]), image_id


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15, interpolation=InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def effective_number_weights(
    labels: Iterable[int], beta: float, device: torch.device
) -> torch.Tensor:
    counts = np.bincount(np.asarray(list(labels), dtype=np.int64), minlength=NUM_CLASSES)
    effective = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(effective, 1e-12)
    weights = weights / weights.sum() * NUM_CLASSES
    print(f"Class counts: {counts.tolist()}")
    print(f"Effective-number weights: {weights.round(4).tolist()}")
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_sampler(
    labels: list[int], *, generator: torch.Generator | None = None
) -> WeightedRandomSampler:
    counts = np.bincount(np.asarray(labels), minlength=NUM_CLASSES)
    class_weights = 1.0 / np.maximum(counts, 1)
    sample_weights = torch.tensor([class_weights[label] for label in labels], dtype=torch.double)
    return WeightedRandomSampler(
        sample_weights,
        len(sample_weights),
        replacement=True,
        generator=generator,
    )


def resize_pos_embed_for_model(state: dict[str, torch.Tensor], model: nn.Module) -> None:
    """Resize a ViT checkpoint's spatial position grid to the target model."""
    source = state.get("pos_embed")
    target = getattr(model, "pos_embed", None)
    if source is None or target is None or source.shape == target.shape:
        return

    if source.ndim != 3 or target.ndim != 3 or source.shape[-1] != target.shape[-1]:
        return

    grid_size = getattr(getattr(model, "patch_embed", None), "grid_size", None)
    if grid_size is None:
        return

    num_prefix_tokens = int(getattr(model, "num_prefix_tokens", 1))
    resized = resample_abs_pos_embed(
        source,
        new_size=list(grid_size),
        num_prefix_tokens=num_prefix_tokens,
        interpolation="bicubic",
        antialias=True,
    )
    if resized.shape != target.shape:
        raise RuntimeError(
            "Could not resize checkpoint pos_embed from "
            f"{tuple(source.shape)} to {tuple(target.shape)}"
        )
    print(
        "Resized checkpoint pos_embed from "
        f"{tuple(source.shape)} to {tuple(resized.shape)}"
    )
    state["pos_embed"] = resized


def load_retfound_dinov2(args: argparse.Namespace, output_dim: int) -> nn.Module:
    print(f"Loading retina-specific checkpoint {args.retfound_id}")
    hf_token = os.environ.get("HF_TOKEN") or get_token()
    if not hf_token:
        raise RuntimeError(
            "Missing Hugging Face authentication. Add HF_TOKEN to Colab Secrets "
            "or run `huggingface-cli login`; never commit the token to Git."
        )
    model = timm.create_model(
        "vit_large_patch14_dinov2.lvd142m",
        pretrained=True,
        img_size=224,
        num_classes=output_dim,
    )
    checkpoint_path = hf_hub_download(
        repo_id=f"YukunZhou/{args.retfound_id}",
        filename=f"{args.retfound_id}.pth",
        token=hf_token,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "teacher" not in checkpoint:
        raise KeyError("RETFound-DINOv2 checkpoint does not contain a 'teacher' state")
    state = checkpoint["teacher"]
    state = {key.replace("backbone.", ""): value for key, value in state.items()}
    state = {key.replace("mlp.w12.", "mlp.fc1."): value for key, value in state.items()}
    state = {key.replace("mlp.w3.", "mlp.fc2."): value for key, value in state.items()}

    current = model.state_dict()
    for key in ("head.weight", "head.bias"):
        if key in state and key in current and state[key].shape != current[key].shape:
            del state[key]
    resize_pos_embed_for_model(state, model)
    incompatible = model.load_state_dict(state, strict=False)
    print(
        f"Loaded RETFound backbone; missing={len(incompatible.missing_keys)}, "
        f"unexpected={len(incompatible.unexpected_keys)}"
    )
    nn.init.trunc_normal_(model.head.weight, std=2e-5)
    if model.head.bias is not None:
        nn.init.zeros_(model.head.bias)
    return model


def build_model(args: argparse.Namespace) -> nn.Module:
    if args.model_source == "retfound":
        return load_retfound_dinov2(args, NUM_CLASSES)
    print(f"Loading timm pretrained model {args.model_name}")
    return timm.create_model(args.model_name, pretrained=True, num_classes=NUM_CLASSES)


def is_head_parameter(name: str) -> bool:
    parts = set(name.split("."))
    return bool(parts.intersection({"head", "classifier", "fc"}))


def configure_trainable(
    model: nn.Module, adaptation: str, last_n_blocks: int = 6
) -> None:
    if adaptation == "full_finetune":
        for parameter in model.parameters():
            parameter.requires_grad = True
    elif adaptation == "linear_probe":
        for name, parameter in model.named_parameters():
            parameter.requires_grad = is_head_parameter(name)
    elif adaptation == "last_n_blocks":
        blocks = getattr(model, "blocks", None)
        if blocks is None:
            raise ValueError("last_n_blocks adaptation requires a ViT-style model.blocks")
        if last_n_blocks > len(blocks):
            raise ValueError(
                f"Requested {last_n_blocks} blocks, but model only has {len(blocks)}"
            )
        first_trainable = len(blocks) - last_n_blocks
        for name, parameter in model.named_parameters():
            block_match = re.match(r"^blocks\.(\d+)\.", name)
            train_block = bool(
                block_match and int(block_match.group(1)) >= first_trainable
            )
            final_norm = name.startswith(("norm.", "fc_norm."))
            parameter.requires_grad = (
                is_head_parameter(name) or train_block or final_norm
            )
    else:
        raise ValueError(f"Unsupported adaptation mode: {adaptation}")
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Trainable parameters: {count / 1e6:.2f}M; adaptation={adaptation}")
    if count == 0:
        raise RuntimeError("No trainable parameters found; classifier name is unsupported")


def build_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    blocks = getattr(model, "blocks", None)
    block_count = len(blocks) if blocks is not None else 0
    maximum_layer = block_count + 1
    groups: dict[tuple[int, bool], dict[str, Any]] = {}

    def layer_id(name: str) -> int:
        if is_head_parameter(name) or name.startswith(("norm.", "fc_norm.")):
            return maximum_layer
        match = re.match(r"^blocks\.(\d+)\.", name)
        if match:
            return int(match.group(1)) + 1
        return 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        current_layer = layer_id(name)
        no_decay = parameter.ndim == 1 or name.endswith(".bias") or name in {
            "pos_embed",
            "cls_token",
            "mask_token",
        }
        key = (current_layer, no_decay)
        if key not in groups:
            lr_scale = args.layer_decay ** (maximum_layer - current_layer)
            groups[key] = {
                "params": [],
                "lr": args.peak_lr * lr_scale,
                "base_lr": args.peak_lr * lr_scale,
                "lr_scale": lr_scale,
                "weight_decay": 0.0 if no_decay else args.weight_decay,
                "group_name": f"layer_{current_layer}_{'no_decay' if no_decay else 'decay'}",
            }
        groups[key]["params"].append(parameter)
    ordered_groups = [groups[key] for key in sorted(groups)]
    if not ordered_groups:
        raise RuntimeError("No trainable parameters available to the optimizer")
    print("Optimizer parameter groups:")
    for group in ordered_groups:
        parameter_count = sum(parameter.numel() for parameter in group["params"])
        print(
            f"  {group['group_name']}: {parameter_count / 1e6:.2f}M, "
            f"peak_lr={group['base_lr']:.3e}, wd={group['weight_decay']}"
        )
    return torch.optim.AdamW(ordered_groups, lr=args.peak_lr)


def build_criterion(
    args: argparse.Namespace, train_labels: list[int], device: torch.device
) -> nn.Module:
    weights = None
    if args.balance == "effective":
        weights = effective_number_weights(train_labels, args.effective_beta, device)
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)


def predictions_from_logits(logits: torch.Tensor, loss_name: str = "ce") -> torch.Tensor:
    return logits.argmax(dim=1)


class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        total_updates: int,
        warmup_updates: int,
        min_lr: float,
    ) -> None:
        if total_updates < 1:
            raise ValueError("Scheduler needs at least one optimizer update")
        self.optimizer = optimizer
        self.total_updates = total_updates
        self.warmup_updates = max(0, min(warmup_updates, total_updates - 1))
        self.min_lr = min_lr
        self.last_update = -1

    def _base_lr(self, update: int) -> float:
        if self.warmup_updates and update < self.warmup_updates:
            return (update + 1) / self.warmup_updates
        cosine_updates = max(self.total_updates - self.warmup_updates, 1)
        progress = (update - self.warmup_updates) / max(cosine_updates - 1, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    def step_update(self, update: int) -> None:
        multiplier = self._base_lr(update)
        for group in self.optimizer.param_groups:
            peak = float(group["base_lr"])
            floor = self.min_lr * float(group.get("lr_scale", 1.0))
            group["lr"] = floor + (peak - floor) * multiplier
        self.last_update = update

    def state_dict(self) -> dict[str, int]:
        return {"last_update": self.last_update}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.last_update = int(state.get("last_update", -1))


def create_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # PyTorch before the device-aware AMP API
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    args: argparse.Namespace,
    amp_enabled: bool,
    scheduler: WarmupCosineScheduler,
    global_update: int,
) -> tuple[float, float, int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    sample_count = 0
    gradient_norm_sum = 0.0
    optimizer_steps = 0
    progress = tqdm(loader, desc="train", leave=False)
    for step, (images, targets, _) in enumerate(progress):
        if step % args.accum_steps == 0:
            scheduler.step_update(global_update)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss / args.accum_steps).backward()
        should_step = (step + 1) % args.accum_steps == 0 or step + 1 == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            gradient_norm_sum += float(gradient_norm)
            optimizer_steps += 1
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_update += 1
        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        sample_count += batch_size
        progress.set_postfix(loss=f"{running_loss / sample_count:.4f}")
    return (
        running_loss / max(sample_count, 1),
        gradient_norm_sum / max(optimizer_steps, 1),
        global_update,
    )


@dataclass
class EvaluationResult:
    loss: float
    targets: list[int]
    predictions: list[int]
    probabilities: list[list[float]]
    image_ids: list[str]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    loss_name: str,
    amp_enabled: bool,
) -> EvaluationResult:
    model.eval()
    total_loss = 0.0
    sample_count = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []
    all_probabilities: list[list[float]] = []
    all_ids: list[str] = []
    for images, targets, image_ids in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)
        predictions = predictions_from_logits(logits, loss_name)
        probabilities = torch.softmax(logits, dim=1)
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        sample_count += batch_size
        all_targets.extend(targets.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())
        all_probabilities.extend(probabilities.float().cpu().tolist())
        all_ids.extend(image_ids)
    return EvaluationResult(
        loss=total_loss / max(sample_count, 1),
        targets=all_targets,
        predictions=all_predictions,
        probabilities=all_probabilities,
        image_ids=all_ids,
    )


def expected_calibration_error(
    targets: list[int], probabilities: list[list[float]], bins: int = 15
) -> float:
    target_array = np.asarray(targets, dtype=np.int64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    confidence = probability_array.max(axis=1)
    predicted = probability_array.argmax(axis=1)
    correct = predicted == target_array
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence > lower) & (confidence <= upper)
        if index == 0:
            mask |= confidence == 0.0
        if mask.any():
            error += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return error


def multiclass_brier_score(
    targets: list[int], probabilities: list[list[float]]
) -> float:
    target_array = np.asarray(targets, dtype=np.int64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    one_hot = np.eye(NUM_CLASSES, dtype=np.float64)[target_array]
    return float(np.mean(np.sum((probability_array - one_hot) ** 2, axis=1)))


def calculate_metrics(
    targets: list[int],
    predictions: list[int],
    probabilities: list[list[float]] | None = None,
) -> dict[str, Any]:
    report = classification_report(
        targets,
        predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                labels=list(range(NUM_CLASSES)),
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "qwk": float(cohen_kappa_score(targets, predictions, weights="quadratic")),
        "per_class_recall": {
            name: float(report[name]["recall"]) for name in CLASS_NAMES
        },
    }
    if probabilities is not None:
        probability_array = np.asarray(probabilities, dtype=np.float64)
        one_hot = np.eye(NUM_CLASSES, dtype=np.int64)[np.asarray(targets)]
        try:
            metrics["macro_auroc"] = float(
                roc_auc_score(one_hot, probability_array, average="macro")
            )
        except ValueError:
            metrics["macro_auroc"] = float("nan")
        try:
            metrics["macro_auprc"] = float(
                average_precision_score(one_hot, probability_array, average="macro")
            )
        except ValueError:
            metrics["macro_auprc"] = float("nan")
        metrics["ece_15_bin"] = expected_calibration_error(targets, probabilities)
        metrics["multiclass_brier"] = multiclass_brier_score(targets, probabilities)
    return metrics


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    epoch: int,
    best_qwk: float,
    stale_epochs: int,
    args: argparse.Namespace,
    metadata: ArtifactMetadata,
    scheduler: WarmupCosineScheduler,
    global_update: int,
    loader_generator: torch.Generator,
    *,
    include_training_state: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "model": model.state_dict(),
        "epoch": epoch,
        "best_qwk": best_qwk,
        "stale_epochs": stale_epochs,
        "args": vars(args),
        "metadata": metadata.to_dict(),
    }
    if include_training_state:
        state.update(
            {
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "scheduler": scheduler.state_dict(),
                "global_update": global_update,
                "rng_state": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all(),
                    "loader_generator": loader_generator.get_state(),
                },
            }
        )
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary_path)
    os.replace(temporary_path, path)


def restore_rng_state(state: dict[str, Any], loader_generator: torch.Generator) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])
    if state.get("loader_generator") is not None:
        loader_generator.set_state(state["loader_generator"])


def save_evaluation_artifacts(
    output_dir: Path,
    targets: list[int],
    predictions: list[int],
    probabilities: list[list[float]],
    image_ids: list[str],
    metrics: dict[str, Any],
) -> None:
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    prediction_frame = pd.DataFrame(
        {"image_id": image_ids, "true_grade": targets, "predicted_grade": predictions}
    )
    probability_array = np.asarray(probabilities, dtype=np.float64)
    for grade in range(NUM_CLASSES):
        prediction_frame[f"prob_grade_{grade}"] = probability_array[:, grade]
    prediction_frame.to_csv(output_dir / "test_predictions.csv", index=False)

    matrix = confusion_matrix(
        targets, predictions, labels=list(range(NUM_CLASSES)), normalize="true"
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        vmin=0,
        vmax=1,
    )
    plt.xlabel("Predicted grade")
    plt.ylabel("True grade")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix_normalized.png", dpi=180)
    plt.close()

    confidence = probability_array.max(axis=1)
    correct = probability_array.argmax(axis=1) == np.asarray(targets)
    edges = np.linspace(0.0, 1.0, 16)
    bin_confidence: list[float] = []
    bin_accuracy: list[float] = []
    for index in range(15):
        mask = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if index == 0:
            mask |= confidence == 0.0
        if mask.any():
            bin_confidence.append(float(confidence[mask].mean()))
            bin_accuracy.append(float(correct[mask].mean()))
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    plt.plot(bin_confidence, bin_accuracy, marker="o", label="model")
    plt.xlabel("Mean confidence")
    plt.ylabel("Accuracy")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "reliability_diagram.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    raw_splits = load_splits(args, apply_limits=False)
    if args.leakage_policy == "off":
        audited_splits = raw_splits
    else:
        audit_result = audit_splits(
            raw_splits,
            args,
            args.output_dir / "audit",
            phash_distance=args.phash_distance,
            fail_on_leakage=args.leakage_policy == "error",
        )
        audited_splits = audit_result.splits
        print(json.dumps(audit_result.report, indent=2, ensure_ascii=False))
    if args.audit_only:
        print(f"Audit complete: {args.output_dir / 'audit'}")
        return

    splits = limit_splits_for_experiment(audited_splits, args)
    manifest_dir = args.split_dir or (args.output_dir / "splits")
    save_split_manifests(splits, manifest_dir)
    project_root = Path(__file__).resolve().parents[2]
    metadata = build_metadata(
        args, manifest_dir=manifest_dir, project_root=project_root
    )
    dump_metadata_json(metadata, args.output_dir / "artifact_metadata.json")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. In Colab choose a GPU runtime first.")
    device = torch.device("cuda")
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {torch.cuda.get_device_name(0)} ({memory_gb:.1f} GB)")
    amp_enabled = not args.no_amp

    train_transform, eval_transform = build_transforms(args.image_size)
    datasets = {
        "train": FundusDataset(splits["train"], args, train_transform),
        "val": FundusDataset(splits["val"], args, eval_transform),
        "test": FundusDataset(splits["test"], args, eval_transform),
    }
    train_labels = splits["train"][args.label_column].astype(int).tolist()
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    sampler = (
        build_sampler(train_labels, generator=loader_generator)
        if args.balance == "sampler"
        else None
    )
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            generator=loader_generator,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        ),
    }

    model = build_model(args).to(device)
    configure_trainable(model, args.adaptation, args.last_n_blocks)
    criterion = build_criterion(args, train_labels, device)
    scaler = create_scaler(amp_enabled)
    start_epoch, best_qwk, stale_epochs = 0, -1.0, 0
    resume_state = None
    if args.resume:
        resume_state = load_torch_checkpoint(args.resume)
        saved_metadata = checkpoint_metadata(resume_state)
        if saved_metadata.schema_version > 0:
            validate_resume_metadata(saved_metadata, metadata)
        else:
            print(
                "Warning: legacy checkpoint has no artifact contract; "
                "optimizer/scheduler state will not be restored."
            )
        model.load_state_dict(resume_state["model"])
        start_epoch = int(resume_state["epoch"]) + 1
        best_qwk = float(resume_state.get("best_qwk", -1.0))
        stale_epochs = int(resume_state.get("stale_epochs", 0))
        print(f"Resuming from epoch {start_epoch}; best QWK={best_qwk:.4f}")

    optimizer = build_optimizer(model, args)
    updates_per_epoch = math.ceil(len(loaders["train"]) / args.accum_steps)
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_updates=args.epochs * updates_per_epoch,
        warmup_updates=round(args.warmup_epochs * updates_per_epoch),
        min_lr=args.min_lr,
    )
    global_update = 0
    if resume_state and checkpoint_metadata(resume_state).schema_version > 0:
        optimizer.load_state_dict(resume_state["optimizer"])
        scaler.load_state_dict(resume_state.get("scaler", {}))
        scheduler.load_state_dict(resume_state.get("scheduler", {}))
        global_update = int(
            resume_state.get("global_update", start_epoch * updates_per_epoch)
        )
        restore_rng_state(resume_state.get("rng_state", {}), loader_generator)

    history_path = args.output_dir / "history.jsonl"
    for epoch in range(start_epoch, args.epochs):
        train_loss, gradient_norm, global_update = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            scaler,
            device,
            args,
            amp_enabled,
            scheduler,
            global_update,
        )
        validation = evaluate(
            model,
            loaders["val"],
            criterion,
            device,
            args.loss,
            amp_enabled,
        )
        metrics = calculate_metrics(
            validation.targets,
            validation.predictions,
            validation.probabilities,
        )
        current_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": validation.loss,
            "gradient_norm": gradient_norm,
            "lr_min": min(current_lrs),
            "lr_max": max(current_lrs),
            "global_update": global_update,
            **metrics,
        }
        print(json.dumps(record, ensure_ascii=False))
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        improved = metrics["qwk"] > best_qwk
        if improved:
            best_qwk = metrics["qwk"]
            stale_epochs = 0
            save_checkpoint(
                args.output_dir / "checkpoint-best.pth",
                model,
                optimizer,
                scaler,
                epoch,
                best_qwk,
                stale_epochs,
                args,
                metadata,
                scheduler,
                global_update,
                loader_generator,
                include_training_state=False,
            )
        else:
            stale_epochs += 1
        save_checkpoint(
            args.output_dir / "checkpoint-last.pth",
            model,
            optimizer,
            scaler,
            epoch,
            best_qwk,
            stale_epochs,
            args,
            metadata,
            scheduler,
            global_update,
            loader_generator,
            include_training_state=True,
        )
        if stale_epochs >= args.patience:
            print(f"Early stopping after {stale_epochs} epochs without QWK improvement")
            break

    best = load_torch_checkpoint(args.output_dir / "checkpoint-best.pth")
    model.load_state_dict(best["model"])
    model.to(device)
    test_result = evaluate(
        model,
        loaders["test"],
        criterion,
        device,
        args.loss,
        amp_enabled,
    )
    metrics = {
        "loss": test_result.loss,
        **calculate_metrics(
            test_result.targets,
            test_result.predictions,
            test_result.probabilities,
        ),
    }
    save_evaluation_artifacts(
        args.output_dir,
        test_result.targets,
        test_result.predictions,
        test_result.probabilities,
        test_result.image_ids,
        metrics,
    )
    print("Final test metrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
