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
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from timm.layers.pos_embed import resample_abs_pos_embed
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

from ai.grading.preprocessing import PREPROCESSING_RECIPES, PreprocessingSpec, preprocess_fundus
from ai.grading.taxonomy import CLASS_NAMES, NUM_CLASSES
from ai.grading.backbones import BACKBONE_PRESETS


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
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
        "--architecture",
        choices=tuple(BACKBONE_PRESETS),
        default="convnext",
        help="Auditable backbone family; --model-name may override its preset",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="timm model used when --model-source=timm",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--loss",
        choices=("ce", "coral"),
        default="ce",
        help="Cross-entropy baseline or ordinal CORAL-style thresholds",
    )
    parser.add_argument(
        "--balance",
        choices=("none", "effective", "sampler"),
        default="none",
        help="Use only one rebalancing method per experiment",
    )
    parser.add_argument("--effective-beta", type=float, default=0.9999)

    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=8)
    parser.add_argument("--head-lr", type=float, default=5e-5)
    parser.add_argument("--backbone-lr", type=float, default=5e-6)
    parser.add_argument("--min-lr", type=float, default=5e-7)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate --resume checkpoint on the test split without training",
    )
    parser.add_argument(
        "--preprocessing",
        choices=PREPROCESSING_RECIPES,
        default="rgb_crop",
        help="Fundus preprocessing recipe stored with the checkpoint",
    )
    parser.add_argument("--enhance", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-amp", action="store_true")
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
    if args.loss == "coral" and args.balance == "effective":
        raise ValueError(
            "Use --balance none or --balance sampler with --loss coral so two "
            "reweighting methods are not mixed."
        )
    if args.val_size <= 0 or args.test_size <= 0:
        raise ValueError("Validation and test sizes must be positive")
    if args.val_size + args.test_size >= 1:
        raise ValueError("val-size + test-size must be below 1")
    if args.accum_steps < 1:
        raise ValueError("accum-steps must be at least 1")
    if args.eval_only and args.resume is None:
        raise ValueError("--eval-only requires --resume CHECKPOINT")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_or_create_csv_splits(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    split_dir = args.split_dir or (args.output_dir / "splits")
    split_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: split_dir / f"{name}.csv" for name in ("train", "val", "test")}
    if all(path.exists() for path in paths.values()):
        print(f"Reusing fixed splits from {split_dir}")
        return {name: pd.read_csv(path) for name, path in paths.items()}

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


def load_predefined_splits(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
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


def load_splits(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    if args.dataset_dir is not None:
        return load_predefined_splits(args)
    return load_or_create_csv_splits(args)


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
        recipe = getattr(self.args, "preprocessing", "rgb_crop")
        if getattr(self.args, "enhance", False) and recipe == "rgb_crop":
            recipe = "ben_graham"
        image_bgr = preprocess_fundus(
            image_bgr,
            PreprocessingSpec(recipe=recipe, image_size=self.args.image_size),
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


def build_sampler(labels: list[int]) -> WeightedRandomSampler:
    counts = np.bincount(np.asarray(labels), minlength=NUM_CLASSES)
    class_weights = 1.0 / np.maximum(counts, 1)
    sample_weights = torch.tensor([class_weights[label] for label in labels], dtype=torch.double)
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


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
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
    output_dim = NUM_CLASSES if args.loss == "ce" else NUM_CLASSES - 1
    if args.model_source == "retfound":
        return load_retfound_dinov2(args, output_dim)
    model_name = args.model_name or BACKBONE_PRESETS[args.architecture]
    print(f"Loading {args.architecture} backbone via timm model {model_name}")
    return timm.create_model(model_name, pretrained=True, num_classes=output_dim)


def is_head_parameter(name: str) -> bool:
    parts = set(name.split("."))
    return bool(parts.intersection({"head", "classifier", "fc"}))


def configure_trainable(model: nn.Module, freeze_backbone: bool) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = (not freeze_backbone) or is_head_parameter(name)
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Trainable parameters: {count / 1e6:.2f}M; backbone frozen={freeze_backbone}")
    if count == 0:
        raise RuntimeError("No trainable parameters found; classifier name is unsupported")


def build_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    head, backbone = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (head if is_head_parameter(name) else backbone).append(parameter)
    groups: list[dict[str, Any]] = []
    if backbone:
        groups.append({"params": backbone, "lr": args.backbone_lr, "base_lr": args.backbone_lr})
    if head:
        groups.append({"params": head, "lr": args.head_lr, "base_lr": args.head_lr})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


class CoralLoss(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        levels = torch.arange(NUM_CLASSES - 1, device=targets.device).unsqueeze(0)
        ordinal_targets = (targets.unsqueeze(1) > levels).float()
        return nn.functional.binary_cross_entropy_with_logits(logits, ordinal_targets)


def build_criterion(
    args: argparse.Namespace, train_labels: list[int], device: torch.device
) -> nn.Module:
    if args.loss == "coral":
        return CoralLoss()
    weights = None
    if args.balance == "effective":
        weights = effective_number_weights(train_labels, args.effective_beta, device)
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)


def predictions_from_logits(logits: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == "coral":
        return (torch.sigmoid(logits) > 0.5).sum(dim=1)
    return logits.argmax(dim=1)


def create_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:  # PyTorch before the device-aware AMP API
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
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    sample_count = 0
    progress = tqdm(loader, desc="train", leave=False)
    for step, (images, targets, _) in enumerate(progress):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss / args.accum_steps).backward()
        should_step = (step + 1) % args.accum_steps == 0 or step + 1 == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        sample_count += batch_size
        progress.set_postfix(loss=f"{running_loss / sample_count:.4f}")
    return running_loss / max(sample_count, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    loss_name: str,
    amp_enabled: bool,
) -> tuple[float, list[int], list[int], list[str]]:
    model.eval()
    total_loss = 0.0
    sample_count = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []
    all_ids: list[str] = []
    for images, targets, image_ids in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)
        predictions = predictions_from_logits(logits, loss_name)
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        sample_count += batch_size
        all_targets.extend(targets.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())
        all_ids.extend(image_ids)
    return total_loss / max(sample_count, 1), all_targets, all_predictions, all_ids


def calculate_metrics(targets: list[int], predictions: list[int]) -> dict[str, Any]:
    report = classification_report(
        targets,
        predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return {
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


def set_cosine_lr(
    optimizer: torch.optim.Optimizer, epoch: int, args: argparse.Namespace
) -> None:
    if epoch < args.freeze_epochs:
        return
    total = max(args.epochs - args.freeze_epochs - 1, 1)
    progress = min(max((epoch - args.freeze_epochs) / total, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        base_lr = group["base_lr"]
        group["lr"] = args.min_lr + (base_lr - args.min_lr) * cosine


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    epoch: int,
    best_qwk: float,
    stale_epochs: int,
    freeze_backbone: bool,
    args: argparse.Namespace,
    *,
    include_training_state: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "model": model.state_dict(),
        "epoch": epoch,
        "best_qwk": best_qwk,
        "stale_epochs": stale_epochs,
        "freeze_backbone": freeze_backbone,
        "args": vars(args),
        "grading_contract": {
            "schema_version": 1,
            "standard": "ICDR",
            "etdrs_correspondence": True,
            "class_names": CLASS_NAMES,
            "preprocessing": PreprocessingSpec(
                recipe=(
                    "ben_graham"
                    if getattr(args, "enhance", False)
                    and getattr(args, "preprocessing", "rgb_crop") == "rgb_crop"
                    else getattr(args, "preprocessing", "rgb_crop")
                ),
                image_size=args.image_size,
            ).to_dict(),
        },
    }
    if include_training_state:
        state.update(
            {
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
            }
        )
    torch.save(state, path)


def save_evaluation_artifacts(
    output_dir: Path,
    targets: list[int],
    predictions: list[int],
    image_ids: list[str],
    metrics: dict[str, Any],
) -> None:
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    pd.DataFrame(
        {"image_id": image_ids, "true_grade": targets, "predicted_grade": predictions}
    ).to_csv(output_dir / "test_predictions.csv", index=False)

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


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. In Colab choose a GPU runtime first.")
    device = torch.device("cuda")
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {torch.cuda.get_device_name(0)} ({memory_gb:.1f} GB)")
    amp_enabled = not args.no_amp

    splits = load_splits(args)
    train_transform, eval_transform = build_transforms(args.image_size)
    datasets = {
        "train": FundusDataset(splits["train"], args, train_transform),
        "val": FundusDataset(splits["val"], args, eval_transform),
        "test": FundusDataset(splits["test"], args, eval_transform),
    }
    train_labels = splits["train"][args.label_column].astype(int).tolist()
    sampler = build_sampler(train_labels) if args.balance == "sampler" else None
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
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
    criterion = build_criterion(args, train_labels, device)
    scaler = create_scaler(amp_enabled)
    start_epoch, best_qwk, stale_epochs = 0, -1.0, 0
    resume_state = None
    if args.resume:
        resume_state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_state["model"])
        start_epoch = int(resume_state["epoch"]) + 1
        best_qwk = float(resume_state.get("best_qwk", -1.0))
        stale_epochs = int(resume_state.get("stale_epochs", 0))
        print(f"Resuming from epoch {start_epoch}; best QWK={best_qwk:.4f}")

    if args.eval_only:
        test_loss, targets, predictions, image_ids = evaluate(
            model,
            loaders["test"],
            criterion,
            device,
            args.loss,
            amp_enabled,
        )
        metrics = {"loss": test_loss, **calculate_metrics(targets, predictions)}
        save_evaluation_artifacts(
            args.output_dir, targets, predictions, image_ids, metrics
        )
        print(f"Test metrics for checkpoint epoch {start_epoch - 1}:")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        return

    freeze_backbone = start_epoch < args.freeze_epochs
    configure_trainable(model, freeze_backbone)
    optimizer = build_optimizer(model, args)
    if resume_state and bool(resume_state.get("freeze_backbone")) == freeze_backbone:
        optimizer.load_state_dict(resume_state["optimizer"])
        scaler.load_state_dict(resume_state.get("scaler", {}))

    history_path = args.output_dir / "history.jsonl"
    for epoch in range(start_epoch, args.epochs):
        if freeze_backbone and epoch >= args.freeze_epochs:
            freeze_backbone = False
            configure_trainable(model, freeze_backbone=False)
            optimizer = build_optimizer(model, args)
        set_cosine_lr(optimizer, epoch, args)

        train_loss = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            scaler,
            device,
            args,
            amp_enabled,
        )
        val_loss, targets, predictions, _ = evaluate(
            model,
            loaders["val"],
            criterion,
            device,
            args.loss,
            amp_enabled,
        )
        metrics = calculate_metrics(targets, predictions)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
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
                freeze_backbone,
                args,
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
            freeze_backbone,
            args,
            include_training_state=True,
        )
        if stale_epochs >= args.patience:
            print(f"Early stopping after {stale_epochs} epochs without QWK improvement")
            break

    best = torch.load(args.output_dir / "checkpoint-best.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"])
    model.to(device)
    test_loss, targets, predictions, image_ids = evaluate(
        model,
        loaders["test"],
        criterion,
        device,
        args.loss,
        amp_enabled,
    )
    metrics = {"loss": test_loss, **calculate_metrics(targets, predictions)}
    save_evaluation_artifacts(args.output_dir, targets, predictions, image_ids, metrics)
    print("Final test metrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
