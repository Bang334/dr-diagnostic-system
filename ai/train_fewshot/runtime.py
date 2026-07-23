"""Runtime utilities owned by the grading-connected few-shot module.

The helpers in this module deliberately load only an existing grading
checkpoint. They never download a new backbone and never include the held-out
test split in training or model selection.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset

from ai.grading.train import (
    IMAGE_EXTENSIONS,
    NUM_CLASSES,
    find_predefined_splits,
    scan_classification_split,
)
from ai.preprocessing.fundus_prep import preprocess_fundus_array


@dataclass
class GradingCheckpoint:
    model: nn.Module
    saved_args: argparse.Namespace
    state: Dict[str, Any]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_fresh_output_dir(output_dir: Path) -> Path:
    """Create an empty run directory and refuse to mix experiment artifacts."""
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Choose a new run name."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _portable_torch_load(path: Path, map_location: Any) -> Dict[str, Any]:
    """Load checkpoints created on Linux when tests run on Windows."""
    original_posix_path = pathlib.PosixPath
    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath
    try:
        state = torch.load(path, map_location=map_location, weights_only=False)
    finally:
        pathlib.PosixPath = original_posix_path
    if not isinstance(state, dict):
        raise ValueError("Checkpoint must contain a dictionary state")
    return state


def load_grading_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    *,
    require_ce: bool = True,
) -> GradingCheckpoint:
    """Reconstruct the grading model without downloading pretrained weights."""
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = _portable_torch_load(checkpoint_path, map_location="cpu")
    if "model" not in state or "args" not in state:
        raise ValueError(
            "Expected a grading checkpoint containing both 'model' and 'args'"
        )

    raw_args = state["args"]
    if isinstance(raw_args, argparse.Namespace):
        saved_args = raw_args
    elif isinstance(raw_args, Mapping):
        saved_args = argparse.Namespace(**dict(raw_args))
    else:
        raise ValueError("Checkpoint field 'args' must be a mapping")

    loss_name = getattr(saved_args, "loss", "ce")
    if require_ce and loss_name != "ce":
        raise ValueError(
            "This research pipeline currently requires a cross-entropy checkpoint; "
            f"received loss={loss_name!r}"
        )
    image_size = int(getattr(saved_args, "image_size", 224))
    model_source = getattr(saved_args, "model_source", "retfound")
    output_dim = NUM_CLASSES if loss_name == "ce" else NUM_CLASSES - 1

    if model_source == "retfound":
        model = timm.create_model(
            "vit_large_patch14_dinov2.lvd142m",
            pretrained=False,
            img_size=image_size,
            num_classes=output_dim,
        )
    elif model_source == "timm":
        model_name = getattr(saved_args, "model_name", None)
        if not model_name:
            raise ValueError("timm checkpoint does not record model_name")
        model = timm.create_model(model_name, pretrained=False, num_classes=output_dim)
    else:
        raise ValueError(f"Unsupported model_source in checkpoint: {model_source!r}")

    model.load_state_dict(state["model"], strict=True)
    model.to(device)
    return GradingCheckpoint(model=model, saved_args=saved_args, state=state)


def load_split_frames(dataset_dir: Path) -> Tuple[Dict[str, pd.DataFrame], Path]:
    """Load fixed train/val/test manifests and verify path-level separation."""
    split_paths = find_predefined_splits(dataset_dir)
    dataset_root = next(iter(split_paths.values())).parent
    frames = {
        name: scan_classification_split(path, dataset_root)
        for name, path in split_paths.items()
    }

    owners: Dict[str, str] = {}
    for split_name, frame in frames.items():
        for raw_path in frame["image_path"]:
            resolved = os.path.normcase(os.fspath(Path(raw_path).resolve()))
            previous = owners.get(resolved)
            if previous is not None:
                raise ValueError(
                    f"Image path appears in both {previous} and {split_name}: {raw_path}"
                )
            owners[resolved] = split_name
    return frames, dataset_root


def discover_images(root_dir: Path) -> List[Path]:
    root_dir = root_dir.expanduser().resolve()
    if not root_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root_dir}")
    images = sorted(
        path.resolve()
        for path in root_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No supported fundus images found under {root_dir}")
    return images


def prepare_deepdrid_target(source_root: Path, output_dir: Path) -> Path:
    """Convert official DeepDRiD v1.1 metadata into fixed class-folder splits."""
    source_root = source_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    source_id = "github:deepdrdoc/DeepDRiD@v1.1"
    marker = output_dir / ".prepared_source"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == source_id:
        return output_dir

    regular_candidates = [
        path for path in source_root.rglob("regular_fundus_images") if path.is_dir()
    ]
    if len(regular_candidates) != 1:
        raise ValueError(
            f"Expected exactly one regular_fundus_images directory under {source_root}; "
            f"found {len(regular_candidates)}"
        )
    regular_root = regular_candidates[0]
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for split_name in ("train", "validation", "test"):
        for grade in range(NUM_CLASSES):
            (output_dir / split_name / str(grade)).mkdir(parents=True, exist_ok=True)

    def link_rows(
        frame: pd.DataFrame,
        image_root: Path,
        split_name: str,
        label_column: str,
    ) -> None:
        invalid = sorted(set(frame[label_column].astype(int)) - set(range(NUM_CLASSES)))
        if invalid:
            raise ValueError(f"DeepDRiD labels outside 0..4: {invalid}")
        for row in frame.itertuples(index=False):
            image_id = str(row.image_id)
            patient_id = image_id.split("_", 1)[0]
            source = image_root / patient_id / f"{image_id}.jpg"
            if not source.is_file():
                raise FileNotFoundError(f"Missing DeepDRiD image: {source}")
            grade = int(getattr(row, label_column))
            destination = output_dir / split_name / str(grade) / source.name
            try:
                os.link(source, destination)
            except OSError:
                destination.symlink_to(source)

    official_splits = {
        "train": ("regular-fundus-training", "regular-fundus-training.csv"),
        "validation": (
            "regular-fundus-validation",
            "regular-fundus-validation.csv",
        ),
    }
    for split_name, (folder_name, csv_name) in official_splits.items():
        folder = regular_root / folder_name
        frame = pd.read_csv(folder / csv_name)
        required = {"image_id", "left_eye_DR_Level", "right_eye_DR_Level"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"DeepDRiD {csv_name} is missing columns: {sorted(missing)}")
        frame["diagnosis"] = (
            frame["left_eye_DR_Level"]
            .combine_first(frame["right_eye_DR_Level"])
            .astype(int)
        )
        link_rows(frame, folder / "Images", split_name, "diagnosis")

    evaluation = regular_root / "Online-Challenge1&2-Evaluation"
    test_frame = pd.read_excel(evaluation / "Challenge1_labels.xlsx")
    required_test = {"image_id", "DR_Levels"}
    missing_test = required_test.difference(test_frame.columns)
    if missing_test:
        raise ValueError(
            f"DeepDRiD Challenge1 labels are missing columns: {sorted(missing_test)}"
        )
    test_frame = test_frame.rename(columns={"DR_Levels": "diagnosis"})
    link_rows(test_frame, evaluation / "Images", "test", "diagnosis")

    marker.write_text(source_id, encoding="utf-8")
    counts = {
        split: sum(1 for path in (output_dir / split).rglob("*.jpg"))
        for split in ("train", "validation", "test")
    }
    print(f"Prepared DeepDRiD patient splits: {counts}")
    return output_dir


def assert_unlabeled_is_external(
    unlabeled_paths: Sequence[Path], frames: Mapping[str, pd.DataFrame]
) -> None:
    """Reject path or byte-identical reuse of fixed split images as unlabeled."""
    labeled_image_paths = [
        Path(path).resolve()
        for frame in frames.values()
        for path in frame["image_path"]
    ]
    labeled_paths = {
        os.path.normcase(os.fspath(path)) for path in labeled_image_paths
    }
    overlap = [
        path
        for path in unlabeled_paths
        if os.path.normcase(os.fspath(path.resolve())) in labeled_paths
    ]
    if overlap:
        preview = ", ".join(os.fspath(path) for path in overlap[:3])
        raise ValueError(
            "Unlabeled data overlaps a fixed labeled split. Keep train/val/test "
            f"separate. Examples: {preview}"
        )

    # ZIP extraction can give copied split images different absolute paths. Use
    # the inexpensive name/size signature first, then hash only candidates.
    labeled_by_signature: Dict[Tuple[str, int], List[Path]] = {}
    for path in labeled_image_paths:
        signature = (path.name.casefold(), path.stat().st_size)
        labeled_by_signature.setdefault(signature, []).append(path)

    digest_cache: Dict[Path, str] = {}

    def digest(path: Path) -> str:
        cached = digest_cache.get(path)
        if cached is not None:
            return cached
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        value = hasher.hexdigest()
        digest_cache[path] = value
        return value

    copied: List[Path] = []
    for path in unlabeled_paths:
        signature = (path.name.casefold(), path.stat().st_size)
        candidates = labeled_by_signature.get(signature, [])
        if candidates and any(digest(path) == digest(candidate) for candidate in candidates):
            copied.append(path)
    if copied:
        preview = ", ".join(os.fspath(path) for path in copied[:3])
        raise ValueError(
            "Unlabeled data contains byte-identical copies from a fixed labeled "
            f"split. Examples: {preview}"
        )


class UnlabeledFundusDataset(Dataset):
    def __init__(
        self,
        image_paths: Sequence[Path],
        image_size: int,
        transform: Any,
        *,
        enhance: bool = False,
    ) -> None:
        self.image_paths = list(image_paths)
        self.image_size = image_size
        self.transform = transform
        self.enhance = enhance

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_image(self, path: Path) -> torch.Tensor:
        image_bgr = cv2.imread(os.fspath(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        image_bgr = preprocess_fundus_array(
            image_bgr,
            self.image_size,
            enhance=self.enhance,
        )
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return self.transform(Image.fromarray(image_rgb))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        path = self.image_paths[index]
        return self._load_image(path), os.fspath(path)


class PseudoLabeledFundusDataset(UnlabeledFundusDataset):
    def __init__(
        self,
        pseudo_labels: pd.DataFrame,
        image_size: int,
        transform: Any,
        *,
        pseudo_weight: float,
        enhance: bool = False,
    ) -> None:
        required = {"image_path", "pseudo_label", "confidence"}
        missing = required.difference(pseudo_labels.columns)
        if missing:
            raise ValueError(f"Pseudo-label manifest is missing columns: {sorted(missing)}")
        self.frame = pseudo_labels.reset_index(drop=True)
        super().__init__(
            [Path(path) for path in self.frame["image_path"]],
            image_size,
            transform,
            enhance=enhance,
        )
        self.pseudo_weight = pseudo_weight

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str, float]:
        image, path = super().__getitem__(index)
        row = self.frame.iloc[index]
        weight = self.pseudo_weight * float(row["confidence"])
        return image, int(row["pseudo_label"]), path, weight


class WeightedLabeledDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str, float]:
        image, label, image_id = self.dataset[index]
        return image, int(label), str(image_id), 1.0


def checkpoint_args_with_metadata(
    saved_args: argparse.Namespace,
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    result = dict(vars(saved_args))
    result.update(metadata)
    return result


def save_classifier_checkpoint(
    path: Path,
    model: nn.Module,
    checkpoint_args: Mapping[str, Any],
    *,
    epoch: int,
    best_qwk: float,
    parent_checkpoint: Path,
    best_epoch: Optional[int] = None,
    stale_epochs: Optional[int] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state: Dict[str, Any] = {
        "model": model.state_dict(),
        "epoch": epoch,
        "best_qwk": float(best_qwk),
        "args": dict(checkpoint_args),
        "parent_checkpoint": os.fspath(parent_checkpoint.resolve()),
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    if best_epoch is not None:
        state["best_epoch"] = int(best_epoch)
    if stale_epochs is not None:
        state["stale_epochs"] = int(stale_epochs)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary_path)
    temporary_path.replace(path)
