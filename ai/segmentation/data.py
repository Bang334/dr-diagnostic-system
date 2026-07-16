"""IDRiD adapter and synchronized image/mask preprocessing."""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Dict, Mapping, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ai.preprocessing.fundus_prep import ben_graham_enhance


LESION_NAMES: Tuple[str, ...] = (
    "microaneurysm",
    "hemorrhage",
    "hard_exudate",
)
LESION_MASK_DIRS: Mapping[str, Tuple[str, str]] = {
    "microaneurysm": ("Microaneurysms", "MA"),
    "hemorrhage": ("Haemorrhages", "HE"),
    "hard_exudate": ("Hard Exudates", "EX"),
}
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def find_idrid_segmentation_root(dataset_dir: Path) -> Path:
    dataset_dir = dataset_dir.expanduser().resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    candidates = [dataset_dir]
    candidates.extend(path for path in dataset_dir.rglob("Segmentation") if path.is_dir())
    valid = [
        path
        for path in candidates
        if (path / "Original_Images").is_dir()
        and (path / "Segmentation_Groundtruths").is_dir()
    ]
    unique = list(dict.fromkeys(path.resolve() for path in valid))
    if len(unique) != 1:
        raise ValueError(
            "Expected exactly one IDRiD Segmentation directory containing "
            f"Original_Images and Segmentation_Groundtruths; found {len(unique)}"
        )
    return unique[0]


def _official_records(segmentation_root: Path, split_folder: str) -> pd.DataFrame:
    image_dir = segmentation_root / "Original_Images" / split_folder
    groundtruth_root = segmentation_root / "Segmentation_Groundtruths" / split_folder
    images = sorted(image_dir.glob("*.jpg"))
    if not images:
        raise ValueError(f"No IDRiD JPG images found in {image_dir}")

    records = []
    for image_path in images:
        record: Dict[str, object] = {
            "image_id": image_path.stem,
            "image_path": os.fspath(image_path.resolve()),
            "official_split": split_folder,
        }
        for lesion_name, (mask_dir, suffix) in LESION_MASK_DIRS.items():
            mask_path = groundtruth_root / mask_dir / f"{image_path.stem}_{suffix}.tif"
            record[f"{lesion_name}_mask"] = (
                os.fspath(mask_path.resolve()) if mask_path.is_file() else ""
            )
        records.append(record)
    return pd.DataFrame.from_records(records)


def build_idrid_manifest(dataset_dir: Path) -> pd.DataFrame:
    """Discover all 81 official segmentation images and their three masks."""
    root = find_idrid_segmentation_root(dataset_dir)
    frame = pd.concat(
        [
            _official_records(root, "Training Set"),
            _official_records(root, "Testing Set"),
        ],
        ignore_index=True,
    )
    if frame["image_id"].duplicated().any():
        raise ValueError("IDRiD segmentation manifest contains duplicate image IDs")
    expected = {"Training Set": 54, "Testing Set": 27}
    counts = frame["official_split"].value_counts().to_dict()
    if counts != expected:
        raise ValueError(f"Unexpected IDRiD official split counts: {counts}; expected {expected}")

    # Official IDRiD has masks for all MA/EX images and one HE-negative training
    # image without a file. Reject partial extractions instead of silently
    # treating missing annotations as negatives.
    missing = {
        lesion: int((frame[f"{lesion}_mask"] == "").sum()) for lesion in LESION_NAMES
    }
    if missing["microaneurysm"] != 0 or missing["hard_exudate"] != 0:
        raise ValueError(f"IDRiD extraction is missing required masks: {missing}")
    if missing["hemorrhage"] > 1:
        raise ValueError(f"IDRiD extraction is missing too many hemorrhage masks: {missing}")
    return frame


def create_idrid_splits(
    dataset_dir: Path,
    split_dir: Path,
    *,
    seed: int = 42,
    val_fraction: float = 0.2,
) -> Dict[str, pd.DataFrame]:
    """Split only official training data; preserve the 27-image official test."""
    if not 0.0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be between 0 and 0.5")
    split_dir = split_dir.expanduser().resolve()
    paths = {name: split_dir / f"{name}.csv" for name in ("train", "val", "test")}
    if all(path.is_file() for path in paths.values()):
        return {name: pd.read_csv(path).fillna("") for name, path in paths.items()}

    frame = build_idrid_manifest(dataset_dir)
    official_train = frame[frame["official_split"] == "Training Set"].reset_index(drop=True)
    official_test = frame[frame["official_split"] == "Testing Set"].reset_index(drop=True)
    val_count = max(1, int(math.ceil(len(official_train) * val_fraction)))
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(official_train))
    val_indices = set(int(index) for index in permutation[:val_count])
    train = official_train[
        [index not in val_indices for index in range(len(official_train))]
    ].reset_index(drop=True)
    val = official_train[
        [index in val_indices for index in range(len(official_train))]
    ].reset_index(drop=True)
    result = {"train": train, "val": val, "test": official_test}

    split_dir.mkdir(parents=True, exist_ok=True)
    for name, split in result.items():
        split.to_csv(paths[name], index=False)
    return result


def _fundus_bbox(image_bgr: np.ndarray, tolerance: int = 7) -> Tuple[int, int, int, int]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    foreground = gray > tolerance
    if not foreground.any():
        return 0, image_bgr.shape[0], 0, image_bgr.shape[1]
    rows = np.flatnonzero(foreground.any(axis=1))
    cols = np.flatnonzero(foreground.any(axis=0))
    return int(rows[0]), int(rows[-1] + 1), int(cols[0]), int(cols[-1] + 1)


def _augment(image: np.ndarray, masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        image = np.flip(image, axis=1)
        masks = np.flip(masks, axis=2)
    if random.random() < 0.5:
        image = np.flip(image, axis=0)
        masks = np.flip(masks, axis=1)
    if random.random() < 0.6:
        angle = random.uniform(-15.0, 15.0)
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
        image = cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        masks = np.stack(
            [
                cv2.warpAffine(
                    mask,
                    matrix,
                    (width, height),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                for mask in masks
            ]
        )
    if random.random() < 0.4:
        alpha = random.uniform(0.9, 1.1)
        beta = random.uniform(-10.0, 10.0)
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image), np.ascontiguousarray(masks)


class IDRiDLesionDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        image_size: int,
        *,
        training: bool,
        enhance: bool = False,
    ) -> None:
        self.frame = frame.fillna("").reset_index(drop=True)
        self.image_size = image_size
        self.training = training
        self.enhance = enhance

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row = self.frame.iloc[index]
        image_path = Path(str(row["image_path"]))
        image_bgr = cv2.imread(os.fspath(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read fundus image: {image_path}")

        masks = []
        for lesion_name in LESION_NAMES:
            raw_path = str(row[f"{lesion_name}_mask"])
            if not raw_path:
                mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
            else:
                mask = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise FileNotFoundError(f"Could not read lesion mask: {raw_path}")
                if mask.shape != image_bgr.shape[:2]:
                    raise ValueError(
                        f"Image/mask shape mismatch for {row['image_id']}: "
                        f"{image_bgr.shape[:2]} vs {mask.shape}"
                    )
            masks.append((mask > 0).astype(np.uint8))
        mask_array = np.stack(masks)

        y0, y1, x0, x1 = _fundus_bbox(image_bgr)
        image_bgr = image_bgr[y0:y1, x0:x1]
        mask_array = mask_array[:, y0:y1, x0:x1]
        if self.enhance:
            image_bgr = ben_graham_enhance(image_bgr)
        image_bgr = cv2.resize(
            image_bgr,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )
        mask_array = np.stack(
            [
                cv2.resize(
                    mask,
                    (self.image_size, self.image_size),
                    interpolation=cv2.INTER_NEAREST,
                )
                for mask in mask_array
            ]
        )
        if self.training:
            image_bgr, mask_array = _augment(image_bgr, mask_array)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image_rgb = (image_rgb - IMAGENET_MEAN) / IMAGENET_STD
        image_tensor = torch.from_numpy(np.ascontiguousarray(image_rgb.transpose(2, 0, 1)))
        mask_tensor = torch.from_numpy((mask_array > 0).astype(np.float32))
        return image_tensor, mask_tensor, str(row["image_id"])

