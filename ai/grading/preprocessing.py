"""Fundus preprocessing owned by the grading module.

Every recipe is deterministic and serializable in checkpoint metadata so that
training, evaluation, and deployment use the same image contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


PREPROCESSING_RECIPES = ("rgb_crop", "green", "clahe", "ben_graham")


@dataclass(frozen=True)
class PreprocessingSpec:
    recipe: str = "rgb_crop"
    image_size: int = 224
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    ben_graham_sigma: float = 10.0

    def __post_init__(self) -> None:
        if self.recipe not in PREPROCESSING_RECIPES:
            raise ValueError(f"Unknown preprocessing recipe: {self.recipe}")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")

    def to_dict(self) -> dict:
        return asdict(self)


def _crop_fundus(image: np.ndarray) -> np.ndarray:
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a non-empty BGR image with three channels")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > max(5, int(np.percentile(gray, 5)))
    points = cv2.findNonZero(mask.astype(np.uint8))
    if points is None:
        raise ValueError("No fundus field detected")
    x, y, w, h = cv2.boundingRect(points)
    return image[y : y + h, x : x + w]


def _green_channel(image: np.ndarray) -> np.ndarray:
    green = image[:, :, 1]
    return cv2.cvtColor(green, cv2.COLOR_GRAY2BGR)


def _clahe_green(image: np.ndarray, spec: PreprocessingSpec) -> np.ndarray:
    green = image[:, :, 1]
    clahe = cv2.createCLAHE(
        clipLimit=spec.clahe_clip_limit,
        tileGridSize=(spec.clahe_grid_size, spec.clahe_grid_size),
    )
    return cv2.cvtColor(clahe.apply(green), cv2.COLOR_GRAY2BGR)


def _ben_graham(image: np.ndarray, spec: PreprocessingSpec) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), spec.ben_graham_sigma)
    return cv2.addWeighted(image, 4.0, blurred, -4.0, 128.0)


def preprocess_fundus(image_bgr: np.ndarray, spec: PreprocessingSpec) -> np.ndarray:
    image = _crop_fundus(image_bgr)
    image = cv2.resize(image, (spec.image_size, spec.image_size), interpolation=cv2.INTER_AREA)
    if spec.recipe == "green":
        image = _green_channel(image)
    elif spec.recipe == "clahe":
        image = _clahe_green(image, spec)
    elif spec.recipe == "ben_graham":
        image = _ben_graham(image, spec)
    return np.ascontiguousarray(image, dtype=np.uint8)

