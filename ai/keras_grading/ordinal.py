"""TensorFlow-independent CORAL ordinal decoding utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


CLASS_NAMES = (
    "No DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "Proliferative DR",
)
DEFAULT_ORDINAL_THRESHOLDS = np.array(
    [0.55, 0.50, 0.435, 0.31], dtype=np.float32
)


@dataclass(frozen=True)
class OrdinalPrediction:
    grade: int
    label: str
    confidence: float
    class_probabilities: np.ndarray
    ordinal_probabilities: np.ndarray
    thresholds: np.ndarray

    def as_dict(self) -> dict[str, object]:
        return {
            "dr_grade": self.grade,
            "dr_label": self.label,
            "confidence": round(self.confidence, 4),
            "probabilities": {
                name: round(float(probability), 4)
                for name, probability in zip(
                    CLASS_NAMES, self.class_probabilities, strict=True
                )
            },
            "ordinal_probabilities": [
                round(float(value), 4) for value in self.ordinal_probabilities
            ],
            "ordinal_thresholds": [
                round(float(value), 4) for value in self.thresholds
            ],
        }


def validate_thresholds(thresholds: Sequence[float]) -> np.ndarray:
    values = np.asarray(thresholds, dtype=np.float32).reshape(-1)
    if values.shape != (4,):
        raise ValueError(f"Expected 4 ordinal thresholds, received {values.shape}")
    if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("Ordinal thresholds must be finite values in [0, 1]")
    return values


def enforce_monotonic(ordinal_probabilities: Sequence[float]) -> np.ndarray:
    values = np.asarray(ordinal_probabilities, dtype=np.float32).reshape(-1)
    if values.shape != (4,):
        raise ValueError(f"Expected 4 ordinal probabilities, received {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("Ordinal probabilities must be finite")
    return np.minimum.accumulate(np.clip(values, 0.0, 1.0))


def ordinal_to_class_probabilities(
    ordinal_probabilities: Sequence[float],
) -> np.ndarray:
    """Convert P(Y>k) into a normalized five-grade distribution."""
    ordinal = enforce_monotonic(ordinal_probabilities)
    probabilities = np.array(
        [
            1.0 - ordinal[0],
            ordinal[0] - ordinal[1],
            ordinal[1] - ordinal[2],
            ordinal[2] - ordinal[3],
            ordinal[3],
        ],
        dtype=np.float32,
    )
    probabilities = np.clip(probabilities, 0.0, 1.0)
    total = float(probabilities.sum())
    if total <= 0:
        raise ValueError("Ordinal output produced an invalid class distribution")
    return probabilities / total


def decode_ordinal(
    ordinal_probabilities: Sequence[float],
    thresholds: Sequence[float] = DEFAULT_ORDINAL_THRESHOLDS,
) -> OrdinalPrediction:
    ordinal = enforce_monotonic(ordinal_probabilities)
    calibrated_thresholds = validate_thresholds(thresholds)
    grade = int(np.sum(ordinal > calibrated_thresholds))
    class_probabilities = ordinal_to_class_probabilities(ordinal)
    return OrdinalPrediction(
        grade=grade,
        label=CLASS_NAMES[grade],
        confidence=float(class_probabilities[grade]),
        class_probabilities=class_probabilities,
        ordinal_probabilities=ordinal,
        thresholds=calibrated_thresholds,
    )
