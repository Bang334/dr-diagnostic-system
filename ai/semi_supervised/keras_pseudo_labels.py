"""Pure pseudo-label selection for the Keras ordinal teacher."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ai.keras_grading.ordinal import OrdinalPrediction


@dataclass(frozen=True)
class PseudoLabel:
    image_path: Path
    grade: int
    confidence: float
    ordinal_probabilities: tuple[float, float, float, float]


def ordinal_decision_confidence(prediction: OrdinalPrediction) -> float:
    """Return confidence in all four calibrated threshold decisions.

    Zero means at least one boundary is exactly undecided; one means every
    boundary probability is at the extreme supporting the decoded grade.
    """
    probabilities = prediction.ordinal_probabilities
    thresholds = prediction.thresholds
    decisions = probabilities > thresholds
    distances = np.abs(probabilities - thresholds)
    available = np.where(decisions, 1.0 - thresholds, thresholds)
    normalized = distances / np.maximum(available, 1e-7)
    return float(np.clip(np.min(normalized), 0.0, 1.0))


def select_pseudo_labels(
    predictions: Iterable[tuple[Path, OrdinalPrediction]],
    *,
    minimum_confidence: float,
    max_per_class: int = 0,
) -> list[PseudoLabel]:
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must be in [0, 1]")
    if max_per_class < 0:
        raise ValueError("max_per_class cannot be negative")

    candidates: list[PseudoLabel] = []
    for path, prediction in predictions:
        confidence = ordinal_decision_confidence(prediction)
        if confidence < minimum_confidence:
            continue
        candidates.append(
            PseudoLabel(
                image_path=Path(path),
                grade=prediction.grade,
                confidence=confidence,
                ordinal_probabilities=tuple(
                    float(value) for value in prediction.ordinal_probabilities
                ),
            )
        )

    candidates.sort(key=lambda item: (-item.confidence, str(item.image_path)))
    if max_per_class == 0:
        return candidates

    counts = {grade: 0 for grade in range(5)}
    selected: list[PseudoLabel] = []
    for candidate in candidates:
        if counts[candidate.grade] >= max_per_class:
            continue
        counts[candidate.grade] += 1
        selected.append(candidate)
    return selected


def class_counts(labels: Sequence[PseudoLabel]) -> dict[int, int]:
    return {
        grade: sum(label.grade == grade for label in labels) for grade in range(5)
    }
