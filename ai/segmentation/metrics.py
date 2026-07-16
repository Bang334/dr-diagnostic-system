"""Aggregate multilabel lesion-segmentation metrics and threshold calibration."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import torch

from ai.segmentation.data import LESION_NAMES


class SegmentationMeter:
    def __init__(self, thresholds: Sequence[float]) -> None:
        if len(thresholds) != len(LESION_NAMES):
            raise ValueError(f"Expected {len(LESION_NAMES)} thresholds")
        self.thresholds = torch.tensor(thresholds, dtype=torch.float32).view(1, -1, 1, 1)
        self.tp = torch.zeros(len(LESION_NAMES), dtype=torch.float64)
        self.fp = torch.zeros(len(LESION_NAMES), dtype=torch.float64)
        self.fn = torch.zeros(len(LESION_NAMES), dtype=torch.float64)
        self.tn = torch.zeros(len(LESION_NAMES), dtype=torch.float64)

    def update(self, probabilities: torch.Tensor, targets: torch.Tensor) -> None:
        probabilities = probabilities.detach().cpu()
        targets = targets.detach().cpu() > 0.5
        if probabilities.shape != targets.shape or probabilities.ndim != 4:
            raise ValueError(
                f"Expected matching NCHW probabilities/targets; got "
                f"{tuple(probabilities.shape)} and {tuple(targets.shape)}"
            )
        predictions = probabilities >= self.thresholds
        reduce_dims = (0, 2, 3)
        self.tp += (predictions & targets).sum(dim=reduce_dims).to(torch.float64)
        self.fp += (predictions & ~targets).sum(dim=reduce_dims).to(torch.float64)
        self.fn += (~predictions & targets).sum(dim=reduce_dims).to(torch.float64)
        self.tn += (~predictions & ~targets).sum(dim=reduce_dims).to(torch.float64)

    def compute(self) -> Dict[str, Any]:
        eps = 1e-8
        per_class: Dict[str, Dict[str, float]] = {}
        for index, name in enumerate(LESION_NAMES):
            tp, fp, fn, tn = (
                float(self.tp[index]),
                float(self.fp[index]),
                float(self.fn[index]),
                float(self.tn[index]),
            )
            per_class[name] = {
                "dice": (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps),
                "iou": (tp + eps) / (tp + fp + fn + eps),
                "precision": (tp + eps) / (tp + fp + eps),
                "recall": (tp + eps) / (tp + fn + eps),
                "specificity": (tn + eps) / (tn + fp + eps),
            }
        macro = {
            metric: sum(values[metric] for values in per_class.values()) / len(per_class)
            for metric in ("dice", "iou", "precision", "recall", "specificity")
        }
        return {"macro": macro, "per_class": per_class}


def calibrate_dice_thresholds(
    probability_batches: Iterable[torch.Tensor],
    target_batches: Iterable[torch.Tensor],
    *,
    grid: Sequence[float] = tuple(index / 20.0 for index in range(2, 19)),
) -> List[float]:
    probabilities = torch.cat([batch.detach().cpu() for batch in probability_batches])
    targets = torch.cat([batch.detach().cpu() for batch in target_batches])
    if probabilities.shape != targets.shape:
        raise ValueError("Threshold calibration probabilities and targets do not match")
    thresholds: List[float] = []
    for class_index in range(len(LESION_NAMES)):
        target = targets[:, class_index] > 0.5
        best_threshold = 0.5
        best_dice = -1.0
        for threshold in grid:
            prediction = probabilities[:, class_index] >= threshold
            tp = float((prediction & target).sum())
            fp = float((prediction & ~target).sum())
            fn = float((~prediction & target).sum())
            dice = (2.0 * tp + 1e-8) / (2.0 * tp + fp + fn + 1e-8)
            if dice > best_dice:
                best_dice = dice
                best_threshold = float(threshold)
        thresholds.append(best_threshold)
    return thresholds

