"""Compatibility wrapper for legacy callers of :class:`DRModelHandler`.

New code should depend on ``DRPredictor`` and use ``load_predictor`` instead.
"""

from __future__ import annotations

from ai.grading.artifacts import CLASS_NAMES
from ai.grading.predictor import KerasEfficientNetPredictor


class DRModelHandler:
    """Preserve the former Keras-only interface while delegating to its adapter."""

    def __init__(self, model_path: str):
        self._predictor = None
        self.model = None
        self.input_size = (300, 300)
        self.model_version = "efficientnet_b3_v1.0"
        try:
            self._predictor = KerasEfficientNetPredictor(model_path)
            self.model = self._predictor.model
        except (FileNotFoundError, OSError, ValueError) as error:
            print(f"[x] Could not load legacy Keras grading model: {error}")

    def predict(self, preprocessed_img):
        if self._predictor is None:
            raise ValueError("Model is not loaded; check the configured Keras weights")
        return self._predictor.predict(preprocessed_img).to_api_dict()


__all__ = ["CLASS_NAMES", "DRModelHandler"]
