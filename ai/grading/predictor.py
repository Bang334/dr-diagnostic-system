"""Single inference seam for Keras and PyTorch DR grading checkpoints."""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from ai.grading.preprocessing import PreprocessingSpec, preprocess_fundus
from ai.grading.taxonomy import CLASS_NAMES, NUM_CLASSES, grade_definition


@dataclass(frozen=True)
class Prediction:
    grade: int
    probabilities: tuple[float, ...]
    model_version: str
    preprocessing: PreprocessingSpec
    preprocessed_bgr: np.ndarray

    def to_api_dict(self) -> dict:
        definition = grade_definition(self.grade)
        return {
            "dr_grade": self.grade,
            "dr_label": definition.icdr_label,
            "etdrs_level_range": definition.etdrs_range,
            "clinical_definition": definition.clinical_definition,
            "confidence": round(self.probabilities[self.grade], 4),
            "probabilities": {
                name: round(probability, 4)
                for name, probability in zip(CLASS_NAMES, self.probabilities)
            },
            "model_version": self.model_version,
            "preprocessing": self.preprocessing.to_dict(),
        }


@runtime_checkable
class GradingPredictor(Protocol):
    model_version: str

    def predict(self, image_bgr: np.ndarray) -> Prediction: ...


def _normalize_probabilities(values) -> tuple[float, ...]:
    probabilities = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(probabilities) != NUM_CLASSES or not np.isfinite(probabilities).all():
        raise ValueError("Model must return five finite class probabilities")
    total = float(probabilities.sum())
    if total <= 0:
        raise ValueError("Model returned invalid probabilities")
    return tuple(float(value / total) for value in probabilities)


class KerasPredictor:
    def __init__(self, path: Path, preprocessing: PreprocessingSpec | None = None):
        from ai.grading.model_handler import DRModelHandler

        self.handler = DRModelHandler(os.fspath(path))
        if self.handler.model is None:
            raise RuntimeError(f"Could not load Keras grading model: {path}")
        sidecar = path.with_suffix(path.suffix + ".json")
        if preprocessing is None and sidecar.is_file():
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            preprocessing = PreprocessingSpec(**metadata.get("preprocessing", {}))
        self.preprocessing = preprocessing or PreprocessingSpec("rgb_crop", 300)
        self.model_version = self.handler.model_version

    def predict(self, image_bgr: np.ndarray) -> Prediction:
        processed = preprocess_fundus(image_bgr, self.preprocessing)
        result = self.handler.predict(processed)
        probabilities = _normalize_probabilities(result["probabilities"].values())
        return Prediction(
            grade=int(np.argmax(probabilities)),
            probabilities=probabilities,
            model_version=self.model_version,
            preprocessing=self.preprocessing,
            preprocessed_bgr=processed,
        )


class TorchPredictor:
    def __init__(self, path: Path):
        import torch
        import timm

        state = torch.load(path, map_location="cpu", weights_only=False)
        saved = state.get("args", {})
        if saved.get("loss", "ce") != "ce":
            raise ValueError("Production inference currently requires a five-class CE checkpoint")
        model_source = saved.get("model_source", "timm")
        if model_source == "retfound":
            model_name = "vit_large_patch14_dinov2.lvd142m"
        else:
            from ai.grading.backbones import BACKBONE_PRESETS

            model_name = saved.get("model_name") or BACKBONE_PRESETS[saved.get("architecture", "convnext")]
        image_size = int(saved.get("image_size", 224))
        kwargs = {"pretrained": False, "num_classes": NUM_CLASSES}
        if model_source == "retfound":
            kwargs["img_size"] = image_size
        self.model = timm.create_model(model_name, **kwargs)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.torch = torch
        contract_spec = state.get("grading_contract", {}).get("preprocessing")
        self.preprocessing = (
            PreprocessingSpec(**contract_spec)
            if contract_spec
            else PreprocessingSpec(saved.get("preprocessing", "rgb_crop"), image_size)
        )
        self.model_version = f"pytorch:{model_name}:epoch-{state.get('epoch', 'unknown')}"

    def predict(self, image_bgr: np.ndarray) -> Prediction:
        processed = preprocess_fundus(image_bgr, self.preprocessing)
        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = self.torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
        mean = self.torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = self.torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        with self.torch.inference_mode():
            probabilities = self.torch.softmax(self.model((tensor - mean) / std), dim=1)[0].numpy()
        normalized = _normalize_probabilities(probabilities)
        return Prediction(
            grade=int(np.argmax(normalized)),
            probabilities=normalized,
            model_version=self.model_version,
            preprocessing=self.preprocessing,
            preprocessed_bgr=processed,
        )


def load_predictor(path: str | Path) -> GradingPredictor:
    model_path = Path(path)
    if not model_path.is_file() or model_path.stat().st_size == 0:
        raise FileNotFoundError(f"DR grading model does not exist or is empty: {model_path}")
    if model_path.suffix.lower() in {".keras", ".h5"}:
        return KerasPredictor(model_path)
    if model_path.suffix.lower() in {".pth", ".pt"}:
        return TorchPredictor(model_path)
    raise ValueError("Supported grading checkpoints are .keras, .h5, .pth, and .pt")
