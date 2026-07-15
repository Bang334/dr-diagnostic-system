"""Inference seam for diabetic-retinopathy grading models."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import cv2
import numpy as np

from ai.grading.artifacts import (
    CLASS_NAMES,
    PreprocessingSpec,
    checkpoint_metadata,
    load_torch_checkpoint,
)
from ai.preprocessing.fundus_prep import preprocess_fundus_array


@dataclass(frozen=True)
class ModelInfo:
    backend: str
    architecture: str
    model_version: str
    class_names: tuple[str, ...]
    preprocessing: PreprocessingSpec
    artifact_schema_version: int
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["class_names"] = list(self.class_names)
        return result


@dataclass
class DRPrediction:
    grade: int
    label: str
    confidence: float
    probabilities: tuple[float, ...]
    model_version: str
    class_names: tuple[str, ...]
    preprocessed_bgr: np.ndarray = field(repr=False)

    def to_api_dict(self) -> dict[str, Any]:
        return {
            "dr_grade": self.grade,
            "dr_label": self.label,
            "confidence": round(self.confidence, 4),
            "probabilities": {
                name: round(float(probability), 4)
                for name, probability in zip(self.class_names, self.probabilities)
            },
            "model_version": self.model_version,
        }


@runtime_checkable
class DRPredictor(Protocol):
    @property
    def info(self) -> ModelInfo:
        ...

    def predict(self, image_bgr: np.ndarray) -> DRPrediction:
        ...


def _prepare_fundus(
    image_bgr: np.ndarray, spec: PreprocessingSpec
) -> tuple[np.ndarray, np.ndarray]:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Fundus image is empty")
    cropped = preprocess_fundus_array(
        image_bgr,
        img_size=None,
        enhance=spec.enhance,
        crop_tolerance=spec.crop_tolerance,
    )
    preview = cv2.resize(
        cropped,
        (spec.image_size, spec.image_size),
        interpolation=cv2.INTER_AREA,
    )
    return cropped, preview


def _prediction_from_probabilities(
    probabilities: np.ndarray,
    *,
    class_names: tuple[str, ...],
    model_version: str,
    preview: np.ndarray,
) -> DRPrediction:
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(probabilities) != len(class_names):
        raise ValueError(
            f"Model returned {len(probabilities)} probabilities for {len(class_names)} classes"
        )
    if not np.isfinite(probabilities).all() or probabilities.sum() <= 0:
        raise ValueError("Model returned invalid probabilities")
    probabilities = probabilities / probabilities.sum()
    grade = int(np.argmax(probabilities))
    return DRPrediction(
        grade=grade,
        label=class_names[grade],
        confidence=float(probabilities[grade]),
        probabilities=tuple(float(value) for value in probabilities),
        model_version=model_version,
        class_names=class_names,
        preprocessed_bgr=preview,
    )


class PyTorchRETFoundPredictor:
    def __init__(self, model_path: str | os.PathLike[str], *, device: str | None = None):
        import timm
        import torch
        from PIL import Image
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode

        self._torch = torch
        self._image_type = Image
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"DR model does not exist: {self.model_path}")

        checkpoint = load_torch_checkpoint(self.model_path)
        metadata = checkpoint_metadata(checkpoint)
        if metadata.model.framework != "pytorch":
            raise ValueError("Checkpoint is not a PyTorch grading model")
        if metadata.model.output_type != "logits" or metadata.model.num_classes != len(
            metadata.class_names
        ):
            raise ValueError("Only five-class cross-entropy checkpoints are deployable")

        model_kwargs: dict[str, Any] = {
            "pretrained": False,
            "num_classes": metadata.model.num_classes,
        }
        if metadata.model.model_source == "retfound":
            model_kwargs["img_size"] = metadata.preprocessing.image_size
        self.model = timm.create_model(metadata.model.architecture, **model_kwargs)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.model.eval()
        self.metadata = metadata
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (metadata.preprocessing.image_size, metadata.preprocessing.image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    metadata.preprocessing.mean, metadata.preprocessing.std
                ),
            ]
        )
        self._info = ModelInfo(
            backend="pytorch",
            architecture=metadata.model.architecture,
            model_version=f"{metadata.model.architecture}:{self.model_path.stem}",
            class_names=metadata.class_names,
            preprocessing=metadata.preprocessing,
            artifact_schema_version=metadata.schema_version,
            provenance=metadata.provenance,
        )

    @property
    def info(self) -> ModelInfo:
        return self._info

    def predict(self, image_bgr: np.ndarray) -> DRPrediction:
        cropped, preview = _prepare_fundus(image_bgr, self.metadata.preprocessing)
        image_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        tensor = self.transform(self._image_type.fromarray(image_rgb)).unsqueeze(0)
        tensor = tensor.to(self.device)
        with self._torch.inference_mode():
            logits = self.model(tensor)
            probabilities = self._torch.softmax(logits, dim=1)[0].cpu().numpy()
        return _prediction_from_probabilities(
            probabilities,
            class_names=self.metadata.class_names,
            model_version=self.info.model_version,
            preview=preview,
        )


class KerasEfficientNetPredictor:
    """Backward-compatible adapter for the legacy EfficientNet-B3 weights."""

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        *,
        enhance: bool = False,
    ):
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file() or self.model_path.stat().st_size == 0:
            raise FileNotFoundError(f"DR model does not exist or is empty: {self.model_path}")
        self.preprocessing = PreprocessingSpec(image_size=300, enhance=enhance)
        self.model = self._build_model()
        self.model.load_weights(os.fspath(self.model_path))
        self._info = ModelInfo(
            backend="keras",
            architecture="efficientnet_b3",
            model_version="efficientnet_b3_v1.0",
            class_names=CLASS_NAMES,
            preprocessing=self.preprocessing,
            artifact_schema_version=0,
            provenance={"legacy_checkpoint": True},
        )

    @staticmethod
    def _build_model():
        from tensorflow.keras import layers, models
        from tensorflow.keras.applications import EfficientNetB3
        from tensorflow.keras.applications.efficientnet import preprocess_input

        data_augmentation = models.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.05),
                layers.RandomZoom((-0.1, 0.1)),
            ],
            name="data_augmentation",
        )
        inputs = layers.Input(shape=(300, 300, 3))
        x = data_augmentation(inputs)
        x = layers.Lambda(preprocess_input, name="preprocess_input")(x)
        base_model = EfficientNetB3(
            weights=None, include_top=False, input_shape=(300, 300, 3)
        )
        x = base_model(x, training=False)
        average = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        maximum = layers.GlobalMaxPooling2D(name="max_pool")(x)
        x = layers.Concatenate(name="dual_pool")([average, maximum])
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512, activation="relu", name="head_dense1")(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation="relu", name="head_dense2")(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(
            5, activation="softmax", dtype="float32", name="predictions"
        )(x)
        return models.Model(inputs, outputs)

    @property
    def info(self) -> ModelInfo:
        return self._info

    def predict(self, image_bgr: np.ndarray) -> DRPrediction:
        _, preview = _prepare_fundus(image_bgr, self.preprocessing)
        image_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB).astype("float32")
        probabilities = self.model.predict(np.expand_dims(image_rgb, axis=0), verbose=0)[0]
        return _prediction_from_probabilities(
            probabilities,
            class_names=self.info.class_names,
            model_version=self.info.model_version,
            preview=preview,
        )


def load_predictor(
    model_path: str | os.PathLike[str],
    *,
    backend: str = "auto",
    device: str | None = None,
    legacy_enhance: bool = False,
) -> DRPredictor:
    path = Path(model_path)
    selected = backend.lower()
    if selected == "auto":
        selected = "pytorch" if path.suffix.lower() in {".pth", ".pt"} else "keras"
    if selected == "pytorch":
        return PyTorchRETFoundPredictor(path, device=device)
    if selected == "keras":
        return KerasEfficientNetPredictor(path, enhance=legacy_enhance)
    raise ValueError(f"Unsupported DR model backend: {backend}")
