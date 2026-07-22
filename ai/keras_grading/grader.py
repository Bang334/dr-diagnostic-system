"""Deep module for loading and running the Keras ordinal teacher model."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from ai.keras_grading.ordinal import (
    DEFAULT_ORDINAL_THRESHOLDS,
    OrdinalPrediction,
    decode_ordinal,
    validate_thresholds,
)
from ai.preprocessing.fundus_prep import preprocess_rgb_crop_512_from_bgr


class KerasOrdinalGrader:
    """Load once and predict calibrated grades from paths or BGR arrays."""

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        *,
        thresholds: Sequence[float] = DEFAULT_ORDINAL_THRESHOLDS,
        threshold_path: str | os.PathLike[str] | None = None,
        use_tta: bool = True,
    ):
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Keras model not found: {self.model_path}")
        if threshold_path is not None:
            thresholds = np.load(Path(threshold_path).expanduser().resolve())
        self.thresholds = validate_thresholds(thresholds)
        self.use_tta = bool(use_tta)
        self.model = self._load_model()
        shape = self.model.input_shape
        if len(shape) != 4 or shape[-1] != 3:
            raise ValueError(f"Expected an NHWC RGB model input, received {shape}")
        self.input_size = (int(shape[2]), int(shape[1]))
        if self.model.output_shape[-1] != 4:
            raise ValueError(
                f"Expected 4 ordinal outputs, received {self.model.output_shape}"
            )

    def _load_model(self):
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError(
                "TensorFlow is required to load the Keras teacher model. "
                "Install ai/semi_supervised/requirements-keras.txt."
            ) from exc

        from ai.keras_grading.custom_objects import get_custom_objects

        try:
            model = tf.keras.models.load_model(
                self.model_path,
                compile=False,
                safe_mode=False,
                custom_objects=get_custom_objects(),
            )
            self.load_mode = "full-model"
            return model
        except Exception as load_error:
            print(
                "Full-model load failed; rebuilding the known EfficientNetB3 "
                f"graph and loading weights instead: {load_error}",
                flush=True,
            )
            model = self._build_compatible_model(tf)
            model.load_weights(self.model_path)
            self.load_mode = "rebuilt-weights"
            return model

    @staticmethod
    def _build_compatible_model(tf):
        from ai.keras_grading.custom_objects import (
            CutOutLayer,
            GeMPoolingLayer,
            PreprocessInputLayer,
        )

        layers = tf.keras.layers
        augmentation = tf.keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.05),
                layers.RandomZoom((-0.15, 0.15)),
                layers.RandomContrast(0.10),
                layers.RandomTranslation(0.02, 0.02),
                CutOutLayer(mask_size_ratio=0.0),
            ],
            name="data_augmentation",
        )
        inputs = layers.Input(shape=(384, 384, 3))
        features = augmentation(inputs)
        features = PreprocessInputLayer(
            model_name="EfficientNetB3", name="preprocess_input"
        )(features)
        backbone = tf.keras.applications.EfficientNetB3(
            weights=None, include_top=False, input_shape=(384, 384, 3)
        )
        backbone.trainable = True
        for layer in backbone.layers[:269]:
            layer.trainable = False
        features = backbone(features, training=False)
        features = GeMPoolingLayer(p=3.0, name="gem_pooling")(features)
        features = layers.BatchNormalization(name="head_bn")(features)
        features = layers.Dense(256, name="head_dense1")(features)
        features = layers.Activation("gelu", name="head_gelu1")(features)
        features = layers.Dropout(0.4, name="head_drop1")(features)
        features = layers.Dense(256, name="head_dense2")(features)
        features = layers.Activation("gelu", name="head_gelu2")(features)
        features = layers.Dropout(0.25, name="head_drop2")(features)
        outputs = layers.Dense(
            4, activation="sigmoid", dtype="float32", name="ordinal_output"
        )(features)
        return tf.keras.Model(inputs, outputs)

    def prepare_bgr(self, image_bgr: np.ndarray) -> np.ndarray:
        rgb_crop = preprocess_rgb_crop_512_from_bgr(image_bgr)
        return cv2.resize(rgb_crop, self.input_size, interpolation=cv2.INTER_AREA).astype(
            np.float32
        )

    def predict_bgr(self, image_bgr: np.ndarray) -> OrdinalPrediction:
        return self.predict_prepared(self.prepare_bgr(image_bgr))

    def predict_path(self, image_path: str | os.PathLike[str]) -> OrdinalPrediction:
        path = Path(image_path)
        image = cv2.imread(os.fspath(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read fundus image: {path}")
        return self.predict_bgr(image)

    def predict_prepared(self, image_rgb: np.ndarray) -> OrdinalPrediction:
        batch = np.expand_dims(np.asarray(image_rgb, dtype=np.float32), axis=0)
        ordinal = np.asarray(self.model.predict(batch, verbose=0))[0]
        if self.use_tta:
            flipped = np.flip(batch, axis=2).copy()
            ordinal = (
                ordinal + np.asarray(self.model.predict(flipped, verbose=0))[0]
            ) / 2.0
        return decode_ordinal(ordinal, self.thresholds)

    def predict_paths(
        self, paths: Iterable[str | os.PathLike[str]]
    ) -> list[tuple[Path, OrdinalPrediction]]:
        return [(Path(path), self.predict_path(path)) for path in paths]
