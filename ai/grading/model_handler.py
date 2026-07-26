import os
import json
import zipfile

import cv2
import numpy as np
import tensorflow as tf

from ai.grading.taxonomy import CLASS_NAMES

# Mapping classes dựa trên thang chuẩn ICDR
class DRModelHandler:
    def __init__(self, model_path: str, threshold_path: str | None = None):
        self.model_path = model_path
        self.threshold_path = threshold_path
        self.model = None
        self.input_size = (384, 384)
        self.thresholds = DEFAULT_ORDINAL_THRESHOLDS.copy()
        self.model_version = "efficientnet_b3_rgb_crop_v1"
        self._load_model()

    def _read_saved_model_config(self):
        defaults = {
            "input_shape": (384, 384, 3),
            "model_name": "EfficientNetB3",
            "head_dense1_units": 256,
            "head_dense2_units": 256,
            "head_drop1_rate": 0.4,
            "head_drop2_rate": 0.25,
        }
        try:
            with zipfile.ZipFile(self.model_path) as archive:
                config = json.loads(archive.read("config.json"))
            for layer in config["config"]["layers"]:
                layer_config = layer.get("config", {})
                name = layer_config.get("name")
                if layer["class_name"] == "InputLayer" and layer_config.get("batch_shape"):
                    defaults["input_shape"] = tuple(layer_config["batch_shape"][1:])
                elif layer["class_name"] == "Functional":
                    backbone_name = str(layer_config.get("name", "")).lower()
                    if backbone_name == "efficientnetb4":
                        defaults["model_name"] = "EfficientNetB4"
                    elif backbone_name == "efficientnetb3":
                        defaults["model_name"] = "EfficientNetB3"
                elif name == "head_dense1":
                    defaults["head_dense1_units"] = int(layer_config["units"])
                elif name == "head_dense2":
                    defaults["head_dense2_units"] = int(layer_config["units"])
                elif name == "head_drop1":
                    defaults["head_drop1_rate"] = float(layer_config["rate"])
                elif name == "head_drop2":
                    defaults["head_drop2_rate"] = float(layer_config["rate"])
        except Exception as exc:
            print(f"[!] Could not read model config; using fallback defaults: {exc}")
        return defaults

    def _build_compatible_model(self):
        """Rebuild the training graph for Keras-version-compatible weight loading."""
        saved_config = self._read_saved_model_config()
        input_shape = saved_config["input_shape"]
        model_name = saved_config["model_name"]

        augmentation = models.Sequential(
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

        inputs = layers.Input(shape=input_shape)
        features = augmentation(inputs)
        features = PreprocessInputLayer(
            model_name=model_name, name="preprocess_input"
        )(features)
        backbone_class = {
            "EfficientNetB3": EfficientNetB3,
            "EfficientNetB4": EfficientNetB4,
        }[model_name]
        backbone = backbone_class(
            weights=None,
            include_top=False,
            input_shape=input_shape,
        )
        features = backbone(features, training=False)
        features = GeMPoolingLayer(p=3.0, name="gem_pooling")(features)
        features = layers.BatchNormalization(name="head_bn")(features)
        features = layers.Dense(
            saved_config["head_dense1_units"], name="head_dense1"
        )(features)
        features = layers.Activation("gelu", name="head_gelu1")(features)
        features = layers.Dropout(saved_config["head_drop1_rate"], name="head_drop1")(
            features
        )
        features = layers.Dense(
            saved_config["head_dense2_units"], name="head_dense2"
        )(features)
        features = layers.Activation("gelu", name="head_gelu2")(features)
        features = layers.Dropout(saved_config["head_drop2_rate"], name="head_drop2")(
            features
        )
        outputs = layers.Dense(
            4,
            activation="sigmoid",
            dtype="float32",
            name="ordinal_output",
        )(features)
        return models.Model(inputs, outputs)

    def _load_model(self):
        if not os.path.isfile(self.model_path) or os.path.getsize(self.model_path) == 0:
            print(f"[!] Model file not found: {self.model_path}")
            return

        try:
            try:
                self.model = tf.keras.models.load_model(
                    self.model_path,
                    compile=False,
                    safe_mode=False,
                    custom_objects={
                        "CutOutLayer": CutOutLayer,
                        "PreprocessInputLayer": PreprocessInputLayer,
                        "GeMPoolingLayer": GeMPoolingLayer,
                        "preprocess_input": effnet_preprocess,
                    },
                )
            except Exception as load_error:
                print(f"[!] Full-model load failed ({load_error}); loading weights instead")
                self.model = self._build_compatible_model()
                self.model.load_weights(self.model_path)
            height, width = self.model.input_shape[1:3]
            self.input_size = (int(width), int(height))
            backbone_names = {
                layer.name.lower()
                for layer in self.model.layers
                if layer.name.lower() in {"efficientnetb3", "efficientnetb4"}
            }
            if "efficientnetb4" in backbone_names:
                self.model_version = "efficientnet_b4_rgb_crop_v1_continued"
            elif "efficientnetb3" in backbone_names:
                self.model_version = "efficientnet_b3_rgb_crop_v1"

            if self.threshold_path and os.path.isfile(self.threshold_path):
                thresholds = np.load(self.threshold_path).astype(np.float32).reshape(-1)
                if thresholds.shape != (4,):
                    raise ValueError(f"Expected 4 ordinal thresholds, received {thresholds.shape}")
                if not np.all(np.isfinite(thresholds)) or np.any(
                    (thresholds < 0.0) | (thresholds > 1.0)
                ):
                    raise ValueError(
                        f"Ordinal thresholds must be finite values in [0, 1]: {thresholds}"
                    )
                self.thresholds = thresholds
            else:
                print(
                    "[!] Threshold file not found; using calibrated defaults "
                    f"{self.thresholds.tolist()}"
                )

            print(
                f"[v] Loaded {self.model_path} | input={self.input_size} "
                f"| thresholds={self.thresholds.tolist()}"
            )
        except Exception as exc:
            self.model = None
            print(f"[x] Could not load grading model: {exc}")

    def predict(self, preprocessed_img: np.ndarray):
        """Predict one RGB image produced by preprocess_rgb_crop_512_from_bgr."""
        if self.model is None:
            raise ValueError("Grading model is not loaded")

        image = cv2.resize(preprocessed_img, self.input_size, interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32)
        original = np.expand_dims(image, axis=0)
        flipped = np.flip(original, axis=2).copy()

        original_probs = self.model.predict(original, verbose=0)
        flipped_probs = self.model.predict(flipped, verbose=0)
        ordinal_probs = ((original_probs + flipped_probs) / 2.0)[0]
        ordinal_probs = np.minimum.accumulate(ordinal_probs)

        predicted_class = int(np.sum(ordinal_probs > self.thresholds))
        class_probs = np.diff(
            -np.concatenate(([1.0], ordinal_probs, [0.0]))
        )
        class_probs = np.clip(class_probs, 0.0, 1.0)
        total = float(class_probs.sum())
        if total > 0:
            class_probs = class_probs / total

        return {
            "dr_grade": predicted_class,
            "dr_label": CLASS_NAMES[predicted_class],
            "confidence": round(float(class_probs[predicted_class]), 4),
            "probabilities": {
                name: round(float(probability), 4)
                for name, probability in zip(CLASS_NAMES, class_probs)
            },
            "ordinal_probabilities": [round(float(value), 4) for value in ordinal_probs],
            "ordinal_thresholds": [round(float(value), 4) for value in self.thresholds],
            "model_version": self.model_version,
        }
