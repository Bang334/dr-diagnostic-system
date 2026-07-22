"""Keras/TensorFlow semi-supervised training pipeline for EfficientNetB3 DR Grade classification.

This module allows loading a pre-trained EfficientNetB3 Keras model (e.g. best_EfficientNetB3_rgb_crop_v1.keras)
from local path or Google Drive, generating pseudo-labels for unlabeled fundus images,
and fine-tuning with a combined labeled + pseudo-labeled dataset.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks, optimizers

CLASS_NAMES = [
    "No DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "Proliferative DR"
]


def build_efficientnet_b3(
    input_shape: Tuple[int, int, int] = (300, 300, 3),
    num_classes: int = 5,
    weights: Optional[str] = None,
    training_base: bool = False,
    include_augmentation: bool = True,
    use_dual_pooling: bool = False,
) -> tf.keras.Model:
    """
    Xây dựng kiến trúc EfficientNetB3 chuẩn cho classification 5 lớp DR grade.
    Khớp cấu trúc lớp với checkpoint best_EfficientNetB3_rgb_crop_v1.keras.
    """
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import EfficientNetB3
    from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess

    inputs = layers.Input(shape=input_shape)
    x = inputs

    if include_augmentation:
        data_augmentation = models.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom((-0.1, 0.1)),
        ], name="data_augmentation")
        x = data_augmentation(x)

    x = layers.Lambda(effnet_preprocess, name="preprocess_input")(x)

    base_model = EfficientNetB3(weights=weights, include_top=False, input_shape=input_shape)
    x = base_model(x, training=training_base)

    if use_dual_pooling:
        avg_pool = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        max_pool = layers.GlobalMaxPooling2D(name="max_pool")(x)
        x = layers.Concatenate(name="dual_pool")([avg_pool, max_pool])
    else:
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu', name="head_dense1")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu', name="head_dense2")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32', name="predictions")(x)
    return models.Model(inputs, outputs, name="DR_EfficientNetB3_Grading")

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@tf.keras.utils.register_keras_serializable(package="Custom", name="CutOutLayer")
class CutOutLayer(tf.keras.layers.Layer):
    """Custom CutOut augmentation layer present in teacher's model checkpoint."""
    def __init__(self, mask_size_ratio: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.mask_size_ratio = mask_size_ratio

    def call(self, inputs, training=None):
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({"mask_size_ratio": self.mask_size_ratio})
        return config


@tf.keras.utils.register_keras_serializable(package="Custom", name="PreprocessInputLayer")
class PreprocessInputLayer(tf.keras.layers.Layer):
    """Custom PreprocessInputLayer present in teacher's model checkpoint."""
    def __init__(self, model_name: str = "EfficientNetB3", **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name

    def call(self, inputs, training=None):
        from tensorflow.keras.applications.efficientnet import preprocess_input
        return preprocess_input(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"model_name": self.model_name})
        return config


@tf.keras.utils.register_keras_serializable(package="Custom", name="GeMPoolingLayer")
class GeMPoolingLayer(tf.keras.layers.Layer):
    """Custom Generalized Mean Pooling (GeM) layer present in teacher's model checkpoint."""
    def __init__(self, p: float = 3.0, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.p_init = float(p)
        self.eps = float(eps)
        self.p = self.add_weight(
            name="p",
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.p_init),
            trainable=True,
        )

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=None):
        x = tf.clip_by_value(inputs, self.eps, tf.float32.max)
        x = tf.pow(x, self.p)
        x = tf.reduce_mean(x, axis=[1, 2], keepdims=False)
        x = tf.pow(x, 1.0 / self.p)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"p": self.p_init, "eps": self.eps})
        return config


@tf.keras.utils.register_keras_serializable(package="Custom", name="CohenKappaMetric")
class CohenKappaMetric(tf.keras.metrics.Metric):
    """Custom Cohen Kappa Metric present in teacher's model checkpoint."""
    def __init__(self, name="kappa", num_classes=5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes

    def update_state(self, y_true, y_pred, sample_weight=None):
        pass

    def result(self):
        return tf.constant(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config


@tf.keras.utils.register_keras_serializable(package="Custom", name="ordinal_loss")
def ordinal_loss(y_true, y_pred):
    """Custom ordinal loss function present in teacher's model checkpoint."""
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)


def build_coral_efficientnet_b3(input_shape: Tuple[int, int, int] = (300, 300, 3)) -> tf.keras.Model:
    """
    Manually build CORAL EfficientNetB3 model architecture for 5 DR grades (4 output sigmoid neurons).
    Matches teacher's layer names: GeMPoolingLayer, BatchNormalization, Dense(256), Dense(256), Dense(4).
    """
    from tensorflow.keras.applications import EfficientNetB3
    from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess

    inputs = tf.keras.Input(shape=input_shape, name="input_layer")
    x = tf.keras.layers.Lambda(effnet_preprocess, name="preprocess_input")(inputs)

    base = EfficientNetB3(weights=None, include_top=False, input_tensor=x)
    
    x = GeMPoolingLayer(p=3.0, name="gem_pooling")(base.output)
    x = tf.keras.layers.BatchNormalization(name="batch_normalization")(x)
    x = tf.keras.layers.Dense(256, name="head_dense1")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, name="head_dense2")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(4, activation="sigmoid", name="ordinal_output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="CORAL_EfficientNetB3")
    return model


def load_keras_grade_model(
    model_path: str,
    input_shape: Tuple[int, int, int] = (300, 300, 3),
) -> tf.keras.Model:
    """
    Load teacher's CORAL EfficientNetB3 grade model from .keras checkpoint.
    """
    import zipfile

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    print(f"[*] Loading Keras Grade Model from: {model_path}")

    custom_objs = {
        "CutOutLayer": CutOutLayer,
        "PreprocessInputLayer": PreprocessInputLayer,
        "GeMPoolingLayer": GeMPoolingLayer,
        "ordinal_loss": ordinal_loss,
        "CohenKappaMetric": CohenKappaMetric,
    }

    dummy = tf.zeros((1,) + tuple(input_shape), dtype=tf.float32)

    # Strategy 1: Direct tf.keras.models.load_model with registered custom objects & loss
    e1 = None
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objs, compile=False)
        print("[v] Successfully loaded model via tf.keras.models.load_model!")
        return model
    except Exception as err:
        e1 = err
        print(f"[!] Direct load_model failed ({type(e1).__name__}: {e1}). Trying config-from-zip fallback...")

    # Strategy 2: Read config.json from zip -> model_from_json -> dummy forward -> load_weights
    e2 = None
    try:
        with zipfile.ZipFile(model_path, "r") as zf:
            config_json = zf.read("config.json").decode("utf-8")

        with tf.keras.utils.custom_object_scope(custom_objs):
            full_model = tf.keras.models.model_from_json(config_json)

        # Detect expected input shape dynamically from reconstructed model
        in_shape = full_model.input_shape
        if isinstance(in_shape, list):
            in_shape = in_shape[0]
        h = in_shape[1] if (len(in_shape) > 1 and in_shape[1] is not None) else input_shape[0]
        w = in_shape[2] if (len(in_shape) > 2 and in_shape[2] is not None) else input_shape[1]

        dummy_zip = tf.zeros((1, h, w, 3), dtype=tf.float32)
        _ = full_model(dummy_zip, training=False)
        full_model.load_weights(model_path)
        print(f"[v] Loaded model via config-from-zip fallback! (Input resolution: {h}x{w})")
        return full_model
    except Exception as err:
        e2 = err

    # Strategy 3: Local architecture build + load_weights
    e3 = None
    try:
        local_model = build_coral_efficientnet_b3(input_shape=input_shape)
        _ = local_model(dummy, training=False)
        local_model.load_weights(model_path)
        print("[v] Loaded weights into local CORAL model architecture!")
        return local_model
    except Exception as err:
        e3 = err

    raise ValueError(
        f"Could not load model from {model_path}.\n"
        f"  - load_model          : {e1}\n"
        f"  - config_zip_fallback : {e2}\n"
        f"  - local_build         : {e3}"
    )



def discover_unlabeled_images(unlabeled_dir: Path) -> List[Path]:
    """Scan directory recursively for supported fundus image files."""
    unlabeled_dir = Path(unlabeled_dir).expanduser().resolve()
    if not unlabeled_dir.is_dir():
        raise FileNotFoundError(f"Unlabeled directory does not exist: {unlabeled_dir}")

    images = [
        path for path in unlabeled_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images, key=os.fspath)


def generate_pseudo_labels_keras(
    model: tf.keras.Model,
    image_paths: List[Path],
    threshold: float = 0.95,
    batch_size: int = 32,
    input_size: Tuple[int, int] = (300, 300),
    max_per_class: int = 0,
) -> pd.DataFrame:
    """
    Run batch inference on unlabeled images and collect pseudo-labels passing threshold.
    Supports both CORAL Ordinal Regression (4 output sigmoid neurons, 5 DR classes) and standard Softmax.
    """
    print(f"[*] Generating pseudo-labels for {len(image_paths):,} images (threshold={threshold})...")
    records = []

    def load_and_preprocess_img(path_str):
        img_raw = tf.io.read_file(path_str)
        img = tf.image.decode_image(img_raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, input_size)
        img = tf.cast(img, tf.float32)
        return img, path_str

    path_strings = [os.fspath(p.resolve()) for p in image_paths]
    path_dataset = tf.data.Dataset.from_tensor_slices(path_strings)
    dataset = (
        path_dataset
        .map(load_and_preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    total_batches = math.ceil(len(image_paths) / batch_size)
    processed_count = 0

    for batch_imgs, batch_paths in dataset:
        preds = model.predict(batch_imgs, verbose=0)
        
        # Check output type: CORAL Ordinal Regression (4 outputs) vs Standard Softmax
        if preds.shape[-1] == 4:
            # CORAL Ordinal Regression: 4 sigmoid outputs [DR>0, DR>1, DR>2, DR>3]
            # Grade class k in {0, 1, 2, 3, 4} = count of sigmoid probabilities >= 0.5
            labels = np.sum(preds >= 0.5, axis=1)

            confidences = []
            for p, k in zip(preds, labels):
                # Expected target for grade k is 1 for i < k else 0
                target = np.array([1.0 if i < k else 0.0 for i in range(4)])
                certainty = float(np.mean(np.where(target == 1.0, p, 1.0 - p)))
                confidences.append(certainty)
            confidences = np.array(confidences)
        else:
            confidences = np.max(preds, axis=1)
            labels = np.argmax(preds, axis=1)

        for p_str, label, conf in zip(batch_paths.numpy(), labels, confidences):
            path_decoded = p_str.decode("utf-8") if isinstance(p_str, bytes) else str(p_str)
            if float(conf) >= threshold:
                records.append({
                    "image_path": path_decoded,
                    "pseudo_label": int(label),
                    "confidence": float(conf),
                })
            processed_count += 1

        if processed_count % (batch_size * 10) == 0 or processed_count == len(image_paths):
            print(f"  Processed {processed_count}/{len(image_paths)} images, accepted={len(records)}")

    frame = pd.DataFrame.from_records(records, columns=["image_path", "pseudo_label", "confidence"])
    if frame.empty:
        print("[!] Warning: No unlabeled images met the confidence threshold!")
        return frame

    frame = frame.sort_values("confidence", ascending=False)
    if max_per_class > 0:
        frame = frame.groupby("pseudo_label", group_keys=False).head(max_per_class)

    frame = frame.sort_values(["pseudo_label", "confidence"], ascending=[True, False]).reset_index(drop=True)
    print(f"[v] Accepted {len(frame):,} pseudo-labels across classes (0-4):")
    print(frame["pseudo_label"].value_counts().sort_index().to_dict())
    return frame


def create_semi_tf_dataset(
    labeled_df: pd.DataFrame,
    pseudo_df: pd.DataFrame,
    batch_size: int = 16,
    input_size: Tuple[int, int] = (300, 300),
    pseudo_weight: float = 0.25,
    is_training: bool = True,
    is_coral: bool = True,
    num_classes: int = 5,
) -> Tuple[tf.data.Dataset, int]:
    """
    Create a combined tf.data.Dataset with labeled data (weight 1.0) and pseudo-labeled data (weight pseudo_weight * confidence).
    Supports CORAL Ordinal Regression (4 binary outputs for 5 DR grades).
    """
    paths = []
    labels = []
    sample_weights = []

    # Process labeled data
    for _, row in labeled_df.iterrows():
        paths.append(str(row["image_path"]))
        labels.append(int(row["diagnosis"]))
        sample_weights.append(1.0)

    # Process pseudo-labeled data
    if pseudo_df is not None and not pseudo_df.empty:
        for _, row in pseudo_df.iterrows():
            paths.append(str(row["image_path"]))
            labels.append(int(row["pseudo_label"]))
            weight = float(pseudo_weight * row.get("confidence", 1.0))
            sample_weights.append(weight)

    paths_tensor = tf.constant(paths)
    labels_tensor = tf.constant(labels, dtype=tf.int32)
    weights_tensor = tf.constant(sample_weights, dtype=tf.float32)

    def parse_sample(path_str, label, weight):
        img_raw = tf.io.read_file(path_str)
        img = tf.image.decode_image(img_raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, input_size)
        img = tf.cast(img, tf.float32)
        if is_coral:
            # CORAL label target: binary vector of length 4 for DR > 0, DR > 1, DR > 2, DR > 3
            # Grade 0 -> [0,0,0,0], Grade 1 -> [1,0,0,0], Grade 2 -> [1,1,0,0], Grade 3 -> [1,1,1,0], Grade 4 -> [1,1,1,1]
            target_label = tf.cast(tf.range(4) < label, tf.float32)
        else:
            target_label = tf.one_hot(label, depth=num_classes)
        return img, target_label, weight

    ds = tf.data.Dataset.from_tensor_slices((paths_tensor, labels_tensor, weights_tensor))
    ds = ds.map(parse_sample, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        ds = ds.shuffle(buffer_size=min(len(paths), 2000), seed=42)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, len(paths)


def train_keras_semi_supervised(
    model_path: str,
    labeled_csv: str,
    unlabeled_dir: str,
    output_dir: str,
    threshold: float = 0.95,
    pseudo_weight: float = 0.25,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-4,
    input_size: Tuple[int, int] = (300, 300),
):
    """
    Main function to run Keras semi-supervised pipeline.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Model
    model = load_keras_grade_model(model_path, input_shape=(*input_size, 3))
    m_shape = model.input_shape
    if isinstance(m_shape, list):
        m_shape = m_shape[0]
    if len(m_shape) >= 3 and m_shape[1] is not None and m_shape[2] is not None:
        input_size = (int(m_shape[1]), int(m_shape[2]))
        print(f"[*] Auto-adjusted input_size to model's native resolution: {input_size}")

    num_outputs = int(model.output_shape[-1])
    is_coral = (num_outputs == 4)
    print(f"[*] Detected model output neurons: {num_outputs} ({'CORAL Ordinal Regression (5 DR grades)' if is_coral else 'Standard Softmax'})")

    # 2. Discover Unlabeled Images & Generate Pseudo-labels
    unlabeled_images = discover_unlabeled_images(Path(unlabeled_dir))
    pseudo_df = generate_pseudo_labels_keras(
        model,
        unlabeled_images,
        threshold=threshold,
        batch_size=batch_size,
        input_size=input_size,
    )
    pseudo_csv_path = output_path / "pseudo_labels.csv"
    pseudo_df.to_csv(pseudo_csv_path, index=False)
    print(f"[v] Saved pseudo-labels to {pseudo_csv_path}")

    # 3. Load Labeled Data
    labeled_df = pd.read_csv(labeled_csv)

    # 4. Create Combined Dataset
    train_ds, total_samples = create_semi_tf_dataset(
        labeled_df=labeled_df,
        pseudo_df=pseudo_df,
        batch_size=batch_size,
        input_size=input_size,
        pseudo_weight=pseudo_weight,
        is_training=True,
        is_coral=is_coral,
        num_classes=5,
    )

    # 5. Compile Model
    optimizer = optimizers.Adam(learning_rate=lr)
    if is_coral:
        loss_fn = "binary_crossentropy"
        metrics_list = ["binary_accuracy", "mae"]
    else:
        loss_fn = "categorical_crossentropy"
        metrics_list = ["accuracy"]

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=metrics_list,
    )

    # 6. Callbacks
    best_model_path = output_path / "best_semi_EfficientNetB3.keras"
    cb_list = [
        callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            save_best_only=True,
            monitor="loss",
            mode="min",
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    # 7. Fit Model
    print(f"[*] Starting semi-supervised fine-tuning for {epochs} epochs on {total_samples} samples...")
    history = model.fit(
        train_ds,
        epochs=epochs,
        callbacks=cb_list,
    )

    # 8. Save final status
    history_dict = {k: [float(val) for val in v] for k, v in history.history.items()}
    with open(output_path / "history.json", "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=2)

    print(f"[v] Training complete! Best checkpoint saved to: {best_model_path}")
    return best_model_path


def parse_args():
    parser = argparse.ArgumentParser(description="Keras Semi-Supervised Training for EfficientNetB3 DR Grade Model")
    parser.add_argument("--model-path", required=True, type=str, help="Path to base .keras checkpoint")
    parser.add_argument("--labeled-csv", required=True, type=str, help="CSV file containing labeled image_path & diagnosis")
    parser.add_argument("--unlabeled-dir", required=True, type=str, help="Directory containing unlabeled images")
    parser.add_argument("--output-dir", required=True, type=str, help="Output directory for checkpoints and logs")
    parser.add_argument("--threshold", type=float, default=0.95, help="Pseudo-label confidence threshold")
    parser.add_argument("--pseudo-weight", type=float, default=0.25, help="Loss weight for pseudo-labeled samples")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_keras_semi_supervised(
        model_path=args.model_path,
        labeled_csv=args.labeled_csv,
        unlabeled_dir=args.unlabeled_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
        pseudo_weight=args.pseudo_weight,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
