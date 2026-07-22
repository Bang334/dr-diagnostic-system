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
    x = layers.Dense(512, activation='relu', name="head_dense1")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu', name="head_dense2")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32', name="predictions")(x)
    return models.Model(inputs, outputs, name="DR_EfficientNetB3_Grading")

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def load_keras_grade_model(
    model_path: str,
    input_shape: Tuple[int, int, int] = (300, 300, 3),
    num_classes: int = 5,
) -> tf.keras.Model:
    """
    Build EfficientNetB3 architecture and load weights from .keras or .h5 checkpoint.
    Robust loader trying direct load_model first, followed by custom functional and sequential fallbacks.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    print(f"[*] Loading Keras Grade Model from: {model_path}")

    # Strategy 1: Direct Keras model load (reads model config & weights stored inside .keras zip)
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print("[v] Keras Grade Model loaded successfully via tf.keras.models.load_model!")
        return model
    except Exception as e_direct:
        print(f"[!] Direct load_model failed ({e_direct}). Attempting custom architecture reconstruction...")

    # Strategy 2: Single GlobalAveragePooling2D (1536 channels)
    try:
        model = build_efficientnet_b3(
            input_shape=input_shape,
            num_classes=num_classes,
            weights=None,
            training_base=True,
            include_augmentation=True,
            use_dual_pooling=False,
        )
        model.load_weights(model_path)
        print("[v] Keras Grade Model loaded successfully (Single Pooling - 1536 channels)!")
        return model
    except Exception as e1:
        print(f"[!] Single pooling load failed. Trying Dual Pooling (3072 channels)...")

    # Strategy 3: Dual Pooling (3072 channels)
    try:
        model = build_efficientnet_b3(
            input_shape=input_shape,
            num_classes=num_classes,
            weights=None,
            training_base=True,
            include_augmentation=True,
            use_dual_pooling=True,
        )
        model.load_weights(model_path)
        print("[v] Keras Grade Model loaded successfully (Dual Pooling - 3072 channels)!")
        return model
    except Exception as e2:
        print(f"[!] Dual pooling load failed. Trying Simple Sequential fallback...")

    # Strategy 4: Simple Sequential (Base -> GAP -> Dropout -> Dense(5))
    try:
        from tensorflow.keras import layers, models
        from tensorflow.keras.applications import EfficientNetB3

        base_model = EfficientNetB3(weights=None, include_top=False, input_shape=input_shape)
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax"),
        ])
        model.load_weights(model_path)
        print("[v] Keras Grade Model loaded successfully (Simple Sequential)!")
        return model
    except Exception as e3:
        raise ValueError(
            f"Could not load model checkpoint from {model_path}. "
            f"Errors: direct_load={e_direct}, single_pool={e1}, dual_pool={e2}, sequential={e3}"
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
    print(f"[v] Accepted {len(frame):,} pseudo-labels across classes:")
    print(frame["pseudo_label"].value_counts().sort_index().to_dict())
    return frame


def create_semi_tf_dataset(
    labeled_df: pd.DataFrame,
    pseudo_df: pd.DataFrame,
    batch_size: int = 16,
    input_size: Tuple[int, int] = (300, 300),
    pseudo_weight: float = 0.25,
    is_training: bool = True,
) -> tf.data.Dataset:
    """
    Create a combined tf.data.Dataset with labeled data (weight 1.0) and pseudo-labeled data (weight pseudo_weight * confidence).
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
        one_hot_label = tf.one_hot(label, depth=5)
        return img, one_hot_label, weight

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
    )

    # 5. Compile Model
    optimizer = optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
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
