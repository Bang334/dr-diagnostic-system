"""Fine-tune the Keras ordinal teacher with confidence-filtered pseudo-labels.

The fixed validation split selects checkpoints. The fixed test split is never
read by this training command; evaluate it separately after model selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd

from ai.keras_grading.grader import KerasOrdinalGrader
from ai.preprocessing.fundus_prep import preprocess_rgb_crop_512_from_bgr
from ai.semi_supervised.keras_pseudo_labels import (
    select_pseudo_labels,
)


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLIT_ALIASES = {
    "train": ("train", "training"),
    "val": ("val", "valid", "validation"),
    "test": ("test", "testing"),
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--unlabeled-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--pseudo-confidence", type=float, default=0.50)
    parser.add_argument("--pseudo-weight", type=float, default=0.25)
    parser.add_argument("--max-pseudo-per-class", type=int, default=0)
    parser.add_argument("--max-unlabeled-images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every-batches", type=int, default=25)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the same Drive run using Keras BackupAndRestore state",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("epochs, batch-size and patience must be positive")
    if not 0.0 <= args.pseudo_confidence <= 1.0:
        raise ValueError("pseudo-confidence must be in [0, 1]")
    if not 0.0 < args.pseudo_weight <= 1.0:
        raise ValueError("pseudo-weight must be in (0, 1]")
    if args.max_pseudo_per_class < 0 or args.max_unlabeled_images < 0:
        raise ValueError("pseudo-label limits cannot be negative")
    if args.log_every_batches < 1:
        raise ValueError("log-every-batches must be positive")


def discover_images(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    images = sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No supported images found under {root}")
    return images


def find_splits(dataset_dir: Path) -> dict[str, Path]:
    root = dataset_dir.expanduser().resolve()
    candidates = [root, root / "split_dataset"]
    candidates.extend(path for path in root.rglob("split_dataset") if path.is_dir())
    candidates.extend(path for path in root.iterdir() if path.is_dir())
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        children = {path.name.lower(): path for path in candidate.iterdir() if path.is_dir()}
        result = {}
        for split, aliases in SPLIT_ALIASES.items():
            match = next((children[name] for name in aliases if name in children), None)
            if match is None:
                break
            result[split] = match.resolve()
        if len(result) == 3:
            return result
    raise FileNotFoundError(f"Could not locate fixed train/val/test splits under {root}")


def scan_labeled_split(root: Path) -> pd.DataFrame:
    rows = []
    for grade in range(5):
        class_dir = root / str(grade)
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing grade directory: {class_dir}")
        rows.extend(
            {"image_path": str(path.resolve()), "diagnosis": grade}
            for path in sorted(class_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    if not rows:
        raise ValueError(f"No labeled images found under {root}")
    return pd.DataFrame(rows)


def assert_external(unlabeled: Iterable[Path], split_frames: dict[str, pd.DataFrame]) -> None:
    unlabeled = list(unlabeled)
    owners = {
        os.path.normcase(str(Path(path).resolve()))
        for frame in split_frames.values()
        for path in frame["image_path"]
    }
    overlap = [path for path in unlabeled if os.path.normcase(str(path.resolve())) in owners]
    if overlap:
        raise ValueError(f"Unlabeled data overlaps a fixed split: {overlap[0]}")

    labeled_paths = [
        Path(path).resolve()
        for frame in split_frames.values()
        for path in frame["image_path"]
    ]
    labeled_by_signature: dict[tuple[str, int], list[Path]] = {}
    for path in labeled_paths:
        labeled_by_signature.setdefault(
            (path.name.casefold(), path.stat().st_size), []
        ).append(path)

    digest_cache: dict[Path, str] = {}

    def digest(path: Path) -> str:
        if path not in digest_cache:
            hasher = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
            digest_cache[path] = hasher.hexdigest()
        return digest_cache[path]

    for path in unlabeled:
        candidates = labeled_by_signature.get(
            (path.name.casefold(), path.stat().st_size), []
        )
        if any(digest(path) == digest(candidate) for candidate in candidates):
            raise ValueError(
                "Unlabeled data contains a byte-identical copy from a fixed "
                f"split: {path}"
            )


def pseudo_cache_signature(
    args: argparse.Namespace,
    grader: KerasOrdinalGrader,
    unlabeled: Iterable[Path],
) -> dict[str, object]:
    """Describe every input that can change cached pseudo-label assignment."""
    paths = sorted((Path(path).resolve() for path in unlabeled), key=os.fspath)
    inventory = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        inventory.update(os.path.normcase(os.fspath(path)).encode("utf-8"))
        inventory.update(str(stat.st_size).encode("ascii"))
        inventory.update(str(stat.st_mtime_ns).encode("ascii"))
    model_stat = grader.model_path.stat()
    return {
        "schema_version": 1,
        "teacher_model": os.fspath(grader.model_path),
        "teacher_size": model_stat.st_size,
        "teacher_mtime_ns": model_stat.st_mtime_ns,
        "unlabeled_root": os.fspath(args.unlabeled_dir.expanduser().resolve()),
        "unlabeled_images": len(paths),
        "unlabeled_inventory_sha256": inventory.hexdigest(),
        "ordinal_thresholds": [float(value) for value in grader.thresholds],
        "pseudo_confidence": float(args.pseudo_confidence),
        "max_pseudo_per_class": int(args.max_pseudo_per_class),
        "tta": not args.no_tta,
    }


def read_matching_pseudo_cache(
    cache_path: Path,
    pseudo_path: Path,
    expected_signature: dict[str, object],
) -> pd.DataFrame | None:
    if not cache_path.is_file() or not pseudo_path.is_file():
        return None
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if cached.get("signature") != expected_signature or not cached.get("complete"):
        return None
    frame = pd.read_csv(pseudo_path)
    required = {"image_path", "diagnosis", "confidence"}
    if required.difference(frame.columns):
        return None
    return frame


def _prepare_path(path_value, image_size: int) -> np.ndarray:
    if hasattr(path_value, "numpy"):
        path_value = path_value.numpy()
    if isinstance(path_value, np.ndarray):
        path_value = path_value.item()
    if isinstance(path_value, bytes):
        path_value = path_value.decode("utf-8")
    image = cv2.imread(os.fspath(path_value), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read fundus image: {path_value}")
    rgb = preprocess_rgb_crop_512_from_bgr(image)
    return cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA).astype(
        np.float32
    )


def create_dataset(tf, frame: pd.DataFrame, image_size: int, batch_size: int, *, training: bool):
    paths = frame["image_path"].astype(str).to_numpy()
    labels = frame["diagnosis"].astype(np.int32).to_numpy()
    weights = frame.get("sample_weight", pd.Series(np.ones(len(frame)))).astype(
        np.float32
    ).to_numpy()
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels, weights))
    if training:
        dataset = dataset.shuffle(min(len(frame), 4096), reshuffle_each_iteration=True)

    def load(path, label, weight):
        image = tf.numpy_function(
            lambda value: _prepare_path(value, image_size), [path], tf.float32
        )
        image.set_shape((image_size, image_size, 3))
        label.set_shape(())
        weight.set_shape(())
        return image, label, weight

    return dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(
        batch_size
    ).prefetch(tf.data.AUTOTUNE)


def detailed_logging_callback(tf, output_dir: Path, every_batches: int):
    """Stream readable progress and persist one JSON record per epoch."""

    class DetailedTrainingLogger(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.epoch_started = 0.0
            self.log_path = output_dir / "epoch-log.json"

        @staticmethod
        def _number(value):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def on_train_begin(self, logs=None):
            print("\n" + "=" * 78, flush=True)
            print("KERAS ORDINAL SEMI-SUPERVISED TRAINING", flush=True)
            print(f"Detailed epoch log: {self.log_path}", flush=True)
            print("=" * 78, flush=True)

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_started = time.time()
            print(f"\n--- Epoch {epoch + 1} started ---", flush=True)

        def on_train_batch_end(self, batch, logs=None):
            if (batch + 1) % every_batches != 0:
                return
            logs = logs or {}
            fields = [f"batch={batch + 1}"]
            for key in ("loss", "ordinal_accuracy", "kappa", "ordinal_mae"):
                value = self._number(logs.get(key))
                if value is not None:
                    fields.append(f"{key}={value:.5f}")
            print("[train] " + " | ".join(fields), flush=True)

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            duration = time.time() - self.epoch_started
            learning_rate = self._number(self.model.optimizer.learning_rate)
            record = {
                "epoch": epoch + 1,
                "duration_seconds": round(duration, 2),
                "learning_rate": learning_rate,
                **{
                    key: self._number(value)
                    for key, value in logs.items()
                    if self._number(value) is not None
                },
            }
            print("\n" + "-" * 78, flush=True)
            print(
                f"Epoch {epoch + 1} completed in {duration:.1f}s | "
                f"lr={learning_rate:.3e}",
                flush=True,
            )
            print(
                "TRAIN | "
                f"loss={record.get('loss', float('nan')):.5f} | "
                f"accuracy={record.get('ordinal_accuracy', float('nan')):.5f} | "
                f"QWK={record.get('kappa', float('nan')):.5f} | "
                f"MAE={record.get('ordinal_mae', float('nan')):.5f}",
                flush=True,
            )
            print(
                "VAL   | "
                f"loss={record.get('val_loss', float('nan')):.5f} | "
                f"accuracy={record.get('val_ordinal_accuracy', float('nan')):.5f} | "
                f"QWK={record.get('val_kappa', float('nan')):.5f} | "
                f"MAE={record.get('val_ordinal_mae', float('nan')):.5f}",
                flush=True,
            )
            print("-" * 78, flush=True)
            records = []
            if self.log_path.is_file():
                try:
                    loaded = json.loads(self.log_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, list):
                        records = loaded
                except (OSError, json.JSONDecodeError):
                    print(
                        f"Warning: could not read {self.log_path}; rebuilding it",
                        flush=True,
                    )
            records_by_epoch = {
                int(item["epoch"]): item
                for item in records
                if isinstance(item, dict) and "epoch" in item
            }
            records_by_epoch[record["epoch"]] = record
            ordered = [records_by_epoch[key] for key in sorted(records_by_epoch)]
            temporary_path = self.log_path.with_suffix(".json.tmp")
            temporary_path.write_text(
                json.dumps(ordered, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            temporary_path.replace(self.log_path)

    return DetailedTrainingLogger()


def main(argv=None) -> None:
    args = parse_args(argv)
    validate_args(args)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required; install requirements-keras.txt first"
        ) from exc
    from ai.keras_grading.custom_objects import (
        CohenKappaMetric,
        OrdinalAccuracy,
        OrdinalMeanAbsoluteError,
        ordinal_loss,
    )

    tf.keras.utils.set_random_seed(args.seed)
    splits = find_splits(args.dataset_dir)
    frames = {name: scan_labeled_split(path) for name, path in splits.items()}
    unlabeled = discover_images(args.unlabeled_dir)
    if args.max_unlabeled_images:
        rng = np.random.default_rng(args.seed)
        indices = np.sort(
            rng.choice(
                len(unlabeled),
                size=min(args.max_unlabeled_images, len(unlabeled)),
                replace=False,
            )
        )
        unlabeled = [unlabeled[int(index)] for index in indices]
    assert_external(unlabeled, frames)

    grader = KerasOrdinalGrader(
        args.model,
        threshold_path=args.thresholds,
        use_tta=not args.no_tta,
    )
    pseudo_path = output_dir / "pseudo_labels.csv"
    pseudo_cache_path = output_dir / "pseudo-cache.json"
    manifest_path = output_dir / "training_manifest.csv"
    labeled = frames["train"].copy()
    labeled["sample_weight"] = 1.0
    labeled["source"] = "labeled"
    cache_signature = pseudo_cache_signature(args, grader, unlabeled)
    pseudo_frame = read_matching_pseudo_cache(
        pseudo_cache_path, pseudo_path, cache_signature
    )
    if pseudo_frame is not None:
        print(
            "PSEUDO CACHE HIT: thresholds/model/unlabeled inventory unchanged; "
            f"reusing {pseudo_path}",
            flush=True,
        )
        pseudo_frame["sample_weight"] = args.pseudo_weight
        pseudo_frame["source"] = "pseudo"
    else:
        if pseudo_cache_path.exists() or pseudo_path.exists():
            print(
                "PSEUDO CACHE MISS: a threshold, model, TTA setting, selection "
                "limit, or unlabeled inventory changed. Regenerating labels.",
                flush=True,
            )
        predictions = []
        for index, path in enumerate(unlabeled, 1):
            predictions.append((path, grader.predict_path(path)))
            if index == 1 or index % 100 == 0 or index == len(unlabeled):
                print(f"Pseudo-label inference {index}/{len(unlabeled)}", flush=True)
        pseudo = select_pseudo_labels(
            predictions,
            minimum_confidence=args.pseudo_confidence,
            max_per_class=args.max_pseudo_per_class,
        )
        if not pseudo:
            raise ValueError(
                "No pseudo-label met the confidence threshold; lower "
                "--pseudo-confidence only after inspecting calibration"
            )
        pseudo_frame = pd.DataFrame(
            {
                "image_path": [str(item.image_path) for item in pseudo],
                "diagnosis": [item.grade for item in pseudo],
                "confidence": [item.confidence for item in pseudo],
                "sample_weight": [args.pseudo_weight for _ in pseudo],
                "source": "pseudo",
            }
        )
        pseudo_frame.to_csv(pseudo_path, index=False)
        pseudo_cache_path.write_text(
            json.dumps(
                {
                    "complete": True,
                    "signature": cache_signature,
                    "selected_images": len(pseudo_frame),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    combined = pd.concat([labeled, pseudo_frame], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    combined.to_csv(manifest_path, index=False)

    image_size = int(grader.model.input_shape[1])
    train_dataset = create_dataset(
        tf, combined, image_size, args.batch_size, training=True
    )
    validation = frames["val"].copy()
    validation["sample_weight"] = 1.0
    validation_dataset = create_dataset(
        tf, validation, image_size, args.batch_size, training=False
    )

    print("\nDATA SUMMARY", flush=True)
    print(f"  labeled train : {len(labeled):,}", flush=True)
    print(f"  unlabeled scan: {len(unlabeled):,}", flush=True)
    print(f"  pseudo kept   : {len(pseudo_frame):,}", flush=True)
    print(f"  combined train: {len(combined):,}", flush=True)
    print(f"  validation    : {len(validation):,}", flush=True)
    print(
        "  pseudo grades : "
        f"{pseudo_frame['diagnosis'].value_counts().sort_index().to_dict()}",
        flush=True,
    )
    print(f"  model load    : {grader.load_mode}", flush=True)
    print(f"  thresholds    : {grader.thresholds.tolist()}", flush=True)

    metric_thresholds = grader.thresholds.tolist()
    grader.model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=args.learning_rate, weight_decay=args.weight_decay
        ),
        loss=ordinal_loss,
        metrics=[
            OrdinalAccuracy(thresholds=metric_thresholds),
            CohenKappaMetric(thresholds=metric_thresholds),
            OrdinalMeanAbsoluteError(thresholds=metric_thresholds),
        ],
    )
    checkpoint_path = output_dir / "checkpoint-best.keras"
    callbacks = [
        tf.keras.callbacks.BackupAndRestore(
            backup_dir=output_dir / "training-backup",
            delete_checkpoint=False,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_kappa",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_kappa",
            mode="max",
            patience=args.patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.CSVLogger(
            output_dir / "history.csv", append=args.resume
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=output_dir / "tensorboard",
            update_freq="epoch",
            profile_batch=0,
        ),
        detailed_logging_callback(tf, output_dir, args.log_every_batches),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    history = grader.model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    grader.model.save(output_dir / "checkpoint-last.keras")
    metadata = {
        "teacher_model": str(args.model.resolve()),
        "fixed_splits": {name: str(path) for name, path in splits.items()},
        "test_images_used_during_training": 0,
        "labeled_train_images": len(labeled),
        "unlabeled_scanned": len(unlabeled),
        "pseudo_labels_selected": len(pseudo_frame),
        "pseudo_class_counts": {
            int(grade): int(count)
            for grade, count in pseudo_frame["diagnosis"].value_counts().items()
        },
        "ordinal_thresholds": grader.thresholds.tolist(),
        "args": vars(args),
        "history": history.history,
    }
    (output_dir / "run.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
