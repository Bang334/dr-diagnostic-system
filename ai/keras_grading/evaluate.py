"""Evaluate the Keras ordinal teacher on fixed class-folder splits."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

from ai.keras_grading.grader import KerasOrdinalGrader


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def scan_class_folder(root: Path) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for grade in range(5):
        class_dir = root / str(grade)
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing grade directory: {class_dir}")
        samples.extend(
            (path.resolve(), grade)
            for path in sorted(class_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    if not samples:
        raise ValueError(f"No supported images found under {root}")
    return samples


def evaluate_samples(
    grader: KerasOrdinalGrader,
    samples,
    *,
    progress_every: int = 10,
):
    if progress_every < 1:
        raise ValueError("progress_every must be positive")
    truth: list[int] = []
    predicted: list[int] = []
    rows: list[dict[str, object]] = []
    total_samples = len(samples)
    started = time.time()
    print("\n" + "=" * 78, flush=True)
    print("BASELINE VALIDATION STARTED", flush=True)
    print(f"Images       : {total_samples:,}", flush=True)
    print(f"Model input  : {grader.model.input_shape}", flush=True)
    print(f"Model output : {grader.model.output_shape}", flush=True)
    print(f"Load mode    : {grader.load_mode}", flush=True)
    print(f"TTA          : {'original + horizontal flip' if grader.use_tta else 'off'}", flush=True)
    print(f"Thresholds   : {grader.thresholds.tolist()}", flush=True)
    print("Starting first image...", flush=True)
    print("=" * 78, flush=True)
    for index, (path, label) in enumerate(samples, 1):
        prediction = grader.predict_path(path)
        truth.append(label)
        predicted.append(prediction.grade)
        rows.append({"image_path": str(path), "truth": label, **prediction.as_dict()})
        if index == 1 or index % progress_every == 0 or index == total_samples:
            elapsed = time.time() - started
            images_per_second = index / max(elapsed, 1e-6)
            remaining = (total_samples - index) / max(images_per_second, 1e-6)
            running_accuracy = float(np.mean(np.equal(truth, predicted)))
            print(
                f"[baseline] {index:,}/{total_samples:,} "
                f"({index / total_samples:.1%}) | "
                f"elapsed={elapsed / 60:.1f}m | "
                f"speed={images_per_second:.2f} img/s | "
                f"ETA={remaining / 60:.1f}m | "
                f"running_accuracy={running_accuracy:.4f} | "
                f"last=true:{label},pred:{prediction.grade}",
                flush=True,
            )

    metrics = {
        "samples": len(samples),
        "accuracy": accuracy_score(truth, predicted),
        "balanced_accuracy": balanced_accuracy_score(truth, predicted),
        "macro_f1": f1_score(truth, predicted, average="macro", zero_division=0),
        "qwk": cohen_kappa_score(truth, predicted, weights="quadratic"),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=range(5)).tolist(),
        "classification_report": classification_report(
            truth, predicted, labels=range(5), output_dict=True, zero_division=0
        ),
    }
    print("=" * 78, flush=True)
    print(
        "BASELINE COMPLETED | "
        f"accuracy={metrics['accuracy']:.4f} | "
        f"balanced_accuracy={metrics['balanced_accuracy']:.4f} | "
        f"macro_f1={metrics['macro_f1']:.4f} | "
        f"QWK={metrics['qwk']:.4f} | "
        f"duration={(time.time() - started) / 60:.1f}m",
        flush=True,
    )
    print("=" * 78, flush=True)
    return metrics, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading model: {args.model}", flush=True)
    grader = KerasOrdinalGrader(
        args.model,
        threshold_path=args.thresholds,
        use_tta=not args.no_tta,
    )
    print(f"Scanning validation split: {args.split_dir}", flush=True)
    samples = scan_class_folder(args.split_dir)
    print(f"Found {len(samples):,} labeled images", flush=True)
    metrics, rows = evaluate_samples(grader, samples)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
