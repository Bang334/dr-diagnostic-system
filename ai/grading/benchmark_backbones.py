"""Run and summarize a controlled EfficientNet/ResNet/ConvNeXt comparison."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from ai.grading.backbones import BACKBONE_PRESETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preprocessing", default="rgb_crop")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def training_command(args: argparse.Namespace, architecture: str) -> list[str]:
    return [
        sys.executable, "-m", "ai.grading.train",
        "--dataset-dir", str(args.dataset_dir),
        "--output-dir", str(args.output_dir / architecture),
        "--model-source", "timm",
        "--architecture", architecture,
        "--image-size", "300" if architecture == "efficientnet" else "224",
        "--epochs", str(args.epochs),
        "--seed", str(args.seed),
        "--preprocessing", args.preprocessing,
    ]


def summarize(output_dir: Path) -> list[dict]:
    rows = []
    for architecture, model_name in BACKBONE_PRESETS.items():
        path = output_dir / architecture / "test_metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing benchmark result: {path}")
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "architecture": architecture,
            "model_name": model_name,
            "qwk": metrics["qwk"],
            "macro_f1": metrics["macro_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "accuracy": metrics["accuracy"],
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "backbone_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    args = parse_args()
    if not args.summarize_only:
        for architecture in BACKBONE_PRESETS:
            subprocess.run(training_command(args, architecture), check=True)
    rows = summarize(args.output_dir)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
