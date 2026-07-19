"""Download and extract the merged five-source fundus dataset from Kaggle."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path


DATASET_REF = "sehastrajits/fundus-aptosddridirdeyepacsmessidor"
ARCHIVE_NAME = "fundus-aptosddridirdeyepacsmessidor.zip"


def has_kaggle_auth() -> bool:
    return bool(os.environ.get("KAGGLE_API_TOKEN")) or (
        Path.home() / ".kaggle" / "kaggle.json"
    ).is_file()


def extract_archive(archive_path: Path, output_dir: Path) -> None:
    """Extract a trusted Kaggle archive while rejecting path traversal."""
    output_root = output_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            destination = (output_dir / member.filename).resolve()
            if destination != output_root and output_root not in destination.parents:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        archive.extractall(output_dir)


def download_dataset(output_dir: Path, *, force: bool, keep_archive: bool) -> Path:
    split_dir = output_dir / "split_dataset"
    if split_dir.is_dir() and not force:
        print(f"Dataset is already extracted at {split_dir}")
        return split_dir
    if not has_kaggle_auth():
        raise RuntimeError(
            "Kaggle credentials are missing. Set KAGGLE_API_TOKEN or place "
            "kaggle.json under ~/.kaggle/."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET_REF,
        "-p",
        os.fspath(output_dir),
    ]
    if force:
        command.append("--force")
    print(f"Downloading {DATASET_REF} (about 10.9 GB)...")
    subprocess.run(command, check=True)

    archive_path = output_dir / ARCHIVE_NAME
    if not archive_path.is_file():
        raise FileNotFoundError(f"Kaggle did not create the expected archive: {archive_path}")
    print(f"Extracting {archive_path}...")
    extract_archive(archive_path, output_dir)
    if not split_dir.is_dir():
        raise RuntimeError(f"Archive is missing the expected directory: {split_dir}")
    if not keep_archive:
        archive_path.unlink()
    print(f"Dataset ready: {split_dir}")
    return split_dir


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "raw" / "merged_fundus",
    )
    parser.add_argument("--force", action="store_true", help="Download again if present")
    parser.add_argument("--keep-archive", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    options = parse_args()
    download_dataset(
        options.output_dir.expanduser().resolve(),
        force=options.force,
        keep_archive=options.keep_archive,
    )
