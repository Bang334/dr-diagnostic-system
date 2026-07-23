"""Persistent, auditable pseudo-label cache for semi-supervised training.

The cache interface deliberately hides fingerprinting, validation and atomic
writes. Callers supply a prediction contract and a generator; they receive a
validated DataFrame whether it came from disk or fresh inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd


CACHE_SCHEMA_VERSION = 1
PSEUDO_COLUMNS = ("image_path", "pseudo_label", "confidence")


def _file_identity(path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": os.fspath(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _manifest_digest(paths: Iterable[Path]) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for path in sorted((Path(item).expanduser().resolve() for item in paths), key=os.fspath):
        identity = _file_identity(path)
        digest.update(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
        count += 1
    return digest.hexdigest(), count


@dataclass(frozen=True)
class PseudoLabelCacheSpec:
    """Everything that can change teacher predictions or accepted labels."""

    threshold: float
    teacher_checkpoint: Path
    unlabeled_paths: tuple[Path, ...]
    preprocessing: str
    image_size: int
    max_pseudo_per_class: int
    enhance: bool

    def contract(self) -> dict[str, object]:
        manifest_sha256, image_count = _manifest_digest(self.unlabeled_paths)
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "threshold": float(self.threshold),
            "teacher_checkpoint": _file_identity(self.teacher_checkpoint),
            "unlabeled_manifest_sha256": manifest_sha256,
            "unlabeled_image_count": image_count,
            "preprocessing": self.preprocessing,
            "image_size": int(self.image_size),
            "max_pseudo_per_class": int(self.max_pseudo_per_class),
            "enhance": bool(self.enhance),
        }


@dataclass(frozen=True)
class CacheResult:
    frame: pd.DataFrame
    reused: bool
    csv_path: Path
    metadata_path: Path
    cache_key: str


class PseudoLabelCache:
    """Load a matching pseudo-label artifact or create it exactly once."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir.expanduser().resolve()

    def load_or_generate(
        self,
        spec: PseudoLabelCacheSpec,
        generator: Callable[[], pd.DataFrame],
    ) -> CacheResult:
        contract = spec.contract()
        serialized = json.dumps(contract, sort_keys=True, separators=(",", ":"))
        cache_key = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        stem = f"pseudo-labels-{cache_key[:20]}"
        csv_path = self.cache_dir / f"{stem}.csv"
        metadata_path = self.cache_dir / f"{stem}.json"

        if csv_path.is_file() and metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("cache_key") == cache_key and metadata.get("contract") == contract:
                frame = self._read_frame(csv_path)
                return CacheResult(frame, True, csv_path, metadata_path, cache_key)

        frame = self._validate_frame(generator())
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary_csv = csv_path.with_suffix(".csv.tmp")
        temporary_metadata = metadata_path.with_suffix(".json.tmp")
        frame.to_csv(temporary_csv, index=False)
        metadata = {
            "cache_key": cache_key,
            "contract": contract,
            "accepted_pseudo_labels": len(frame),
            "columns": list(PSEUDO_COLUMNS),
        }
        temporary_metadata.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary_csv.replace(csv_path)
        temporary_metadata.replace(metadata_path)
        return CacheResult(frame, False, csv_path, metadata_path, cache_key)

    @staticmethod
    def _read_frame(path: Path) -> pd.DataFrame:
        return PseudoLabelCache._validate_frame(pd.read_csv(path))

    @staticmethod
    def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
        missing = set(PSEUDO_COLUMNS).difference(frame.columns)
        if missing:
            raise ValueError(f"Pseudo-label cache is missing columns: {sorted(missing)}")
        normalized = frame.loc[:, list(PSEUDO_COLUMNS)].copy()
        normalized["pseudo_label"] = normalized["pseudo_label"].astype(int)
        normalized["confidence"] = normalized["confidence"].astype(float)
        if not normalized["pseudo_label"].between(0, 4).all():
            raise ValueError("Pseudo-label cache contains a grade outside 0..4")
        if not normalized["confidence"].between(0.0, 1.0).all():
            raise ValueError("Pseudo-label cache contains confidence outside [0, 1]")
        return normalized
