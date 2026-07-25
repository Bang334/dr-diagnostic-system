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


CACHE_SCHEMA_VERSION = 3
PSEUDO_COLUMNS = ("image_path", "pseudo_label", "confidence")


def _stable_file_identity(path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {
        "name": resolved.name,
        "size": int(stat.st_size),
    }


def _manifest_digest(paths: Iterable[Path]) -> tuple[str, int]:
    resolved_paths = tuple(Path(item).expanduser().resolve() for item in paths)
    if not resolved_paths:
        return hashlib.sha256(b"").hexdigest(), 0
    common_root = Path(os.path.commonpath([os.fspath(path.parent) for path in resolved_paths]))
    digest = hashlib.sha256()
    entries = []
    for path in resolved_paths:
        entries.append(
            {
                "relative_path": path.relative_to(common_root).as_posix(),
                "size": int(path.stat().st_size),
            }
        )
    for identity in sorted(entries, key=lambda item: str(item["relative_path"])):
        digest.update(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest(), len(entries)


@dataclass(frozen=True)
class PseudoLabelCacheSpec:
    """Everything that can change teacher predictions or accepted labels."""

    grade_thresholds: tuple[float, ...]
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
            "grade_thresholds": [float(value) for value in self.grade_thresholds],
            "teacher_checkpoint": _stable_file_identity(self.teacher_checkpoint),
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
        self.progress_path = self.cache_dir / "progress.csv"
        self.progress_metadata_path = self.cache_dir / "work.json"

    def load_or_generate(
        self,
        spec: PseudoLabelCacheSpec,
        generator: Callable[[], pd.DataFrame],
    ) -> CacheResult:
        contract = spec.contract()
        serialized = json.dumps(contract, sort_keys=True, separators=(",", ":"))
        cache_key = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        csv_path = self.cache_dir / "labels.csv"
        metadata_path = self.cache_dir / "meta.json"

        if csv_path.is_file() and metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if self._metadata_matches(metadata, contract, cache_key):
                frame = self._read_frame(csv_path)
                if metadata.get("cache_key") != cache_key:
                    self._write_cache(frame, csv_path, metadata_path, contract, cache_key)
                return CacheResult(frame, True, csv_path, metadata_path, cache_key)

        self._prepare_progress(contract, cache_key)
        frame = self._validate_frame(generator())
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._write_cache(frame, csv_path, metadata_path, contract, cache_key)
        self.progress_path.unlink(missing_ok=True)
        self.progress_metadata_path.unlink(missing_ok=True)
        return CacheResult(frame, False, csv_path, metadata_path, cache_key)

    def _prepare_progress(
        self, contract: dict[str, object], cache_key: str
    ) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        matching = False
        if self.progress_metadata_path.is_file():
            try:
                saved = json.loads(
                    self.progress_metadata_path.read_text(encoding="utf-8")
                )
                matching = (
                    saved.get("cache_key") == cache_key
                    and saved.get("contract") == contract
                )
            except (OSError, ValueError, TypeError):
                matching = False
        if not matching:
            self.progress_path.unlink(missing_ok=True)
        self.progress_metadata_path.write_text(
            json.dumps(
                {"cache_key": cache_key, "contract": contract},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _metadata_matches(
        metadata: dict[str, object],
        contract: dict[str, object],
        cache_key: str,
    ) -> bool:
        saved = metadata.get("contract")
        if not isinstance(saved, dict):
            return False
        if metadata.get("cache_key") == cache_key and saved == contract:
            return True
        # Compatibility with the original Colab cache. Schema v1 included
        # absolute /content paths and mtimes, both of which change whenever a
        # runtime downloads the same files again.
        if saved.get("schema_version") != 1:
            return False
        stable_fields = (
            "threshold",
            "unlabeled_image_count",
            "preprocessing",
            "image_size",
            "max_pseudo_per_class",
            "enhance",
        )
        if any(saved.get(field) != contract.get(field) for field in stable_fields):
            return False
        old_teacher = saved.get("teacher_checkpoint")
        new_teacher = contract.get("teacher_checkpoint")
        if not isinstance(old_teacher, dict) or not isinstance(new_teacher, dict):
            return False
        return (
            Path(str(old_teacher.get("path", ""))).name == new_teacher.get("name")
            and old_teacher.get("size") == new_teacher.get("size")
        )

    @staticmethod
    def _write_cache(
        frame: pd.DataFrame,
        csv_path: Path,
        metadata_path: Path,
        contract: dict[str, object],
        cache_key: str,
    ) -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
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
