"""Leakage audit for multi-source fundus classification manifests."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


SOURCE_ALIASES = {
    "aptos": "APTOS",
    "ddr": "DDR",
    "idrid": "IDRiD",
    "idird": "IDRiD",
    "eyepacs": "EyePACS",
    "messidor": "Messidor",
}


def infer_source(image_id: str) -> str:
    normalized = image_id.lower().replace("\\", "/")
    for token, source in SOURCE_ALIASES.items():
        if token in normalized:
            return source
    return "unknown"


def infer_patient_and_laterality(image_id: str, source: str) -> tuple[str | None, str | None]:
    stem = Path(image_id).stem
    if source == "EyePACS" or re.search(r"_(left|right)$", stem, flags=re.IGNORECASE):
        match = re.match(r"^(.+?)_(left|right)$", stem, flags=re.IGNORECASE)
        if match:
            return f"EyePACS:{match.group(1)}", match.group(2).lower()
    return None, None


def _sha256_bytes(parts: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def decoded_pixel_hash(image: np.ndarray) -> str:
    return _sha256_bytes(
        [
            np.asarray(image.shape, dtype=np.int64).tobytes(),
            str(image.dtype).encode("ascii"),
            np.ascontiguousarray(image).tobytes(),
        ]
    )


def perceptual_hash(image_bgr: np.ndarray) -> int:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    coefficients = cv2.dct(resized.astype(np.float32))[:8, :8]
    values = coefficients.flatten()
    threshold = float(np.median(values[1:]))
    bits = values > threshold
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count() if hasattr(int, "bit_count") else bin(left ^ right).count("1")


class _BKNode:
    def __init__(self, value: int, index: int):
        self.value = value
        self.indices = [index]
        self.children: dict[int, "_BKNode"] = {}


class _BKTree:
    def __init__(self):
        self.root: _BKNode | None = None

    def add(self, value: int, index: int) -> None:
        if self.root is None:
            self.root = _BKNode(value, index)
            return
        node = self.root
        while True:
            distance = hamming_distance(value, node.value)
            if distance == 0:
                node.indices.append(index)
                return
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = _BKNode(value, index)
                return
            node = child

    def query(self, value: int, maximum_distance: int) -> list[int]:
        if self.root is None:
            return []
        matches: list[int] = []
        pending = [self.root]
        while pending:
            node = pending.pop()
            distance = hamming_distance(value, node.value)
            if distance <= maximum_distance:
                matches.extend(node.indices)
            lower = distance - maximum_distance
            upper = distance + maximum_distance
            pending.extend(
                child
                for edge, child in node.children.items()
                if lower <= edge <= upper
            )
        return matches


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


@dataclass(frozen=True)
class AuditResult:
    splits: dict[str, pd.DataFrame]
    report: dict[str, Any]
    violations: pd.DataFrame


def _resolve_image_path(row: pd.Series, args: Any) -> Path:
    if "image_path" in row and pd.notna(row["image_path"]):
        return Path(str(row["image_path"]))
    image_id = str(row[args.image_column])
    path = Path(args.images_dir) / image_id
    if not path.suffix:
        path = path.with_suffix(args.image_extension)
    return path


def audit_splits(
    splits: dict[str, pd.DataFrame],
    args: Any,
    output_dir: Path,
    *,
    phash_distance: int = 4,
    fail_on_leakage: bool = True,
) -> AuditResult:
    records: list[dict[str, Any]] = []
    for split_name, frame in splits.items():
        for _, row in frame.iterrows():
            path = _resolve_image_path(row, args).expanduser().resolve()
            image_id = str(
                row["image_id"] if "image_id" in row else row[args.image_column]
            )
            raw_source = row.get("source")
            source = (
                infer_source(f"{image_id}/{path.as_posix()}")
                if raw_source is None or pd.isna(raw_source) or not str(raw_source).strip()
                else str(raw_source)
            )
            patient_id = row.get("patient_id")
            laterality = row.get("laterality")
            if (
                patient_id is None
                or pd.isna(patient_id)
                or not str(patient_id).strip()
            ):
                patient_id, inferred_laterality = infer_patient_and_laterality(image_id, source)
                if laterality is None or pd.isna(laterality) or not str(laterality).strip():
                    laterality = inferred_laterality
            encoded = path.read_bytes()
            image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Could not read image during leakage audit: {path}")
            records.append(
                {
                    "split": split_name,
                    "image_path": os.fspath(path),
                    "image_id": image_id,
                    "source": source,
                    "patient_id": patient_id,
                    "laterality": laterality,
                    "diagnosis": int(row[args.label_column]),
                    "file_sha256": hashlib.sha256(encoded).hexdigest(),
                    "pixel_sha256": decoded_pixel_hash(image),
                    "phash": f"{perceptual_hash(image):016x}",
                }
            )

    manifest = pd.DataFrame.from_records(records)
    union = _UnionFind(len(manifest))
    for column in ("file_sha256", "pixel_sha256"):
        for indices in manifest.groupby(column, sort=False).indices.values():
            indices = list(indices)
            for index in indices[1:]:
                union.union(indices[0], index)

    tree = _BKTree()
    phashes = [int(value, 16) for value in manifest["phash"]]
    for index, value in enumerate(tqdm(phashes, desc="perceptual duplicate audit", leave=False)):
        for match in tree.query(value, phash_distance):
            union.union(index, match)
        tree.add(value, index)
    roots = [union.find(index) for index in range(len(manifest))]
    root_to_cluster = {
        root: f"dup-{cluster_index:06d}"
        for cluster_index, root in enumerate(sorted(set(roots)))
    }
    manifest["duplicate_cluster_id"] = [root_to_cluster[root] for root in roots]

    violations: list[dict[str, Any]] = []
    checks = [
        ("file_sha256", "exact_file_duplicate", True),
        ("pixel_sha256", "exact_pixel_duplicate", True),
        ("duplicate_cluster_id", "perceptual_duplicate", False),
        ("patient_id", "patient_overlap", True),
    ]
    for column, kind, blocking in checks:
        candidate = manifest.dropna(subset=[column])
        for value, group in candidate.groupby(column, sort=False):
            if group["split"].nunique() <= 1:
                continue
            violations.append(
                {
                    "kind": kind,
                    "blocking": blocking,
                    "value": value,
                    "splits": ",".join(sorted(group["split"].unique())),
                    "image_ids": " | ".join(group["image_id"].astype(str).tolist()),
                    "labels": ",".join(map(str, sorted(group["diagnosis"].unique()))),
                }
            )
    violations_frame = pd.DataFrame.from_records(violations)
    blocking_count = (
        int(violations_frame["blocking"].sum()) if not violations_frame.empty else 0
    )
    exact_label_conflicts = sum(
        int(group["diagnosis"].nunique() > 1)
        for column in ("file_sha256", "pixel_sha256")
        for _, group in manifest.groupby(column)
    )
    perceptual_label_conflicts = int(
        sum(
            group["diagnosis"].nunique() > 1
            for _, group in manifest.groupby("duplicate_cluster_id")
        )
    )
    report = {
        "images": len(manifest),
        "split_counts": manifest["split"].value_counts().sort_index().to_dict(),
        "source_counts": manifest["source"].value_counts().sort_index().to_dict(),
        "patient_id_coverage": float(manifest["patient_id"].notna().mean()),
        "duplicate_clusters": int(manifest["duplicate_cluster_id"].nunique()),
        "cross_split_violations": len(violations_frame),
        "blocking_cross_split_violations": blocking_count,
        "exact_duplicate_label_conflicts": exact_label_conflicts,
        "perceptual_cluster_label_conflicts": perceptual_label_conflicts,
        "phash_max_distance": phash_distance,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "audit_manifest.csv", index=False)
    violations_frame.to_csv(output_dir / "leakage_violations.csv", index=False)
    with (output_dir / "audit_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    augmented: dict[str, pd.DataFrame] = {}
    audit_columns = [
        "image_id",
        "source",
        "patient_id",
        "laterality",
        "file_sha256",
        "pixel_sha256",
        "phash",
        "duplicate_cluster_id",
    ]
    for split_name, frame in splits.items():
        additions = manifest.loc[manifest["split"] == split_name, audit_columns]
        augmented[split_name] = frame.merge(additions, on="image_id", how="left")

    if fail_on_leakage and (blocking_count or exact_label_conflicts):
        raise ValueError(
            "Dataset leakage audit failed: "
            f"{blocking_count} blocking cross-split groups, "
            f"{exact_label_conflicts} exact duplicate groups with conflicting labels. "
            f"See {output_dir / 'leakage_violations.csv'}"
        )
    return AuditResult(augmented, report, violations_frame)
