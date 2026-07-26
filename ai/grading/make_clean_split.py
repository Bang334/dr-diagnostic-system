from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUGMENT_SUFFIXES = (
    "-600-HFF",
    "-600-HB",
    "-600-FS",
    "-600-FA",
    "-600",
)


def base_id(path: Path) -> str:
    stem = path.stem
    for suffix in AUGMENT_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if re.match(r"^[0-4]_", stem):
        stem = stem[2:]
    return stem


def stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha1(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def choose_split(base: str, old_splits: set[str], val_ratio: float, test_ratio: float, seed: int) -> str:
    if len(old_splits) == 1:
        return next(iter(old_splits))

    r = stable_fraction(base, seed)
    if r < test_ratio:
        return "test"
    if r < test_ratio + val_ratio:
        return "validation"
    return "train"


def collect_groups(source: Path) -> dict[tuple[str, str], list[tuple[Path, str, str]]]:
    groups: dict[tuple[str, str], list[tuple[Path, str, str]]] = defaultdict(list)
    for split in ("train", "validation", "test"):
        split_dir = source / split
        if not split_dir.exists():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            label = class_dir.name
            for path in class_dir.iterdir():
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    groups[(label, base_id(path))].append((path, split, label))
    return groups


def copy_dataset(source: Path, output: Path, seed: int, val_ratio: float, test_ratio: float) -> dict:
    groups = collect_groups(source)
    if not groups:
        raise RuntimeError(f"No images found in {source}")

    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        for label in map(str, range(5)):
            (output / split / label).mkdir(parents=True, exist_ok=True)

    file_counts: dict[str, Counter] = defaultdict(Counter)
    group_counts: dict[str, Counter] = defaultdict(Counter)
    moved_mixed_groups = 0
    copied_files = 0

    for (label, base), items in sorted(groups.items()):
        old_splits = {split for _, split, _ in items}
        target_split = choose_split(base, old_splits, val_ratio, test_ratio, seed)
        if len(old_splits) > 1:
            moved_mixed_groups += 1

        group_counts[target_split][label] += 1
        for src, _, _ in items:
            dst = output / target_split / label / src.name
            if dst.exists():
                dst = output / target_split / label / f"{src.stem}__dup_{copied_files}{src.suffix}"
            shutil.copy2(src, dst)
            file_counts[target_split][label] += 1
            copied_files += 1

    leakage = detect_cross_split_leakage(output)
    summary = {
        "source": str(source),
        "output": str(output),
        "seed": seed,
        "val_ratio_for_mixed_groups": val_ratio,
        "test_ratio_for_mixed_groups": test_ratio,
        "total_base_label_groups": len(groups),
        "mixed_groups_reassigned_by_hash": moved_mixed_groups,
        "copied_files": copied_files,
        "file_counts": {k: dict(sorted(v.items())) for k, v in file_counts.items()},
        "group_counts": {k: dict(sorted(v.items())) for k, v in group_counts.items()},
        "cross_split_leakage_after": leakage,
    }
    (output / "clean_split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def detect_cross_split_leakage(root: Path) -> dict:
    seen: dict[tuple[str, str], set[str]] = defaultdict(set)
    for split in ("train", "validation", "test"):
        for path in (root / split).glob("*/*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                seen[(path.parent.name, base_id(path))].add(split)

    leaked = {key: splits for key, splits in seen.items() if len(splits) > 1}
    return {
        "leaked_base_label_groups": len(leaked),
        "checked_base_label_groups": len(seen),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a leakage-free DR dataset split by base image id.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        if not args.overwrite:
            raise SystemExit(f"Output exists: {args.output}. Use --overwrite to replace it.")
        shutil.rmtree(args.output)

    summary = copy_dataset(args.source, args.output, args.seed, args.val_ratio, args.test_ratio)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
