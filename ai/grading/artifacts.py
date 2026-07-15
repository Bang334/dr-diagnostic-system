"""Versioned model artifacts shared by DR grading training and inference."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ARTIFACT_SCHEMA_VERSION = 1
NUM_CLASSES = 5
CLASS_NAMES = (
    "No DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "Proliferative DR",
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ModelSpec:
    framework: str
    model_source: str
    architecture: str
    num_classes: int = NUM_CLASSES
    output_type: str = "logits"


@dataclass(frozen=True)
class PreprocessingSpec:
    image_size: int
    crop_tolerance: int = 7
    enhance: bool = False
    color_space: str = "RGB"
    resize_interpolation: str = "bicubic"
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD


@dataclass(frozen=True)
class ArtifactMetadata:
    schema_version: int
    model: ModelSpec
    preprocessing: PreprocessingSpec
    class_names: tuple[str, ...]
    grading_scale: str
    training: dict[str, Any]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def manifest_hashes(manifest_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        path = manifest_dir / f"{split_name}.csv"
        if path.is_file():
            hashes[f"{split_name}_manifest_sha256"] = sha256_file(path)
    return hashes


def package_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    for module_name in ("torchvision", "timm", "cv2", "numpy", "sklearn"):
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[module_name] = "unavailable"
    return versions


def git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def build_metadata(
    args: Any,
    *,
    manifest_dir: Path,
    project_root: Path,
) -> ArtifactMetadata:
    model_source = str(args.model_source)
    architecture = (
        "vit_large_patch14_dinov2.lvd142m"
        if model_source == "retfound"
        else str(args.model_name)
    )
    training = {
        key: value
        for key, value in vars(args).items()
        if key not in {"resume", "output_dir", "dataset_dir", "images_dir", "labels_csv"}
    }
    provenance = {
        "git_commit": git_commit(project_root),
        "package_versions": package_versions(),
        **manifest_hashes(manifest_dir),
    }
    return ArtifactMetadata(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        model=ModelSpec(
            framework="pytorch",
            model_source=model_source,
            architecture=architecture,
            num_classes=NUM_CLASSES,
            output_type="logits",
        ),
        preprocessing=PreprocessingSpec(
            image_size=int(args.image_size),
            crop_tolerance=int(getattr(args, "crop_tolerance", 7)),
            enhance=bool(args.enhance),
        ),
        class_names=CLASS_NAMES,
        grading_scale="ICDR",
        training=_jsonable(training),
        provenance=_jsonable(provenance),
    )


def metadata_from_dict(raw: Mapping[str, Any]) -> ArtifactMetadata:
    schema_version = int(raw.get("schema_version", -1))
    if schema_version != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported DR artifact schema {schema_version}; "
            f"expected {ARTIFACT_SCHEMA_VERSION}"
        )
    model = ModelSpec(**dict(raw["model"]))
    preprocessing_raw = dict(raw["preprocessing"])
    preprocessing_raw["mean"] = tuple(preprocessing_raw.get("mean", IMAGENET_MEAN))
    preprocessing_raw["std"] = tuple(preprocessing_raw.get("std", IMAGENET_STD))
    preprocessing = PreprocessingSpec(**preprocessing_raw)
    if preprocessing.color_space != "RGB":
        raise ValueError(f"Unsupported checkpoint color space: {preprocessing.color_space}")
    if preprocessing.resize_interpolation != "bicubic":
        raise ValueError(
            "Unsupported checkpoint resize interpolation: "
            f"{preprocessing.resize_interpolation}"
        )
    class_names = tuple(str(name) for name in raw["class_names"])
    if model.num_classes != len(class_names):
        raise ValueError("Checkpoint class names do not match model output dimension")
    return ArtifactMetadata(
        schema_version=schema_version,
        model=model,
        preprocessing=preprocessing,
        class_names=class_names,
        grading_scale=str(raw.get("grading_scale", "ICDR")),
        training=dict(raw.get("training", {})),
        provenance=dict(raw.get("provenance", {})),
    )


def legacy_metadata_from_checkpoint(checkpoint: Mapping[str, Any]) -> ArtifactMetadata:
    """Read checkpoints produced before the versioned artifact contract."""
    args = dict(checkpoint.get("args", {}))
    model_source = str(args.get("model_source", "retfound"))
    architecture = (
        "vit_large_patch14_dinov2.lvd142m"
        if model_source == "retfound"
        else str(args.get("model_name", "convnext_tiny.fb_in22k_ft_in1k"))
    )
    loss_name = str(args.get("loss", "ce"))
    output_dim = NUM_CLASSES if loss_name == "ce" else NUM_CLASSES - 1
    return ArtifactMetadata(
        schema_version=0,
        model=ModelSpec(
            framework="pytorch",
            model_source=model_source,
            architecture=architecture,
            num_classes=output_dim,
            output_type="logits" if loss_name == "ce" else "ordinal_logits",
        ),
        preprocessing=PreprocessingSpec(
            image_size=int(args.get("image_size", 224)),
            enhance=bool(args.get("enhance", False)),
        ),
        class_names=CLASS_NAMES,
        grading_scale="ICDR",
        training=_jsonable(args),
        provenance={"legacy_checkpoint": True},
    )


def load_torch_checkpoint(path: Path, *, map_location: str = "cpu") -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument
        checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Not a DR grading checkpoint: {path}")
    return checkpoint


def checkpoint_metadata(checkpoint: Mapping[str, Any]) -> ArtifactMetadata:
    raw = checkpoint.get("metadata")
    if raw is None:
        return legacy_metadata_from_checkpoint(checkpoint)
    if not isinstance(raw, Mapping):
        raise ValueError("Checkpoint metadata must be a mapping")
    return metadata_from_dict(raw)


def validate_resume_metadata(
    saved: ArtifactMetadata,
    current: ArtifactMetadata,
    *,
    immutable_training_keys: Sequence[str] = (
        "loss",
        "balance",
        "effective_beta",
        "adaptation",
        "last_n_blocks",
        "batch_size",
        "accum_steps",
        "epochs",
        "warmup_epochs",
        "peak_lr",
        "layer_decay",
        "min_lr",
        "weight_decay",
        "label_smoothing",
        "seed",
    ),
) -> None:
    mismatches: list[str] = []
    if saved.model != current.model:
        mismatches.append("model specification")
    if saved.preprocessing != current.preprocessing:
        mismatches.append("preprocessing specification")
    if saved.class_names != current.class_names:
        mismatches.append("class names")
    for key in immutable_training_keys:
        saved_value = saved.training.get(key)
        current_value = current.training.get(key)
        if saved_value is not None and current_value is not None and saved_value != current_value:
            mismatches.append(f"training.{key}: {saved_value!r} != {current_value!r}")
    for split_name in ("train", "val", "test"):
        key = f"{split_name}_manifest_sha256"
        saved_hash = saved.provenance.get(key)
        current_hash = current.provenance.get(key)
        if saved_hash and current_hash and saved_hash != current_hash:
            mismatches.append(key)
    if mismatches:
        raise ValueError("Resume configuration mismatch: " + "; ".join(mismatches))


def dump_metadata_json(metadata: ArtifactMetadata, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata.to_dict(), handle, indent=2, ensure_ascii=False)
