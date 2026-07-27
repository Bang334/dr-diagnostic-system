from __future__ import annotations

import base64
import os
import pathlib
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence


CLASS_NAMES = (
    "No DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "Proliferative DR",
)
ARCHITECTURE = "vit_large_patch14_dinov2.lvd142m"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

GRADING_MODEL = "grading"
FEWSHOT_MODEL = "fewshot"
MODEL_LABELS = {
    GRADING_MODEL: "Grading gốc (checkpoint-best.pth)",
    FEWSHOT_MODEL: "Few-shot DeepDRiD (best-fewshot.pth)",
}
MODEL_ALIASES = {
    GRADING_MODEL: GRADING_MODEL,
    "base": GRADING_MODEL,
    "baseline": GRADING_MODEL,
    "checkpoint-best": GRADING_MODEL,
    "checkpoint-best.pth": GRADING_MODEL,
    FEWSHOT_MODEL: FEWSHOT_MODEL,
    "few-shot": FEWSHOT_MODEL,
    "best-fewshot": FEWSHOT_MODEL,
    "best-fewshot.pth": FEWSHOT_MODEL,
}
PREPROCESSING_RECIPES = {"rgb_crop", "green", "clahe", "ben_graham"}


class DRInferenceError(RuntimeError):
    """Base error for the local DR model."""


class ModelNotReady(DRInferenceError):
    """Raised when the selected model cannot be loaded."""


class InvalidFundusImage(DRInferenceError):
    """Raised when uploaded bytes are not a decodable colour image."""


class InvalidModelSelection(DRInferenceError):
    """Raised when a request names a model outside the configured allow-list."""


def normalize_model_key(model_key: str | None) -> str:
    value = (model_key or "").strip().lower()
    normalized = MODEL_ALIASES.get(value)
    if normalized is None:
        choices = ", ".join(MODEL_LABELS)
        raise InvalidModelSelection(
            f"Model nhận diện không hợp lệ: {model_key!r}. Chọn một trong: {choices}."
        )
    return normalized


def _metadata_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


class DRInferenceService:
    """Lazy, thread-safe inference for a grading or few-shot RETFound checkpoint."""

    def __init__(
        self,
        checkpoint_path: str | os.PathLike[str],
        *,
        model_key: str = GRADING_MODEL,
        device: str = "auto",
        enhance: bool = False,
    ) -> None:
        self.model_key = normalize_model_key(model_key)
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.requested_device = device.strip().lower() or "auto"
        self.configured_enhance = enhance
        self.image_size = 224
        self.preprocessing_recipe = "legacy"
        self.model_version = f"{self.model_key}-not-loaded"
        self.best_qwk: float | None = None
        self.best_selection_loss: float | None = None
        self.epoch: int | None = None
        self.checkpoint_kind = "not-loaded"
        self._device = "not-loaded"
        self._model: Any = None
        self._projection: Any = None
        self._prototypes: Any = None
        self._class_ids: Any = None
        self._temperature = 1.0
        self._torch: Any = None
        self._load_lock = threading.Lock()
        self._predict_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @staticmethod
    def _decode_image(image_bytes: bytes):
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise ModelNotReady(
                "Thiếu thư viện xử lý ảnh. Hãy cài requirements.txt của backend."
            ) from exc

        if not image_bytes:
            raise InvalidFundusImage("Tệp ảnh rỗng.")
        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise InvalidFundusImage("Không thể giải mã ảnh PNG/JPG/JPEG.")
        return image_bgr

    @staticmethod
    def _legacy_crop(image_bgr):
        import cv2
        import numpy as np

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        mask = gray > 7
        if not mask.any():
            return image_bgr.copy()
        rows = np.flatnonzero(mask.any(axis=1))
        columns = np.flatnonzero(mask.any(axis=0))
        return image_bgr[
            rows[0] : rows[-1] + 1,
            columns[0] : columns[-1] + 1,
        ].copy()

    @classmethod
    def _recipe_crop(cls, image_bgr):
        import cv2
        import numpy as np

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        mask = gray > max(5, int(np.percentile(gray, 5)))
        points = cv2.findNonZero(mask.astype(np.uint8))
        if points is None:
            return cls._legacy_crop(image_bgr)
        x, y, width, height = cv2.boundingRect(points)
        return image_bgr[y : y + height, x : x + width].copy()

    @classmethod
    def _preprocess_image(
        cls,
        image_bgr,
        *,
        image_size: int,
        recipe: str,
        enhance: bool,
    ):
        import cv2
        import numpy as np

        if recipe in PREPROCESSING_RECIPES:
            processed = cls._recipe_crop(image_bgr)
        else:
            processed = cls._legacy_crop(image_bgr)
        processed = cv2.resize(
            processed,
            (image_size, image_size),
            interpolation=cv2.INTER_AREA,
        )

        if recipe == "green":
            processed = cv2.cvtColor(processed[:, :, 1], cv2.COLOR_GRAY2BGR)
        elif recipe == "clahe":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed = cv2.cvtColor(
                clahe.apply(processed[:, :, 1]),
                cv2.COLOR_GRAY2BGR,
            )
        elif recipe == "ben_graham" or enhance:
            blurred = cv2.GaussianBlur(processed, (0, 0), sigmaX=10.0)
            processed = cv2.addWeighted(processed, 4.0, blurred, -4.0, 128.0)

        image_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_rgb /= 255.0
        image_rgb = (
            image_rgb - np.asarray(IMAGENET_MEAN, dtype=np.float32)
        ) / np.asarray(IMAGENET_STD, dtype=np.float32)
        chw = np.ascontiguousarray(image_rgb.transpose(2, 0, 1))

        ok, preview = cv2.imencode(".png", processed)
        if not ok:
            raise InvalidFundusImage("Không thể tạo ảnh xem trước sau tiền xử lý.")
        preview_b64 = "data:image/png;base64," + base64.b64encode(preview).decode(
            "ascii"
        )
        return chw, preview_b64

    @staticmethod
    def _load_checkpoint(torch: Any, checkpoint_path: Path) -> dict[str, Any]:
        original_posix_path = pathlib.PosixPath
        if os.name == "nt":
            pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc,assignment]
        try:
            try:
                state = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=False,
                    mmap=True,
                )
            except TypeError:
                state = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=False,
                )
        finally:
            pathlib.PosixPath = original_posix_path  # type: ignore[misc,assignment]
        if not isinstance(state, dict):
            raise ModelNotReady("Checkpoint phải chứa một dictionary state.")
        return state

    def _resolve_device(self, torch: Any) -> str:
        if self.requested_device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise ModelNotReady(
                f"DR_DEVICE={self.requested_device!r} nhưng PyTorch không thấy GPU CUDA."
            )
        return self.requested_device

    @staticmethod
    def _create_encoder(timm: Any, args: Mapping[str, Any]):
        image_size = int(args.get("image_size", 224))
        loss_name = str(args.get("loss", "ce"))
        if loss_name != "ce":
            raise ModelNotReady(
                f"Backend chỉ hỗ trợ checkpoint cross-entropy 5 lớp, nhận loss={loss_name!r}."
            )
        model_source = str(args.get("model_source", "retfound"))
        if model_source == "retfound":
            return timm.create_model(
                ARCHITECTURE,
                pretrained=False,
                img_size=image_size,
                num_classes=len(CLASS_NAMES),
            )
        if model_source == "timm":
            model_name = args.get("model_name")
            if not model_name:
                raise ModelNotReady("Checkpoint timm không ghi model_name.")
            return timm.create_model(
                str(model_name),
                pretrained=False,
                num_classes=len(CLASS_NAMES),
            )
        raise ModelNotReady(f"model_source không được hỗ trợ: {model_source!r}.")

    def _set_preprocessing(self, args: Mapping[str, Any]) -> None:
        self.image_size = int(args.get("image_size", 224))
        saved_recipe = args.get("preprocessing")
        self.preprocessing_recipe = (
            str(saved_recipe) if saved_recipe in PREPROCESSING_RECIPES else "legacy"
        )
        self.configured_enhance = bool(args.get("enhance", False)) or (
            self.configured_enhance
        )

    def _load_grading_checkpoint(
        self,
        *,
        checkpoint: Mapping[str, Any],
        timm: Any,
        torch: Any,
    ) -> None:
        state = checkpoint.get("model")
        args = _metadata_dict(checkpoint.get("args"))
        if not isinstance(state, Mapping) or not args:
            raise ModelNotReady(
                "Checkpoint grading phải chứa cả trường 'model' và 'args'."
            )
        model = self._create_encoder(timm, args)
        model.load_state_dict(state, strict=True)
        self._device = self._resolve_device(torch)
        model.to(self._device)
        model.eval()
        self._set_preprocessing(args)

        self.epoch = int(checkpoint.get("epoch", 0))
        self.best_qwk = float(checkpoint.get("best_qwk", 0.0))
        self.model_version = (
            f"grading-retfound-epoch{self.epoch}-qwk{self.best_qwk:.4f}"
        )
        self.checkpoint_kind = "grading_classifier"
        self._torch = torch
        self._model = model

    def _load_fewshot_checkpoint(
        self,
        *,
        checkpoint: Mapping[str, Any],
        timm: Any,
        torch: Any,
    ) -> None:
        if checkpoint.get("method") != "fixed_support_target_domain_protonet":
            raise ModelNotReady("Checkpoint không phải fixed-support few-shot ProtoNet.")
        base_args = _metadata_dict(checkpoint.get("base_model_args"))
        encoder_state = checkpoint.get("encoder_model")
        projection_state = checkpoint.get("projection")
        prototypes = checkpoint.get("prototypes")
        class_ids = checkpoint.get("class_ids")
        if (
            not base_args
            or not isinstance(encoder_state, Mapping)
            or not isinstance(projection_state, Mapping)
            or not hasattr(prototypes, "shape")
            or not hasattr(class_ids, "shape")
        ):
            raise ModelNotReady("Checkpoint few-shot thiếu encoder/projection/prototype.")

        encoder = self._create_encoder(timm, base_args)
        encoder.load_state_dict(encoder_state, strict=True)
        feature_dim = int(getattr(encoder, "num_features", 0))
        embedding_dim = int(checkpoint.get("embedding_dim", feature_dim))
        if feature_dim <= 0 or embedding_dim <= 0:
            raise ModelNotReady("Checkpoint few-shot có embedding dimension không hợp lệ.")

        if projection_state:
            projection = torch.nn.Linear(feature_dim, embedding_dim, bias=False)
            projection.load_state_dict(projection_state, strict=True)
        else:
            if feature_dim != embedding_dim:
                raise ModelNotReady(
                    "Projection rỗng nhưng embedding dimension không khớp encoder."
                )
            projection = torch.nn.Identity()

        class_values = [int(value) for value in class_ids.detach().cpu().tolist()]
        if sorted(class_values) != list(range(len(CLASS_NAMES))):
            raise ModelNotReady(
                f"Prototype few-shot phải phủ đủ grade 0..4, nhận {class_values}."
            )
        if tuple(prototypes.shape) != (len(class_values), embedding_dim):
            raise ModelNotReady(
                "Kích thước prototype không khớp class_ids và embedding dimension."
            )

        self._device = self._resolve_device(torch)
        encoder.to(self._device)
        encoder.eval()
        projection.to(self._device)
        projection.eval()
        self._set_preprocessing(base_args)

        self.epoch = int(checkpoint.get("epoch", 0))
        self.best_selection_loss = float(
            checkpoint.get("best_selection_loss", 0.0)
        )
        self.model_version = (
            "fewshot-deepdrid-protonet-"
            f"epoch{self.epoch}-selection{self.best_selection_loss:.4f}"
        )
        self.checkpoint_kind = "fewshot_protonet"
        self._temperature = float(checkpoint.get("temperature", 0.1))
        if self._temperature <= 0:
            raise ModelNotReady("Temperature few-shot phải lớn hơn 0.")
        self._torch = torch
        self._model = encoder
        self._projection = projection
        self._prototypes = prototypes.detach().to(self._device)
        self._class_ids = class_ids.detach().to(
            self._device,
            dtype=torch.long,
        )

    def load(self) -> None:
        if self.loaded:
            return
        with self._load_lock:
            if self.loaded:
                return
            if not self.checkpoint_path.is_file():
                raise ModelNotReady(
                    f"Không tìm thấy checkpoint {MODEL_LABELS[self.model_key]}: "
                    f"{self.checkpoint_path}"
                )
            try:
                import timm
                import torch

                checkpoint = self._load_checkpoint(torch, self.checkpoint_path)
                if (
                    checkpoint.get("method")
                    == "fixed_support_target_domain_protonet"
                ):
                    self._load_fewshot_checkpoint(
                        checkpoint=checkpoint,
                        timm=timm,
                        torch=torch,
                    )
                else:
                    self._load_grading_checkpoint(
                        checkpoint=checkpoint,
                        timm=timm,
                        torch=torch,
                    )
            except DRInferenceError:
                raise
            except Exception as exc:
                raise ModelNotReady(
                    f"Không thể nạp {MODEL_LABELS[self.model_key]}: {exc}"
                ) from exc

    def unload(self) -> None:
        torch = self._torch
        self._model = None
        self._projection = None
        self._prototypes = None
        self._class_ids = None
        self._torch = None
        self._device = "not-loaded"
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def format_result(
        probabilities: Sequence[float],
        *,
        preview_b64: str,
        model_version: str,
        device: str,
    ) -> dict[str, Any]:
        if len(probabilities) != len(CLASS_NAMES):
            raise ValueError("Model phải trả đúng 5 xác suất ICDR.")
        values = [float(value) for value in probabilities]
        total = sum(values)
        if total <= 0:
            raise ValueError("Model trả xác suất không hợp lệ.")
        values = [value / total for value in values]
        grade = max(range(len(values)), key=values.__getitem__)
        return {
            "dr_grade": grade,
            "dr_label": CLASS_NAMES[grade],
            "confidence": round(values[grade], 6),
            "probabilities": {
                label: round(value, 6)
                for label, value in zip(CLASS_NAMES, values)
            },
            "model_version": model_version,
            "device": device,
            "preprocessed_preview_b64": preview_b64,
            "disclaimer": (
                "Kết quả chỉ hỗ trợ sàng lọc, không thay thế chẩn đoán "
                "của bác sĩ nhãn khoa."
            ),
        }

    def _predict_grading(self, tensor) -> list[float]:
        torch = self._torch
        with self._predict_lock, torch.inference_mode():
            logits = self._model(tensor)
            return torch.softmax(logits, dim=1)[0].detach().cpu().tolist()

    def _predict_fewshot(self, tensor) -> list[float]:
        torch = self._torch
        with self._predict_lock, torch.inference_mode():
            features = self._model.forward_features(tensor)
            embedding = self._model.forward_head(features, pre_logits=True)
            if embedding.ndim != 2:
                embedding = embedding.flatten(1)
            embedding = torch.nn.functional.normalize(
                self._projection(embedding),
                dim=1,
            )
            distances = torch.cdist(embedding, self._prototypes).pow(2)
            local_probabilities = torch.softmax(
                -distances / self._temperature,
                dim=1,
            )[0]
            probabilities = torch.zeros(
                len(CLASS_NAMES),
                device=self._device,
                dtype=local_probabilities.dtype,
            )
            probabilities.scatter_(0, self._class_ids, local_probabilities)
            return probabilities.detach().cpu().tolist()

    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        # Reject malformed uploads before spending time loading a 1.2+ GB model.
        image_bgr = self._decode_image(image_bytes)
        self.load()
        chw, preview_b64 = self._preprocess_image(
            image_bgr,
            image_size=self.image_size,
            recipe=self.preprocessing_recipe,
            enhance=self.configured_enhance,
        )
        torch = self._torch
        tensor = torch.from_numpy(chw).unsqueeze(0).to(self._device)
        probabilities = (
            self._predict_fewshot(tensor)
            if self.checkpoint_kind == "fewshot_protonet"
            else self._predict_grading(tensor)
        )
        return self.format_result(
            probabilities,
            preview_b64=preview_b64,
            model_version=self.model_version,
            device=self._device,
        )

    def model_info(self) -> dict[str, Any]:
        exists = self.checkpoint_path.is_file()
        return {
            "id": self.model_key,
            "label": MODEL_LABELS[self.model_key],
            "ready": exists,
            "loaded": self.loaded,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_size_mb": (
                round(self.checkpoint_path.stat().st_size / 1024 / 1024, 2)
                if exists
                else None
            ),
            "checkpoint_kind": self.checkpoint_kind,
            "architecture": ARCHITECTURE,
            "image_size": self.image_size,
            "preprocessing": self.preprocessing_recipe,
            "classes": len(CLASS_NAMES),
            "class_names": list(CLASS_NAMES),
            "device": self._device,
            "model_version": self.model_version,
            "epoch": self.epoch,
            "best_qwk": self.best_qwk,
            "best_selection_loss": self.best_selection_loss,
        }


class _DRInferenceRegistry:
    """Keep one large grading model resident and switch it safely on demand."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active_key: str | None = None
        self._service: DRInferenceService | None = None

    @staticmethod
    def _settings():
        from app.core.config import settings

        return settings

    def _new_service(self, model_key: str) -> DRInferenceService:
        settings = self._settings()
        paths = {
            GRADING_MODEL: settings.DR_MODEL_PATH,
            FEWSHOT_MODEL: settings.DR_FEWSHOT_MODEL_PATH,
        }
        return DRInferenceService(
            paths[model_key],
            model_key=model_key,
            device=settings.DR_DEVICE,
            enhance=settings.DR_PREPROCESS_ENHANCE,
        )

    def get(self, model_key: str | None = None) -> DRInferenceService:
        settings = self._settings()
        normalized = normalize_model_key(model_key or settings.DR_DEFAULT_MODEL)
        with self._lock:
            if self._service is None or self._active_key != normalized:
                if self._service is not None:
                    self._service.unload()
                self._service = self._new_service(normalized)
                self._active_key = normalized
            return self._service

    def predict(self, image_bytes: bytes, model_key: str | None = None):
        # Hold the registry lock through inference so another request cannot evict
        # the selected model between load and prediction.
        with self._lock:
            return self.get(model_key).predict(image_bytes)


_registry = _DRInferenceRegistry()


def get_dr_inference_service(
    model_key: str | None = None,
) -> DRInferenceService:
    return _registry.get(model_key)


def predict_with_dr_model(
    image_bytes: bytes,
    model_key: str | None = None,
) -> dict[str, Any]:
    return _registry.predict(image_bytes, model_key)


def available_dr_models() -> dict[str, Any]:
    from app.core.config import settings

    default_model = normalize_model_key(settings.DR_DEFAULT_MODEL)
    paths = {
        GRADING_MODEL: Path(settings.DR_MODEL_PATH).expanduser().resolve(),
        FEWSHOT_MODEL: Path(settings.DR_FEWSHOT_MODEL_PATH).expanduser().resolve(),
    }
    return {
        "default": default_model,
        "models": [
            {
                "id": model_key,
                "label": MODEL_LABELS[model_key],
                "ready": path.is_file(),
                "checkpoint": path.name,
                "size_mb": (
                    round(path.stat().st_size / 1024 / 1024, 2)
                    if path.is_file()
                    else None
                ),
            }
            for model_key, path in paths.items()
        ],
    }
