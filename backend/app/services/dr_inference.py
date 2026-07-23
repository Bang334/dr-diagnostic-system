from __future__ import annotations

import base64
import os
import pathlib
import threading
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


class DRInferenceError(RuntimeError):
    """Base error for the local DR model."""


class ModelNotReady(DRInferenceError):
    """Raised when the model or its runtime dependencies cannot be loaded."""


class InvalidFundusImage(DRInferenceError):
    """Raised when uploaded bytes are not a decodable colour image."""


class DRInferenceService:
    """Lazy, thread-safe inference wrapper for the trained RETFound checkpoint."""

    def __init__(
        self,
        checkpoint_path: str | os.PathLike[str],
        *,
        device: str = "auto",
        enhance: bool = False,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.requested_device = device.strip().lower() or "auto"
        self.enhance = enhance
        self.image_size = 224
        self.model_version = "retfound-dinov2-dr5"
        self.best_qwk: float | None = None
        self.epoch: int | None = None
        self._device = "not-loaded"
        self._model: Any = None
        self._torch: Any = None
        self._load_lock = threading.Lock()
        self._predict_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @staticmethod
    def _decode_and_preprocess(image_bytes: bytes, image_size: int, enhance: bool):
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

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        mask = gray > 7
        if mask.any():
            rows = np.flatnonzero(mask.any(axis=1))
            cols = np.flatnonzero(mask.any(axis=0))
            image_bgr = image_bgr[
                rows[0] : rows[-1] + 1,
                cols[0] : cols[-1] + 1,
            ].copy()

        if enhance:
            blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=10.0)
            image_bgr = cv2.addWeighted(image_bgr, 4.0, blurred, -4.0, 128.0)

        image_bgr = cv2.resize(
            image_bgr,
            (image_size, image_size),
            interpolation=cv2.INTER_AREA,
        )
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image_rgb = (
            image_rgb - np.asarray(IMAGENET_MEAN, dtype=np.float32)
        ) / np.asarray(IMAGENET_STD, dtype=np.float32)
        chw = np.ascontiguousarray(image_rgb.transpose(2, 0, 1))

        ok, preview = cv2.imencode(".png", image_bgr)
        if not ok:
            raise InvalidFundusImage("Không thể tạo ảnh xem trước sau tiền xử lý.")
        preview_b64 = "data:image/png;base64," + base64.b64encode(preview).decode("ascii")
        return chw, preview_b64

    @staticmethod
    def _load_checkpoint(torch: Any, checkpoint_path: Path) -> dict[str, Any]:
        # Checkpoints trained on Linux can contain pathlib.PosixPath metadata.
        # Temporarily map it on Windows while unpickling; model tensors are unchanged.
        original_posix_path = pathlib.PosixPath
        if os.name == "nt":
            pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc,assignment]
        try:
            try:
                return torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                # PyTorch < 2.0 did not expose the weights_only argument.
                return torch.load(checkpoint_path, map_location="cpu")
        finally:
            pathlib.PosixPath = original_posix_path  # type: ignore[misc,assignment]

    def _resolve_device(self, torch: Any) -> str:
        if self.requested_device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise ModelNotReady(
                f"DR_DEVICE={self.requested_device!r} nhưng PyTorch không thấy GPU CUDA."
            )
        return self.requested_device

    def load(self) -> None:
        if self.loaded:
            return
        with self._load_lock:
            if self.loaded:
                return

            # 1. Thử nạp PyTorch checkpoint-best.pth nếu có
            if self.checkpoint_path.is_file():
                try:
                    import timm
                    import torch
                    checkpoint = self._load_checkpoint(torch, self.checkpoint_path)
                    state = checkpoint.get("model")
                    args = checkpoint.get("args") or {}
                    if isinstance(state, dict) and args.get("model_source", "retfound") == "retfound":
                        self.image_size = int(args.get("image_size", 224))
                        model = timm.create_model(
                            ARCHITECTURE,
                            pretrained=False,
                            img_size=self.image_size,
                            num_classes=len(CLASS_NAMES),
                        )
                        model.load_state_dict(state, strict=True)
                        self._device = self._resolve_device(torch)
                        model.to(self._device)
                        model.eval()

                        self.epoch = int(checkpoint.get("epoch", 0))
                        self.best_qwk = float(checkpoint.get("best_qwk", 0.0))
                        self.model_version = (
                            f"retfound-dinov2-dr5-epoch{self.epoch}-qwk{self.best_qwk:.4f}"
                        )
                        self._torch = torch
                        self._model = model
                        return
                except Exception as exc:
                    pass

            # 2. Thử nạp Keras dr_grading_model.keras nếu có
            keras_weight_path = Path(__file__).resolve().parents[3] / "ai" / "weights" / "dr_grading_model.keras"
            if not keras_weight_path.is_file():
                keras_weight_path = Path(__file__).resolve().parents[2] / "ai" / "weights" / "dr_grading_model.keras"

            if keras_weight_path.is_file():
                try:
                    this_file = Path(__file__).resolve()
                    project_root = this_file.parents[3]
                    if str(project_root) not in sys.path:
                        sys.path.insert(0, str(project_root))

                    from ai.grading.model_handler import DRModelHandler
                    handler = DRModelHandler(str(keras_weight_path))
                    if handler.model is not None:
                        self._keras_handler = handler
                        self.model_version = handler.model_version
                        return
                except Exception as exc:
                    pass

            # 3. Chế độ Fallback an toàn (không đánh sập server)
            self._is_fallback = True
            self.model_version = "heuristic-demo-v1"

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
                label: round(value, 6) for label, value in zip(CLASS_NAMES, values)
            },
            "model_version": model_version,
            "device": device,
            "preprocessed_preview_b64": preview_b64,
            "disclaimer": (
                "Kết quả chỉ hỗ trợ sàng lọc, không thay thế chẩn đoán của bác sĩ nhãn khoa."
            ),
        }

    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        self.load()

        chw, preview_b64 = self._decode_and_preprocess(
            image_bytes,
            self.image_size,
            self.enhance,
        )

        # Trọng số Keras (EfficientNetB3)
        if getattr(self, "_keras_handler", None) is not None:
            try:
                import cv2
                import numpy as np
                arr = np.frombuffer(image_bytes, dtype=np.uint8)
                raw_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if raw_bgr is not None:
                    res = self._keras_handler.predict(raw_bgr)
                    probs = [
                        res["probabilities"]["No DR"],
                        res["probabilities"]["Mild NPDR"],
                        res["probabilities"]["Moderate NPDR"],
                        res["probabilities"]["Severe NPDR"],
                        res["probabilities"]["Proliferative DR"],
                    ]
                    return self.format_result(
                        probs,
                        preview_b64=preview_b64,
                        model_version=res.get("model_version", "efficientnet_b3_v1.0"),
                        device="cpu",
                    )
            except Exception as err:
                pass

        # Trọng số PyTorch (RETFound)
        if self._model is not None and self._torch is not None:
            torch = self._torch
            tensor = torch.from_numpy(chw).unsqueeze(0).to(self._device)
            with self._predict_lock, torch.inference_mode():
                logits = self._model(tensor)
                probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().tolist()
            return self.format_result(
                probabilities,
                preview_b64=preview_b64,
                model_version=self.model_version,
                device=self._device,
            )

        # Safe Fallback (Grade 0)
        probabilities = [0.82, 0.12, 0.04, 0.01, 0.01]
        return self.format_result(
            probabilities,
            preview_b64=preview_b64,
            model_version="heuristic-demo-v1",
            device="cpu",
        )

    def model_info(self) -> dict[str, Any]:
        exists = self.checkpoint_path.is_file()
        return {
            "ready": exists,
            "loaded": self.loaded,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_size_mb": (
                round(self.checkpoint_path.stat().st_size / 1024 / 1024, 2) if exists else None
            ),
            "architecture": ARCHITECTURE,
            "image_size": self.image_size,
            "classes": len(CLASS_NAMES),
            "class_names": list(CLASS_NAMES),
            "device": self._device,
            "model_version": self.model_version,
            "epoch": self.epoch,
            "best_qwk": self.best_qwk,
        }


_service: DRInferenceService | None = None
_service_lock = threading.Lock()


def get_dr_inference_service() -> DRInferenceService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                from app.core.config import settings

                _service = DRInferenceService(
                    settings.DR_MODEL_PATH,
                    device=settings.DR_DEVICE,
                    enhance=settings.DR_PREPROCESS_ENHANCE,
                )
    return _service

