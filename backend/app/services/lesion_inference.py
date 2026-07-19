from __future__ import annotations

import os
import sys
import threading
import pathlib
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


# ─────────────────────────────────────────────────────────────────────────────
# Màu sắc chuẩn lâm sàng cho overlay tổn thương
# Theo chuẩn IDRiD Challenge và tài liệu y khoa quốc tế (BGR format cho OpenCV)
#   MA (Vi phình mạch)  = ĐỎ    → chấm đỏ nhỏ như biểu hiện lâm sàng thực
#   HE (Xuất huyết)     = CAM   → phân biệt với MA (cũng đỏ) bằng màu cam
#   EX (Tiết cứng)      = VÀNG  → khớp với màu vàng lắng đọng lipid lâm sàng
# ─────────────────────────────────────────────────────────────────────────────
LESION_COLORS_BGR = {
    "MA": (0,   0,   255),   # BGR: Đỏ thuần
    "HE": (0,   165, 255),   # BGR: Cam
    "EX": (0,   255, 255),   # BGR: Vàng
}

LESION_META = {
    "MA": {"label": "Vi phình mạch (MA)", "confidence": 0.4625},
    "HE": {"label": "Xuất huyết (HE)",    "confidence": 0.4960},
    "EX": {"label": "Tiết cứng (EX)",     "confidence": 0.6995},
}

MODEL_VERSION = "attention-unet-resnet34-50-idrid-v1"


class LesionInferenceError(RuntimeError):
    """Lỗi cơ sở cho lesion segmentation service."""


class LesionModelNotReady(LesionInferenceError):
    """Raised khi checkpoint không tìm thấy hoặc thiếu thư viện."""


# ─────────────────────────────────────────────────────────────────────────────
# Hàm tiền xử lý ảnh – tái hiện pipeline từ segmentation_lesion_v1
# ─────────────────────────────────────────────────────────────────────────────

def _letterbox(image: np.ndarray, target_size: int = 512) -> tuple[np.ndarray, dict]:
    """
    Adaptive Letterboxing: co dãn ảnh theo cạnh dài nhất, thêm viền đen
    để đưa về hình vuông target_size×target_size mà KHÔNG bóp méo hình học.
    Trả về (ảnh đã padding, metadata để restore về kích thước gốc).
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Tạo canvas đen
    canvas = np.zeros((target_size, target_size, image.shape[2]), dtype=resized.dtype)
    pad_top  = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    meta = {"scale": scale, "pad_top": pad_top, "pad_left": pad_left,
            "new_h": new_h, "new_w": new_w, "orig_h": h, "orig_w": w}
    return canvas, meta


def _extract_roi(image: np.ndarray) -> tuple[np.ndarray, tuple]:
    """
    Cắt vùng quan tâm (ROI): loại bỏ viền đen xung quanh võng mạc.
    Trả về (ảnh ROI, bbox=(y_min, y_max, x_min, x_max)).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    rows = np.any(thresh, axis=1)
    cols = np.any(thresh, axis=0)
    if not rows.any() or not cols.any():
        h, w = image.shape[:2]
        return image, (0, h, 0, w)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    # Thêm padding nhỏ 2%
    pad_y = int((y_max - y_min) * 0.02)
    pad_x = int((x_max - x_min) * 0.02)
    y_min = max(0, y_min - pad_y)
    y_max = min(image.shape[0], y_max + pad_y)
    x_min = max(0, x_min - pad_x)
    x_max = min(image.shape[1], x_max + pad_x)
    return image[y_min:y_max, x_min:x_max], (y_min, y_max, x_min, x_max)


def _clahe_green_channel(image_bgr: np.ndarray) -> np.ndarray:
    """
    Trích xuất kênh Green + CLAHE: kênh xanh lá hiển thị tổn thương
    mạch máu rõ nhất và là đầu vào 1-channel của model.
    """
    green = image_bgr[:, :, 1]  # Green channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)
    # Chuẩn hóa [0,1]
    return enhanced.astype(np.float32) / 255.0


def _detect_optic_disc(image_bgr: np.ndarray, radius_ratio: float = 0.12) -> np.ndarray:
    """
    Phát hiện đĩa thị heuristic (dùng cho post-processing EX).
    Trả về mặt nạ nhị phân: 1 tại vùng đĩa thị.
    """
    h, w = image_bgr.shape[:2]
    scale = 0.25
    small = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(small, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    k = max(3, int(w * scale * 0.05) | 1)
    blurred = cv2.GaussianBlur(small, (k, k), 0)
    _, _, _, max_loc = cv2.minMaxLoc(blurred)
    cx = int(max_loc[0] / scale)
    cy = int(max_loc[1] / scale)
    radius = int(w * radius_ratio)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 1, -1)
    return mask


def _post_process(
    prob_map: np.ndarray,
    threshold: float,
    min_area: int,
    raw_image: np.ndarray | None = None,
    use_od_mask: bool = False,
) -> np.ndarray:
    """Phân ngưỡng + morphology + lọc diện tích + che đĩa thị (EX)."""
    binary = (prob_map >= threshold).astype(np.uint8)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)
    # Lọc connected components nhỏ
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    filtered = np.zeros_like(binary)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = 1
    # Che đĩa thị cho EX
    if use_od_mask and raw_image is not None:
        od = _detect_optic_disc(raw_image)
        filtered[od == 1] = 0
    return filtered


def _sliding_window_predict(
    model: Any,
    image_float: np.ndarray,   # (H, W) float32
    device: Any,
    patch_size: int = 512,
    stride: int = 448,
) -> np.ndarray:
    """
    Suy luận trượt cửa sổ với ghép Gaussian.
    image_float: ảnh Green channel CLAHE đã chuẩn hóa [0,1].
    """
    import torch
    h, w = image_float.shape
    acc = np.zeros((h, w), dtype=np.float32)
    wmap = np.zeros((h, w), dtype=np.float32)

    # Tạo cửa sổ Gaussian
    gp = np.zeros((patch_size, patch_size), dtype=np.float32)
    gp[patch_size // 2, patch_size // 2] = 1.0
    gp = gaussian_filter(gp, sigma=patch_size / 6)
    gp /= gp.max()

    model.eval()
    with torch.no_grad():
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = image_float[y:y + patch_size, x:x + patch_size]
                t = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                prob = torch.sigmoid(model(t)).cpu().squeeze().numpy()
                acc[y:y + patch_size, x:x + patch_size] += prob * gp
                wmap[y:y + patch_size, x:x + patch_size] += gp
        # Xử lý rìa phải/dưới
        if (h - patch_size) % stride != 0 or (w - patch_size) % stride != 0:
            yl = max(0, h - patch_size)
            xl = max(0, w - patch_size)
            patch = image_float[yl:yl + patch_size, xl:xl + patch_size]
            if patch.shape == (patch_size, patch_size):
                t = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                prob = torch.sigmoid(model(t)).cpu().squeeze().numpy()
                acc[yl:yl + patch_size, xl:xl + patch_size] += prob * gp
                wmap[yl:yl + patch_size, xl:xl + patch_size] += gp

    result = np.zeros_like(acc)
    valid = wmap > 0
    result[valid] = acc[valid] / wmap[valid]
    return result


def _predict_with_tta(model, image_float, device, patch_size, stride):
    """TTA: gốc + flip ngang + flip dọc → trung bình."""
    p0 = _sliding_window_predict(model, image_float, device, patch_size, stride)
    p1 = cv2.flip(_sliding_window_predict(model, cv2.flip(image_float, 1), device, patch_size, stride), 1)
    p2 = cv2.flip(_sliding_window_predict(model, cv2.flip(image_float, 0), device, patch_size, stride), 0)
    return (p0 + p1 + p2) / 3.0


def _create_overlay(raw_bgr: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
    """
    Tạo ảnh overlay: chồng mask màu bán trong suốt lên ảnh gốc.
    masks: {"MA": binary_mask, "HE": binary_mask, "EX": binary_mask}
    Màu chuẩn lâm sàng: MA=Đỏ, HE=Cam, EX=Vàng
    """
    overlay = raw_bgr.copy().astype(np.float32)
    alpha = 0.45  # Độ trong suốt 45% — đủ thấy ảnh gốc, đủ nổi bật tổn thương

    for lesion_key, binary_mask in masks.items():
        if binary_mask is None or not binary_mask.any():
            continue
        color = LESION_COLORS_BGR[lesion_key]
        color_layer = np.zeros_like(raw_bgr, dtype=np.float32)
        color_layer[binary_mask == 1] = color
        mask_bool = binary_mask.astype(bool)
        overlay[mask_bool] = (
            overlay[mask_bool] * (1 - alpha) + color_layer[mask_bool] * alpha
        )
    return np.clip(overlay, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Service class chính – thread-safe, lazy load
# ─────────────────────────────────────────────────────────────────────────────

class LesionInferenceService:
    """
    Lazy, thread-safe service nạp 3 mô hình Attention U-Net (MA/HE/EX)
    và cung cấp endpoint predict() cho FastAPI backend.
    """

    def __init__(
        self,
        ckpt_ma: str | os.PathLike,
        ckpt_he: str | os.PathLike,
        ckpt_ex: str | os.PathLike,
        upload_dir: str | os.PathLike,
        device: str = "auto",
    ) -> None:
        self.ckpt_paths = {
            "MA": Path(ckpt_ma).expanduser().resolve(),
            "HE": Path(ckpt_he).expanduser().resolve(),
            "EX": Path(ckpt_ex).expanduser().resolve(),
        }
        self.upload_dir = Path(upload_dir).expanduser().resolve()
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.requested_device = device.strip().lower() or "auto"
        self._models: dict[str, Any] = {}
        self._device: Any = None
        self._torch: Any = None
        self._load_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return len(self._models) == 3

    def _resolve_device(self, torch: Any) -> Any:
        if self.requested_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.requested_device)

    def _add_segmentation_lesion_to_path(self) -> None:
        """
        Thêm thư mục segmentation_lesion_v1 vào sys.path để import
        các module RetinalLesionLightningModule, RetinalConfig, get_segmentation_model.
        """
        # Tìm đường dẫn tương đối từ vị trí file này
        this_file = Path(__file__).resolve()
        # Đường dẫn: backend/app/services/lesion_inference.py → lên 3 cấp → dr-diagnostic-system
        # segmentation_lesion_v1 nằm cùng cấp với dr-diagnostic-system
        project_root = this_file.parents[3]  # D:\dr-diagnostic-system
        seg_path = project_root.parent / "segmentation_lesion_v1"
        if seg_path.exists() and str(seg_path) not in sys.path:
            sys.path.insert(0, str(seg_path))

    def load(self) -> None:
        """Nạp 3 checkpoint vào bộ nhớ (thread-safe, chỉ nạp 1 lần)."""
        if self.loaded:
            return
        with self._load_lock:
            if self.loaded:
                return
            # Kiểm tra checkpoint files tồn tại
            for key, path in self.ckpt_paths.items():
                if not path.is_file():
                    raise LesionModelNotReady(
                        f"Không tìm thấy checkpoint {key} tại: {path}"
                    )
            # Import thư viện
            try:
                import torch
                import pytorch_lightning as pl
            except ImportError as exc:
                raise LesionModelNotReady(
                    "Thiếu torch/pytorch-lightning. Hãy cài requirements.txt."
                ) from exc
            try:
                import segmentation_models_pytorch  # noqa: F401
            except ImportError as exc:
                raise LesionModelNotReady(
                    "Thiếu segmentation-models-pytorch. Chạy: pip install segmentation-models-pytorch"
                ) from exc

            # Thêm segmentation_lesion_v1 vào sys.path để import modules
            self._add_segmentation_lesion_to_path()

            try:
                from configs.config import RetinalConfig
                from trainer.lightning_trainer import RetinalLesionLightningModule
            except ImportError as exc:
                raise LesionModelNotReady(
                    f"Không thể import module từ segmentation_lesion_v1: {exc}"
                ) from exc

            device = self._resolve_device(torch)
            models = {}
            for lesion_type, ckpt_path in self.ckpt_paths.items():
                config = RetinalConfig(lesion_type=lesion_type, dataset_name="idrid")
                # Xử lý PosixPath trên Windows khi load checkpoint được train trên Linux
                original_posix = pathlib.PosixPath
                if os.name == "nt":
                    pathlib.PosixPath = pathlib.WindowsPath  # type: ignore
                try:
                    module = RetinalLesionLightningModule.load_from_checkpoint(
                        str(ckpt_path), config=config, weights_only=False
                    )
                finally:
                    pathlib.PosixPath = original_posix  # type: ignore

                model = module.model.to(device)
                model.eval()
                models[lesion_type] = (model, config)

            self._torch = torch
            self._device = device
            self._models = models

    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        """
        Pipeline suy luận đầy đủ cho 1 ảnh võng mạc.

        Returns dict khớp với SegmentationResult schema của hệ thống:
        {
          "lesions": [{"key": "microaneurysm", "label": ..., "detected": bool,
                       "area_pct": float, "confidence": float}, ...],
          "lesion_mask_url": "/uploads/seg-overlay-<uuid>.png",
          "model_version": str,
          "status": "ok"
        }
        """
        self.load()

        # ── 1. Giải mã ảnh từ bytes ──────────────────────────────────────────
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        raw_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if raw_bgr is None:
            raise LesionInferenceError("Không thể giải mã ảnh PNG/JPEG.")
        orig_h, orig_w = raw_bgr.shape[:2]

        # ── 2. Cắt ROI (loại bỏ viền đen) ───────────────────────────────────
        roi_bgr, (ym, yM, xm, xM) = _extract_roi(raw_bgr)

        binary_masks: dict[str, np.ndarray] = {}
        lesion_results = []

        # ── 3. Suy luận từng model MA / HE / EX ─────────────────────────────
        for lesion_type, (model, config) in self._models.items():
            # 3a. Letterboxing để giữ tỷ lệ khung hình → không bóp méo hình học
            lb_img, lb_meta = _letterbox(roi_bgr, target_size=max(roi_bgr.shape[:2]))

            # 3b. CLAHE Green channel → float32 [0,1]
            img_float = _clahe_green_channel(lb_img)

            # 3c. Sliding Window + TTA → xác suất (H_lb, W_lb)
            prob_map = _predict_with_tta(
                model, img_float, self._device,
                patch_size=config.patch_size,
                stride=config.stride,
            )

            # 3d. Crop lại phần không padding từ letterbox
            pt, pl = lb_meta["pad_top"], lb_meta["pad_left"]
            nh, nw = lb_meta["new_h"], lb_meta["new_w"]
            prob_map = prob_map[pt:pt + nh, pl:pl + nw]

            # 3e. Resize prob_map về kích thước ROI gốc
            prob_map = cv2.resize(prob_map, (roi_bgr.shape[1], roi_bgr.shape[0]),
                                  interpolation=cv2.INTER_LINEAR)

            # 3f. Post-processing (threshold + morphology + OD mask)
            binary_roi = _post_process(
                prob_map,
                threshold=config.threshold,
                min_area=config.min_lesion_area,
                raw_image=roi_bgr,
                use_od_mask=config.use_optic_disc_mask,
            )

            # 3g. Stitch mask ROI ngược về kích thước ảnh gốc
            binary_full = np.zeros((orig_h, orig_w), dtype=np.uint8)
            binary_full[ym:yM, xm:xM] = binary_roi

            binary_masks[lesion_type] = binary_full

            # ── 4. Tính thống kê ─────────────────────────────────────────────
            detected = bool(binary_full.any())
            retinal_pixels = np.count_nonzero(
                cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY) > 10
            )
            lesion_pixels = int(binary_full.sum())
            area_pct = round(lesion_pixels / max(retinal_pixels, 1), 6)

            key_map = {"MA": "microaneurysm", "HE": "hemorrhage", "EX": "hard_exudate"}
            label_map = LESION_META
            lesion_results.append({
                "key":        key_map[lesion_type],
                "label":      label_map[lesion_type]["label"],
                "detected":   detected,
                "area_pct":   area_pct,
                "confidence": label_map[lesion_type]["confidence"],
            })

        # ── 5. Tạo ảnh Overlay (MA=Đỏ, HE=Cam, EX=Vàng) ────────────────────
        overlay_bgr = _create_overlay(raw_bgr, binary_masks)

        # ── 6. Lưu ảnh overlay vào thư mục uploads ──────────────────────────
        filename = f"seg-overlay-{uuid4().hex}.png"
        save_path = self.upload_dir / filename
        cv2.imwrite(str(save_path), overlay_bgr)
        mask_url = f"/uploads/{filename}"

        return {
            "lesions":         lesion_results,
            "lesion_mask_url": mask_url,
            "model_version":   MODEL_VERSION,
            "status":          "ok",
        }

    def model_info(self) -> dict[str, Any]:
        info = {"loaded": self.loaded, "model_version": MODEL_VERSION, "checkpoints": {}}
        for key, path in self.ckpt_paths.items():
            info["checkpoints"][key] = {
                "path":  str(path),
                "ready": path.is_file(),
                "size_mb": round(path.stat().st_size / 1024 / 1024, 1) if path.is_file() else None,
            }
        return info


# ─────────────────────────────────────────────────────────────────────────────
# Singleton factory (thread-safe)
# ─────────────────────────────────────────────────────────────────────────────

_service: LesionInferenceService | None = None
_service_lock = threading.Lock()


def get_lesion_inference_service() -> LesionInferenceService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                from app.core.config import BASE_DIR, settings
                _service = LesionInferenceService(
                    ckpt_ma=settings.LESION_MA_CHECKPOINT,
                    ckpt_he=settings.LESION_HE_CHECKPOINT,
                    ckpt_ex=settings.LESION_EX_CHECKPOINT,
                    upload_dir=BASE_DIR / "uploads",
                    device=settings.LESION_DEVICE,
                )
    return _service
