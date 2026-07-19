from __future__ import annotations

import asyncio
from io import BytesIO
from typing import Any, Dict

import httpx

from app.clinical.models import GradingResult, Lesion, SegmentationResult


class AIServiceUnavailable(RuntimeError):
    pass


class LocalGradingAdapter:
    """Run the bundled RETFound checkpoint without a second HTTP service."""

    async def predict(self, image_bytes: bytes, eye: str) -> GradingResult:
        from app.services.dr_inference import DRInferenceError, get_dr_inference_service

        loop = asyncio.get_running_loop()
        try:
            data = await loop.run_in_executor(
                None,
                get_dr_inference_service().predict,
                image_bytes,
            )
        except DRInferenceError as exc:
            raise AIServiceUnavailable(f"Local grading model failed: {exc}") from exc
        return GradingResult(
            dr_grade=int(data["dr_grade"]),
            dr_label=str(data["dr_label"]),
            confidence=float(data["confidence"]),
            probabilities=data["probabilities"],
            model_version=str(data["model_version"]),
        )


class UnavailableSegmentationAdapter:
    """Explicit safe fallback when no lesion-segmentation model is configured."""

    async def predict(self, image_bytes: bytes, eye: str) -> SegmentationResult:
        return SegmentationResult(
            lesions=[],
            lesion_mask_url=None,
            model_version="not-configured",
            status="not_available",
            retinal_thickening_confirmed=None,
            center_involved_confirmed_by_oct=None,
        )


class LocalSegmentationAdapter:
    """
    Chạy 3 mô hình Attention U-Net (MA/HE/EX) trực tiếp trong process Backend.
    Tương đương với LocalGradingAdapter nhưng dành cho phân đoạn tổn thương.
    """

    async def predict(self, image_bytes: bytes, eye: str) -> SegmentationResult:
        from app.services.lesion_inference import (
            LesionInferenceError,
            get_lesion_inference_service,
        )

        loop = asyncio.get_running_loop()
        try:
            data = await loop.run_in_executor(
                None,
                get_lesion_inference_service().predict,
                image_bytes,
            )
        except LesionInferenceError as exc:
            raise AIServiceUnavailable(
                f"Local segmentation model failed: {exc}"
            ) from exc

        return SegmentationResult(
            lesions=[
                Lesion(
                    key=item["key"],
                    label=item["label"],
                    detected=bool(item["detected"]),
                    area_pct=float(item["area_pct"]),
                    confidence=item.get("confidence"),
                )
                for item in data.get("lesions", [])
            ],
            lesion_mask_url=data.get("lesion_mask_url"),
            model_version=str(data.get("model_version", "unknown")),
            status=str(data.get("status", "ok")),
        )


class HttpGradingAdapter:
    def __init__(self, base_url: str, timeout_seconds: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def predict(self, image_bytes: bytes, eye: str) -> GradingResult:
        if not self.base_url:
            raise AIServiceUnavailable("AI grading service is not configured.")
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    f"{self.base_url}/analyze",
                    files={"file": (f"fundus-{eye}.png", BytesIO(image_bytes), "image/png")},
                    data={"eye": eye},
                )
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise AIServiceUnavailable(f"Grading service failed: {exc}") from exc

        return GradingResult(
            dr_grade=int(data["dr_grade"]),
            dr_label=str(data.get("dr_label", f"Grade {data['dr_grade']}")),
            confidence=float(data.get("confidence", 0)),
            probabilities=data.get("probabilities", {}),
            model_version=str(data.get("model_version", "unknown")),
        )


class HttpSegmentationAdapter:
    def __init__(self, base_url: str, timeout_seconds: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _lesions(data: Dict[str, Any]) -> list[Lesion]:
        raw = data.get("lesions", [])
        if isinstance(raw, dict):
            raw = [dict(value, key=key) for key, value in raw.items()]
        return [
            Lesion(
                key=str(item.get("key", "unknown")),
                label=str(item.get("label") or item.get("label_en") or item.get("key", "Unknown")),
                detected=bool(item.get("detected", False)),
                area_pct=float(item.get("area_pct", 0)),
                confidence=item.get("confidence"),
                distance_to_fovea_mm=item.get("distance_to_fovea_mm"),
            )
            for item in raw
        ]

    async def predict(self, image_bytes: bytes, eye: str) -> SegmentationResult:
        if not self.base_url:
            raise AIServiceUnavailable("AI segmentation service is not configured.")
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    f"{self.base_url}/segment",
                    files={"file": (f"fundus-{eye}.png", BytesIO(image_bytes), "image/png")},
                    data={"eye": eye},
                )
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise AIServiceUnavailable(f"Segmentation service failed: {exc}") from exc

        return SegmentationResult(
            lesions=self._lesions(data),
            lesion_mask_url=data.get("lesion_mask_url") or data.get("mask_url"),
            model_version=str(data.get("model_version", "unknown")),
            status=str(data.get("status", "ok")),
            retinal_thickening_confirmed=data.get("retinal_thickening_confirmed"),
            center_involved_confirmed_by_oct=data.get("center_involved_confirmed_by_oct"),
        )
