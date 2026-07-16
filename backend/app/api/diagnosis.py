from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.services.dr_inference import (
    InvalidFundusImage,
    ModelNotReady,
    get_dr_inference_service,
)


router = APIRouter(prefix="/diagnosis", tags=["Quick DR diagnosis"])
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png"}


@router.get("/model-info")
def model_info():
    return get_dr_inference_service().model_info()


@router.post("/analyze")
async def analyze_fundus(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chỉ chấp nhận ảnh PNG, JPG hoặc JPEG.",
        )

    content = await file.read(settings.DR_MAX_UPLOAD_BYTES + 1)
    if len(content) > settings.DR_MAX_UPLOAD_BYTES:
        max_mb = settings.DR_MAX_UPLOAD_BYTES // 1024 // 1024
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Ảnh vượt quá giới hạn {max_mb} MB.",
        )

    try:
        return await run_in_threadpool(get_dr_inference_service().predict, content)
    except InvalidFundusImage as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelNotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

