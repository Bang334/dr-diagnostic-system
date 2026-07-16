from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from io import BytesIO
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.clinical.adapters import (
    AIServiceUnavailable,
    HttpGradingAdapter,
    HttpSegmentationAdapter,
    LocalGradingAdapter,
    UnavailableSegmentationAdapter,
)
from app.clinical.analysis import ClinicalAnalysisModule, InvalidFundusSet
from app.clinical.models import ClinicalContext, EyeImageSet
from app.clinical.report import clinical_report_pdf
from app.clinical.quality import sanitize_fundus_image
from app.core.config import BASE_DIR, settings
from app.core.database import get_db
from app.core.security import get_current_user
from app.models.clinical import AIResult, LesionSegmentationResult, Screening, User
from app.models.patient import Patient
from app.schemas.screening import ScreeningHistoryItem, ScreeningUploadResponse


router = APIRouter(prefix="/screenings", tags=["Screenings"])
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def build_clinical_module() -> ClinicalAnalysisModule:
    timeout = settings.AI_REQUEST_TIMEOUT_SECONDS
    grading_url = settings.AI_GRADING_SERVICE_URL.strip()
    segmentation_url = settings.AI_SEGMENTATION_SERVICE_URL.strip()
    return ClinicalAnalysisModule(
        grading=(
            LocalGradingAdapter()
            if grading_url.lower() in {"", "local"}
            else HttpGradingAdapter(grading_url, timeout)
        ),
        segmentation=(
            UnavailableSegmentationAdapter()
            if segmentation_url.lower() in {"", "disabled", "none"}
            else HttpSegmentationAdapter(segmentation_url, timeout)
        ),
    )


def _validate_content_type(file: UploadFile) -> None:
    if file.content_type not in {"image/png", "image/jpeg", "image/jpg"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chỉ chấp nhận ảnh đáy mắt PNG/JPG/JPEG.",
        )


async def _read_upload(file: UploadFile) -> bytes:
    _validate_content_type(file)
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Tệp ảnh rỗng.")
    return content


def _persist_upload(content: bytes, prefix: str) -> str:
    filename = f"{prefix}-{uuid4().hex}.png"
    (UPLOAD_DIR / filename).write_bytes(content)
    return f"/uploads/{filename}"


def _lesion_map(segmentation: dict) -> dict:
    return {item["key"]: item for item in segmentation.get("lesions", [])}


def _eye_response(assessment: dict, posterior_url: str, disc_url: str) -> dict:
    segmentation = assessment["segmentation"]
    return {
        "image_url": posterior_url,
        "disc_image_url": disc_url,
        "quality": assessment["quality"],
        "ai_result": {"eye": assessment["eye"], **assessment["grading"]},
        "segmentation": {
            "eye": assessment["eye"],
            "lesion_mask_url": segmentation.get("lesion_mask_url"),
            "lesions": segmentation.get("lesions", []),
            "model_version": segmentation["model_version"],
        },
        "review_priority": assessment["review_priority"],
        "follow_up_window": assessment["follow_up_window"],
        "referral": assessment["referral"],
        "macular_status": assessment["macular_status"],
        "findings": assessment["findings"],
        "actions": assessment["actions"],
        "safety_flags": assessment["safety_flags"],
    }


@router.post("/upload", response_model=ScreeningUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_screening(
    patient_id: int = Form(...),
    left_disc_image: UploadFile = File(...),
    left_posterior_pole_image: UploadFile = File(...),
    right_disc_image: UploadFile = File(...),
    right_posterior_pole_image: UploadFile = File(...),
    systolic_bp: Optional[int] = Form(None),
    diastolic_bp: Optional[int] = Form(None),
    visual_acuity_left: Optional[float] = Form(None),
    visual_acuity_right: Optional[float] = Form(None),
    sudden_vision_loss: bool = Form(False),
    pregnant: bool = Form(False),
    kidney_disease: bool = Form(False),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Không tìm thấy bệnh nhân.")

    left_disc, left_posterior, right_disc, right_posterior = await _read_all(
        left_disc_image,
        left_posterior_pole_image,
        right_disc_image,
        right_posterior_pole_image,
    )
    try:
        left_disc, left_posterior, right_disc, right_posterior = (
            sanitize_fundus_image(left_disc),
            sanitize_fundus_image(left_posterior),
            sanitize_fundus_image(right_disc),
            sanitize_fundus_image(right_posterior),
        )
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Không thể giải mã và làm sạch metadata ảnh.") from exc
    context = ClinicalContext(
        patient_code=patient.patient_code,
        diabetes_duration_years=float(patient.diabetes_duration_years) if patient.diabetes_duration_years is not None else None,
        hba1c=float(patient.latest_hba1c) if patient.latest_hba1c is not None else None,
        systolic_bp=systolic_bp,
        diastolic_bp=diastolic_bp,
        visual_acuity_left=visual_acuity_left,
        visual_acuity_right=visual_acuity_right,
        sudden_vision_loss=sudden_vision_loss,
        pregnant=pregnant,
        kidney_disease=kidney_disease,
    )

    try:
        result = await build_clinical_module().analyze(
            EyeImageSet("L", left_disc, left_posterior),
            EyeImageSet("R", right_disc, right_posterior),
            context,
        )
    except InvalidFundusSet as exc:
        raise HTTPException(status_code=422, detail={"message": str(exc), "quality": exc.quality}) from exc
    except AIServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    assessment = result.model_dump(mode="json")
    left_posterior_url = _persist_upload(left_posterior, "left-posterior")
    right_posterior_url = _persist_upload(right_posterior, "right-posterior")
    left_disc_url = _persist_upload(left_disc, "left-disc")
    right_disc_url = _persist_upload(right_disc, "right-disc")

    screening = Screening(
        patient_id=patient.id,
        doctor_id=current_user.id,
        left_eye_image_url=left_posterior_url,
        right_eye_image_url=right_posterior_url,
        left_disc_image_url=left_disc_url,
        right_disc_image_url=right_disc_url,
        left_eye_image_quality="ReviewRequired",
        right_eye_image_quality="ReviewRequired",
        status="AI_Analyzed",
        review_status="draft",
        clinical_assessment=assessment,
    )
    db.add(screening)
    db.flush()

    for eye_key in ("left_eye", "right_eye"):
        eye = assessment[eye_key]
        grading = eye["grading"]
        segmentation = eye["segmentation"]
        db.add(AIResult(
            screening_id=screening.id,
            eye=eye["eye"],
            dr_grade=grading["dr_grade"],
            confidence=Decimal(str(grading["confidence"])),
            probabilities=grading["probabilities"],
            model_version=grading["model_version"],
        ))
        lesions = _lesion_map(segmentation)
        db.add(LesionSegmentationResult(
            screening_id=screening.id,
            eye=eye["eye"],
            lesion_mask_url=segmentation.get("lesion_mask_url"),
            microaneurysm_detected=bool(lesions.get("microaneurysm", {}).get("detected", False)),
            microaneurysm_area_pct=Decimal(str(lesions.get("microaneurysm", {}).get("area_pct", 0))),
            hemorrhage_detected=bool(lesions.get("hemorrhage", {}).get("detected", False)),
            hemorrhage_area_pct=Decimal(str(lesions.get("hemorrhage", {}).get("area_pct", 0))),
            hard_exudate_detected=bool(lesions.get("hard_exudate", {}).get("detected", False)),
            hard_exudate_area_pct=Decimal(str(lesions.get("hard_exudate", {}).get("area_pct", 0))),
            model_version=segmentation["model_version"],
        ))

    db.commit()
    db.refresh(screening)
    return {
        "success": True,
        "screening_id": screening.id,
        "screening_date": screening.screening_date,
        "status": screening.status,
        "review_status": screening.review_status,
        "left_eye": _eye_response(assessment["left_eye"], left_posterior_url, left_disc_url),
        "right_eye": _eye_response(assessment["right_eye"], right_posterior_url, right_disc_url),
        "risk_stratification": assessment["overall_priority"],
        "clinical_recommendation": assessment["clinical_recommendation"],
        "guideline_ids": assessment["guideline_ids"],
        "disclaimer": assessment["disclaimer"],
    }


async def _read_all(*files: UploadFile) -> tuple[bytes, bytes, bytes, bytes]:
    # Kept as one operation so no partial filesystem writes occur before validation.
    values = []
    for file in files:
        values.append(await _read_upload(file))
    return tuple(values)  # type: ignore[return-value]


@router.get("/patient/{patient_id}", response_model=List[ScreeningHistoryItem])
def get_screenings_by_patient(
    patient_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rows = (
        db.query(Screening, User.full_name.label("doctor_name"))
        .outerjoin(User, Screening.doctor_id == User.id)
        .filter(Screening.patient_id == patient_id)
        .order_by(Screening.screening_date.desc())
        .all()
    )
    return [{**screening.__dict__, "doctor_name": doctor_name} for screening, doctor_name in rows]


@router.get("/{screening_id}/report.pdf")
def download_screening_report(
    screening_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    screening = db.query(Screening).filter(Screening.id == screening_id).first()
    if not screening or not screening.clinical_assessment:
        raise HTTPException(status_code=404, detail="Không tìm thấy báo cáo lâm sàng.")
    patient = db.query(Patient).filter(Patient.id == screening.patient_id).first()
    pdf = clinical_report_pdf(
        {"patient_code": patient.patient_code, "full_name": patient.full_name},
        screening.clinical_assessment,
    )
    return StreamingResponse(
        BytesIO(pdf),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="screening-{screening_id}.pdf"'},
    )
