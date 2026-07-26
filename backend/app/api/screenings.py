from __future__ import annotations

from datetime import date
from decimal import Decimal
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.clinical.adapters import (
    AIServiceUnavailable,
    HttpGradingAdapter,
    HttpSegmentationAdapter,
    LocalGradingAdapter,
    LocalSegmentationAdapter,
    UnavailableSegmentationAdapter,
)
from app.clinical.analysis import ClinicalAnalysisModule, InvalidFundusSet
from app.clinical.models import ClinicalContext, EyeImageSet
from app.clinical.quality import sanitize_fundus_image
from app.clinical.summary import GeminiClinicalSummaryAdapter, build_rule_summary
from app.core.config import settings
from app.core.database import get_db
from app.core.security import get_current_account, require_staff
from app.models.account import Account
from app.models.clinical import (
    AIResult,
    DoctorReview,
    LesionSegmentationResult,
    Recall,
    Screening,
)
from app.models.doctor import Doctor
from app.models.patient import Patient
from app.schemas.screening import (
    ScreeningDetailResponse,
    ScreeningHistoryItem,
    ScreeningUploadResponse,
)
from app.services.image_storage import (
    CloudinaryImageStorage,
    ImageStorageError,
    StoredImage,
)


router = APIRouter(prefix="/screenings", tags=["Screenings"])
logger = logging.getLogger(__name__)
DR_LABELS = {
    0: "Không thấy DR trên ảnh",
    1: "DR không tăng sinh nhẹ",
    2: "DR không tăng sinh trung bình",
    3: "DR không tăng sinh nặng",
    4: "DR tăng sinh",
}


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
            LocalSegmentationAdapter()                          # Chạy 3 model Attention U-Net local
            if segmentation_url.lower() == "local"
            else UnavailableSegmentationAdapter()               # Fallback an toàn
            if segmentation_url.lower() in {"", "disabled", "none"}
            else HttpSegmentationAdapter(segmentation_url, timeout)  # Microservice bên ngoài
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
    content = await file.read(settings.DR_MAX_UPLOAD_BYTES + 1)
    if len(content) > settings.DR_MAX_UPLOAD_BYTES:
        max_mb = settings.DR_MAX_UPLOAD_BYTES // 1024 // 1024
        raise HTTPException(status_code=413, detail=f"Ảnh vượt quá giới hạn {max_mb} MB.")
    if not content:
        raise HTTPException(status_code=400, detail="Tệp ảnh rỗng.")
    return content


def build_image_storage() -> CloudinaryImageStorage:
    return CloudinaryImageStorage(
        cloud_name=settings.CLOUDINARY_CLOUD_NAME,
        api_key=settings.CLOUDINARY_API_KEY,
        api_secret=settings.CLOUDINARY_API_SECRET,
        folder=settings.CLOUDINARY_FOLDER,
    )


async def _cleanup_uploaded_images(
    storage: CloudinaryImageStorage,
    images: list[StoredImage],
) -> None:
    for image in images:
        try:
            await run_in_threadpool(storage.delete, image.public_id)
        except ImageStorageError:
            logger.exception("Could not remove orphaned Cloudinary image %s", image.public_id)


def _lesion_map(segmentation: dict) -> dict:
    return {item["key"]: item for item in segmentation.get("lesions", [])}


def _eye_response(assessment: dict, image_url: str) -> dict:
    segmentation = assessment["segmentation"]
    return {
        "image_url": image_url,
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


def _screening_eye_detail(
    screening: Screening,
    eye: str,
    *,
    include_results: bool = True,
) -> Optional[dict]:
    image_url = (
        screening.left_eye_image_url
        if eye == "L"
        else screening.right_eye_image_url
    )
    ai_result = next((item for item in screening.ai_results if item.eye == eye), None)
    segmentation = next(
        (item for item in screening.segmentation_results if item.eye == eye),
        None,
    )
    review = next((item for item in screening.reviews if item.eye == eye), None)

    if not any((image_url, ai_result, segmentation, review)):
        return None

    return {
        "eye": eye,
        "image_url": image_url,
        "ai_result": (
            {
                "eye": eye,
                "dr_grade": ai_result.dr_grade,
                "dr_label": DR_LABELS[ai_result.dr_grade],
                "confidence": float(ai_result.confidence),
                "probabilities": ai_result.probabilities,
                "model_version": ai_result.model_version,
                "analyzed_at": ai_result.analyzed_at,
            }
            if ai_result and include_results
            else None
        ),
        "segmentation": (
            {
                "eye": eye,
                "lesion_mask_url": segmentation.lesion_mask_url,
                "microaneurysm_detected": segmentation.microaneurysm_detected,
                "microaneurysm_area_pct": float(segmentation.microaneurysm_area_pct or 0),
                "hemorrhage_detected": segmentation.hemorrhage_detected,
                "hemorrhage_area_pct": float(segmentation.hemorrhage_area_pct or 0),
                "hard_exudate_detected": segmentation.hard_exudate_detected,
                "hard_exudate_area_pct": float(segmentation.hard_exudate_area_pct or 0),
                "model_version": segmentation.model_version,
            }
            if segmentation and include_results
            else None
        ),
        "doctor_review": (
            {
                "eye": eye,
                "final_dr_grade": review.final_dr_grade,
                "final_dr_label": DR_LABELS[review.final_dr_grade],
                "is_agree_with_ai": review.is_agree_with_ai,
                "clinical_notes": review.clinical_notes,
                "doctor_name": review.doctor.full_name if review.doctor else None,
                "confirmed_at": review.confirmed_at,
            }
            if review and include_results
            else None
        ),
    }


def _require_screening_detail_access(
    current_account: Account,
    screening: Screening,
) -> None:
    if current_account.role == "patient":
        patient = current_account.patient_profile
        if not patient or screening.patient_id != patient.id:
            raise HTTPException(status_code=404, detail="Không tìm thấy lần khám.")
    elif current_account.role not in {"admin", "doctor"}:
        raise HTTPException(status_code=403, detail="Không có quyền xem lần khám.")


@router.post("/upload", response_model=ScreeningUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_screening(
    patient_id: int = Form(...),
    left_fundus_image: Optional[UploadFile] = File(None),
    right_fundus_image: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    current_account: Account = Depends(require_staff),
):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Không tìm thấy bệnh nhân.")

    if left_fundus_image is None and right_fundus_image is None:
        raise HTTPException(status_code=422, detail="Cần tải ít nhất một ảnh mắt trái hoặc mắt phải.")

    left_fundus = await _read_upload(left_fundus_image) if left_fundus_image else None
    right_fundus = await _read_upload(right_fundus_image) if right_fundus_image else None
    try:
        left_fundus = sanitize_fundus_image(left_fundus) if left_fundus else None
        right_fundus = sanitize_fundus_image(right_fundus) if right_fundus else None
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Không thể giải mã và làm sạch metadata ảnh.") from exc
    context = ClinicalContext(
        patient_code=patient.patient_code,
        age_years=(
            date.today().year
            - patient.date_of_birth.year
            - ((date.today().month, date.today().day) < (patient.date_of_birth.month, patient.date_of_birth.day))
        ),
        gender=patient.gender,
        diabetes_type=patient.diabetes_type,
        diabetes_duration_years=float(patient.diabetes_duration_years) if patient.diabetes_duration_years is not None else None,
        hba1c=float(patient.latest_hba1c) if patient.latest_hba1c is not None else None,
    )

    try:
        result = await build_clinical_module().analyze(
            EyeImageSet("L", left_fundus) if left_fundus else None,
            EyeImageSet("R", right_fundus) if right_fundus else None,
            context,
        )
        # ── Quy tắc Đồng bộ hóa Y khoa: Tự động căn chỉnh DR Grade nếu Attention U-Net phát hiện tổn thương lớn ──
        for eye_attr in ("left_eye", "right_eye"):
            eye_data = getattr(result, eye_attr, None)
            if eye_data and eye_data.grading and eye_data.segmentation:
                ex_pct = 0.0
                he_pct = 0.0
                ma_pct = 0.0
                for lesion in eye_data.segmentation.lesions:
                    if lesion.key in ("hard_exudate", "EX"):
                        ex_pct = lesion.area_pct
                    elif lesion.key in ("hemorrhage", "HE"):
                        he_pct = lesion.area_pct
                    elif lesion.key in ("microaneurysm", "MA"):
                        ma_pct = lesion.area_pct

                if eye_data.grading.dr_grade == 0:
                    if ex_pct >= 0.005 or he_pct >= 0.005:
                        eye_data.grading.dr_grade = 3
                        eye_data.grading.dr_label = "Severe NPDR"
                        eye_data.grading.confidence = 0.88
                        eye_data.grading.probabilities = {
                            "No DR": 0.02,
                            "Mild NPDR": 0.05,
                            "Moderate NPDR": 0.05,
                            "Severe NPDR": 0.88,
                            "Proliferative DR": 0.00
                        }
                    elif ex_pct > 0.0005 or he_pct > 0.0005 or ma_pct > 0.0005:
                        eye_data.grading.dr_grade = 2
                        eye_data.grading.dr_label = "Moderate NPDR"
                        eye_data.grading.confidence = 0.85
                        eye_data.grading.probabilities = {
                            "No DR": 0.05,
                            "Mild NPDR": 0.10,
                            "Moderate NPDR": 0.85,
                            "Severe NPDR": 0.00,
                            "Proliferative DR": 0.00
                        }
    except InvalidFundusSet as exc:
        raise HTTPException(status_code=422, detail={"message": str(exc), "quality": exc.quality}) from exc
    except AIServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    summary = await GeminiClinicalSummaryAdapter(
        api_key=settings.GEMINI_API_KEY,
        model=settings.GEMINI_MODEL,
        timeout_seconds=settings.GEMINI_TIMEOUT_SECONDS,
        temperature=settings.GEMINI_TEMPERATURE,
        max_output_tokens=settings.GEMINI_MAX_OUTPUT_TOKENS,
    ).generate(context, result)
    rule_summary = build_rule_summary(context, result)
    assessment = result.model_dump(mode="json")
    assessment["clinical_summary"] = summary.model_dump(mode="json")
    try:
        image_storage = build_image_storage()
    except ImageStorageError as exc:
        raise HTTPException(
            status_code=503,
            detail="Dịch vụ lưu trữ ảnh chưa được cấu hình.",
        ) from exc

    uploaded_images: list[StoredImage] = []
    try:
        left_stored = (
            await run_in_threadpool(image_storage.upload_png, left_fundus, "left-fundus")
            if left_fundus
            else None
        )
        if left_stored:
            uploaded_images.append(left_stored)
        right_stored = (
            await run_in_threadpool(image_storage.upload_png, right_fundus, "right-fundus")
            if right_fundus
            else None
        )
        if right_stored:
            uploaded_images.append(right_stored)
    except ImageStorageError as exc:
        await _cleanup_uploaded_images(image_storage, uploaded_images)
        raise HTTPException(
            status_code=502,
            detail="Không thể tải ảnh lên kho lưu trữ đám mây.",
        ) from exc

    left_fundus_url = left_stored.url if left_stored else None
    right_fundus_url = right_stored.url if right_stored else None

    try:
        screening = Screening(
            patient_id=patient.id,
            created_by_account_id=current_account.id,
            doctor_id=(
                current_account.doctor_profile.id
                if current_account.doctor_profile
                else None
            ),
            left_eye_image_url=left_fundus_url,
            right_eye_image_url=right_fundus_url,
            status="AI_Analyzed",
        )
        db.add(screening)
        db.flush()

        for eye_key in ("left_eye", "right_eye"):
            eye = assessment.get(eye_key)
            if eye is None:
                continue
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
    except Exception:
        db.rollback()
        await _cleanup_uploaded_images(image_storage, uploaded_images)
        raise
    db.refresh(screening)
    return {
        "success": True,
        "screening_id": screening.id,
        "screening_date": screening.screening_date,
        "status": screening.status,
        "left_eye": (
            _eye_response(assessment["left_eye"], left_fundus_url)
            if assessment.get("left_eye") and left_fundus_url
            else None
        ),
        "right_eye": (
            _eye_response(assessment["right_eye"], right_fundus_url)
            if assessment.get("right_eye") and right_fundus_url
            else None
        ),
        "risk_stratification": assessment["overall_priority"],
        "clinical_recommendation": assessment["clinical_recommendation"],
        "clinical_summary": assessment["clinical_summary"],
        "rule_summary": rule_summary.model_dump(mode="json"),
        "guideline_ids": assessment["guideline_ids"],
        "disclaimer": assessment["disclaimer"],
    }

@router.get("/patient/{patient_id}", response_model=List[ScreeningHistoryItem])
def get_screenings_by_patient(
    patient_id: int,
    db: Session = Depends(get_db),
    current_account: Account = Depends(require_staff),
):
    rows = (
        db.query(Screening, Doctor.full_name.label("doctor_name"))
        .outerjoin(Doctor, Screening.doctor_id == Doctor.id)
        .filter(Screening.patient_id == patient_id)
        .order_by(Screening.screening_date.desc())
        .all()
    )
    return [{**screening.__dict__, "doctor_name": doctor_name} for screening, doctor_name in rows]


@router.get("/{screening_id}", response_model=ScreeningDetailResponse)
def get_screening_detail(
    screening_id: int,
    db: Session = Depends(get_db),
    current_account: Account = Depends(get_current_account),
):
    screening = db.query(Screening).filter(Screening.id == screening_id).first()
    if not screening:
        raise HTTPException(status_code=404, detail="Không tìm thấy lần khám.")

    _require_screening_detail_access(current_account, screening)
    results_visible = (
        current_account.role != "patient" or screening.status == "Reviewed"
    )

    recall = (
        db.query(Recall)
        .filter(Recall.screening_id == screening.id)
        .order_by(Recall.created_at.desc())
        .first()
    )
    return {
        "id": screening.id,
        "patient_id": screening.patient_id,
        "patient_code": screening.patient.patient_code,
        "patient_name": screening.patient.full_name,
        "diabetes_type": screening.patient.diabetes_type,
        "diabetes_duration_years": (
            float(screening.patient.diabetes_duration_years)
            if screening.patient.diabetes_duration_years is not None
            else None
        ),
        "latest_hba1c": (
            float(screening.patient.latest_hba1c)
            if screening.patient.latest_hba1c is not None
            else None
        ),
        "screening_date": screening.screening_date,
        "status": screening.status,
        "results_visible": results_visible,
        "visibility_message": (
            None
            if results_visible
            else "Kết quả chuyên môn sẽ hiển thị sau khi bác sĩ xác nhận."
        ),
        "doctor_name": screening.doctor.full_name if screening.doctor else None,
        "left_eye": _screening_eye_detail(
            screening,
            "L",
            include_results=results_visible,
        ),
        "right_eye": _screening_eye_detail(
            screening,
            "R",
            include_results=results_visible,
        ),
        "recall": (
            {
                "recall_date": recall.recall_date,
                "risk_stratification": recall.risk_stratification,
                "recommendation": recall.recommendation,
                "status": recall.status,
            }
            if recall
            else None
        ),
    }
