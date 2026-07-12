from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.clinical import DoctorReview, Recall, Screening, User
from app.schemas.review import ReviewCreate, ReviewResponse


router = APIRouter(prefix="/reviews", tags=["Doctor Reviews"])


def require_ophthalmic_reviewer(current_user: User = Depends(get_current_user)) -> User:
    department = (current_user.hospital_department or "").casefold()
    if current_user.role != "admin" and "nhãn" not in department:
        raise HTTPException(status_code=403, detail="Kết quả phải được bác sĩ/chuyên khoa mắt xác nhận.")
    return current_user


def _add_months_today(months: int) -> date:
    today = date.today()
    month = today.month - 1 + months
    year = today.year + month // 12
    month = month % 12 + 1
    day = min(today.day, 28)
    return date(year, month, day)


@router.post("/{screening_id}", response_model=ReviewResponse)
def review_screening(
    screening_id: int,
    payload: ReviewCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_ophthalmic_reviewer),
):
    screening = db.query(Screening).filter(Screening.id == screening_id).first()
    if not screening:
        raise HTTPException(status_code=404, detail="Screening not found.")
    if payload.left_eye_review.image_quality == "Poor" or payload.right_eye_review.image_quality == "Poor":
        raise HTTPException(
            status_code=422,
            detail="Ảnh Poor không được xác nhận phân loại; cần chụp lại hoặc chuyển chuyên khoa.",
        )

    db.query(DoctorReview).filter(DoctorReview.screening_id == screening_id).delete()

    db.add(
        DoctorReview(
            screening_id=screening.id,
            doctor_id=current_user.id,
            eye="L",
            final_dr_grade=payload.left_eye_review.final_dr_grade,
            is_agree_with_ai=payload.left_eye_review.is_agree_with_ai,
            clinical_notes=payload.left_eye_review.clinical_notes,
        )
    )
    db.add(
        DoctorReview(
            screening_id=screening.id,
            doctor_id=current_user.id,
            eye="R",
            final_dr_grade=payload.right_eye_review.final_dr_grade,
            is_agree_with_ai=payload.right_eye_review.is_agree_with_ai,
            clinical_notes=payload.right_eye_review.clinical_notes,
        )
    )

    recall_date = _add_months_today(payload.recall_settings.recall_in_months)
    db.add(
        Recall(
            patient_id=screening.patient_id,
            screening_id=screening.id,
            recall_date=recall_date,
            risk_stratification=payload.recall_settings.risk_stratification,
            recommendation=payload.recall_settings.recommendation,
            status="Scheduled",
        )
    )

    screening.status = "Reviewed"
    screening.review_status = "confirmed"
    screening.left_eye_image_quality = payload.left_eye_review.image_quality
    screening.right_eye_image_quality = payload.right_eye_review.image_quality
    screening.doctor_id = current_user.id
    db.commit()

    return {
        "success": True,
        "message": "Doctor review and recall schedule were saved successfully.",
        "screening_status": screening.status,
    }
