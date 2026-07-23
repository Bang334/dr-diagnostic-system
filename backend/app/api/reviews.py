from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_account
from app.models.account import Account
from app.models.clinical import DoctorReview, Recall, Screening
from app.schemas.review import ReviewCreate, ReviewResponse


router = APIRouter(prefix="/reviews", tags=["Doctor Reviews"])


def require_ophthalmic_reviewer(
    current_account: Account = Depends(get_current_account),
) -> Account:
    department = (current_account.hospital_department or "").casefold()
    if current_account.role != "admin" and (
        current_account.role != "doctor" or "nhãn" not in department
    ):
        raise HTTPException(status_code=403, detail="Kết quả phải được bác sĩ/chuyên khoa mắt xác nhận.")
    return current_account


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
    current_account: Account = Depends(require_ophthalmic_reviewer),
):
    screening = db.query(Screening).filter(Screening.id == screening_id).first()
    if not screening:
        raise HTTPException(status_code=404, detail="Screening not found.")
    db.query(DoctorReview).filter(DoctorReview.screening_id == screening_id).delete()

    doctor = current_account.doctor_profile
    for eye, eye_review in (
        ("L", payload.left_eye_review),
        ("R", payload.right_eye_review),
    ):
        if eye_review is None:
            continue
        db.add(
            DoctorReview(
                screening_id=screening.id,
                reviewed_by_account_id=current_account.id,
                doctor_id=doctor.id if doctor else None,
                eye=eye,
                final_dr_grade=eye_review.final_dr_grade,
                is_agree_with_ai=eye_review.is_agree_with_ai,
                clinical_notes=eye_review.clinical_notes,
            )
        )

    recall_date = _add_months_today(payload.recall_settings.recall_in_months)
    db.query(Recall).filter(Recall.screening_id == screening.id).delete()
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
    if doctor:
        screening.doctor_id = doctor.id
    db.commit()

    return {
        "success": True,
        "message": "Doctor review and recall schedule were saved successfully.",
        "screening_status": screening.status,
    }
