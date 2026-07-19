from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import require_patient
from app.models.account import Account
from app.models.clinical import DoctorReview, Recall, Screening
from app.models.doctor import Doctor
from app.models.patient import Patient
from app.schemas.patient_portal import PatientPortalOverview


router = APIRouter(prefix="/patient-portal", tags=["Patient Portal"])


@router.get("/overview", response_model=PatientPortalOverview)
def get_patient_overview(
    db: Session = Depends(get_db),
    current_account: Account = Depends(require_patient),
):
    patient = current_account.patient_profile
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient profile was not found.",
        )

    screening_rows = (
        db.query(Screening, Doctor.full_name.label("doctor_name"))
        .outerjoin(Doctor, Screening.doctor_id == Doctor.id)
        .filter(Screening.patient_id == patient.id)
        .order_by(Screening.screening_date.desc())
        .all()
    )
    screening_ids = [screening.id for screening, _ in screening_rows]
    reviews = (
        db.query(DoctorReview)
        .filter(DoctorReview.screening_id.in_(screening_ids))
        .all()
        if screening_ids
        else []
    )
    review_map = {(review.screening_id, review.eye): review for review in reviews}

    screenings = []
    for screening, doctor_name in screening_rows:
        left_review = review_map.get((screening.id, "L"))
        right_review = review_map.get((screening.id, "R"))
        is_reviewed = screening.status == "Reviewed"
        screenings.append(
            {
                "id": screening.id,
                "screening_date": screening.screening_date,
                "status": screening.status,
                "doctor_name": doctor_name,
                "left_eye_grade": left_review.final_dr_grade if is_reviewed and left_review else None,
                "right_eye_grade": right_review.final_dr_grade if is_reviewed and right_review else None,
                "left_eye_notes": left_review.clinical_notes if is_reviewed and left_review else None,
                "right_eye_notes": right_review.clinical_notes if is_reviewed and right_review else None,
            }
        )

    recalls = (
        db.query(Recall)
        .filter(Recall.patient_id == patient.id)
        .order_by(Recall.recall_date.asc())
        .all()
    )
    return {"patient": patient, "screenings": screenings, "recalls": recalls}
