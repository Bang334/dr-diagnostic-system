from fastapi import APIRouter, Depends
from sqlalchemy import case, func
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.clinical import DoctorReview, Screening, User
from app.models.patient import Patient
from app.schemas.report import EpidemiologyReport


router = APIRouter(prefix="/reports", tags=["Reports"])


def _rate(detected: int, total: int) -> float:
    return round((detected * 100 / total), 2) if total else 0.0


@router.get("/epidemiology", response_model=EpidemiologyReport)
def epidemiology_report(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    age_group_expr = case(
        (func.date_part("year", func.age(Patient.date_of_birth)) < 40, "Under 40"),
        (
            func.date_part("year", func.age(Patient.date_of_birth)).between(40, 60),
            "40-60",
        ),
        else_="Over 60",
    )
    duration_group_expr = case(
        (Patient.diabetes_duration_years < 5, "Under 5 years"),
        (Patient.diabetes_duration_years.between(5, 10), "5-10 years"),
        else_="Over 10 years",
    )

    by_age_rows = (
        db.query(
            age_group_expr.label("label"),
            func.count(func.distinct(Patient.id)).label("total"),
            func.count(func.distinct(case((DoctorReview.final_dr_grade > 0, Patient.id)))).label("detected"),
        )
        .outerjoin(Screening, Patient.id == Screening.patient_id)
        .outerjoin(DoctorReview, Screening.id == DoctorReview.screening_id)
        .group_by(age_group_expr)
        .all()
    )

    by_duration_rows = (
        db.query(
            duration_group_expr.label("label"),
            func.count(func.distinct(Patient.id)).label("total"),
            func.count(func.distinct(case((DoctorReview.final_dr_grade > 0, Patient.id)))).label("detected"),
        )
        .outerjoin(Screening, Patient.id == Screening.patient_id)
        .outerjoin(DoctorReview, Screening.id == DoctorReview.screening_id)
        .group_by(duration_group_expr)
        .all()
    )

    grade_rows = (
        db.query(DoctorReview.final_dr_grade, func.count(DoctorReview.id))
        .group_by(DoctorReview.final_dr_grade)
        .order_by(DoctorReview.final_dr_grade)
        .all()
    )

    agreement = db.query(
        func.count(case((DoctorReview.is_agree_with_ai.is_(True), 1))).label("agree"),
        func.count(DoctorReview.id).label("total"),
    ).one()

    return {
        "success": True,
        "by_age_group": [
            {
                "label": row.label,
                "total_screened": int(row.total or 0),
                "dr_detected": int(row.detected or 0),
                "prevalence_rate": _rate(int(row.detected or 0), int(row.total or 0)),
            }
            for row in by_age_rows
        ],
        "by_diabetes_duration": [
            {
                "label": row.label,
                "total_screened": int(row.total or 0),
                "dr_detected": int(row.detected or 0),
                "prevalence_rate": _rate(int(row.detected or 0), int(row.total or 0)),
            }
            for row in by_duration_rows
        ],
        "grade_distribution": [
            {"final_dr_grade": int(grade), "count": int(count)}
            for grade, count in grade_rows
        ],
        "agreement_rate": _rate(int(agreement.agree or 0), int(agreement.total or 0)),
    }
