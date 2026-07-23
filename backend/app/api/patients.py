from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.core.security import hash_password, require_staff
from app.models.account import Account
from app.models.patient import Patient
from app.schemas.patient import PatientCreate, PatientResponse, PatientUpdate


router = APIRouter(prefix="/patients", tags=["Patients"])


@router.get("/", response_model=List[PatientResponse])
def get_patients(
    skip: int = 0,
    limit: int = 100,
    search: Optional[str] = None,
    db: Session = Depends(get_db),
    current_account: Account = Depends(require_staff),
):
    query = db.query(Patient)
    if search:
        pattern = f"%{search}%"
        query = query.filter(
            (Patient.full_name.ilike(pattern))
            | (Patient.patient_code.ilike(pattern))
            | (Patient.phone_number.ilike(pattern))
        )
    return query.order_by(Patient.created_at.desc()).offset(skip).limit(limit).all()


@router.post("/", response_model=PatientResponse, status_code=status.HTTP_201_CREATED)
def create_patient(
    patient_in: PatientCreate,
    db: Session = Depends(get_db),
    current_account: Account = Depends(require_staff),
):
    existing = db.query(Patient).filter(Patient.patient_code == patient_in.patient_code).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Patient code '{patient_in.patient_code}' already exists.",
        )

    account_conflict = (
        db.query(Account)
        .filter(func.lower(Account.username) == patient_in.patient_code.casefold())
        .first()
    )
    if account_conflict:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Username '{patient_in.patient_code}' is already in use.",
        )

    account = Account(
        username=patient_in.patient_code,
        password_hash=hash_password(settings.PATIENT_DEFAULT_PASSWORD),
        display_name=patient_in.full_name,
        role="patient",
        is_active=True,
    )
    db.add(account)
    db.flush()
    patient = Patient(account_id=account.id, **patient_in.model_dump())
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


@router.get("/{patient_id}", response_model=PatientResponse)
def get_patient_by_id(
    patient_id: int,
    db: Session = Depends(get_db),
    current_account: Account = Depends(require_staff),
):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found.")
    return patient


@router.put("/{patient_id}", response_model=PatientResponse)
def update_patient(
    patient_id: int,
    patient_in: PatientUpdate,
    db: Session = Depends(get_db),
    current_account: Account = Depends(require_staff),
):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found.")

    for field, value in patient_in.model_dump(exclude_unset=True).items():
        setattr(patient, field, value)
    if patient.account:
        patient.account.display_name = patient.full_name
    db.commit()
    db.refresh(patient)
    return patient


@router.delete("/{patient_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_patient(
    patient_id: int,
    db: Session = Depends(get_db),
    current_account: Account = Depends(require_staff),
):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found.")
    account = patient.account
    db.delete(patient)
    db.flush()
    if account:
        db.delete(account)
    db.commit()
    return None


@router.get("/{patient_id}/recalls", status_code=status.HTTP_200_OK)
def get_patient_recalls(
    patient_id: int,
    db: Session = Depends(get_db),
    current_account: Account = Depends(require_staff),
):
    from app.models.clinical import Recall
    recalls = db.query(Recall).filter(Recall.patient_id == patient_id).order_by(Recall.recall_date.desc()).all()
    return recalls


@router.post("/{patient_id}/recalls", status_code=status.HTTP_201_CREATED)
def create_quick_recall(
    patient_id: int,
    payload: dict,
    db: Session = Depends(get_db),
    current_account: Account = Depends(require_staff),
):
    from app.models.clinical import Recall, Screening
    from datetime import date
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found.")

    recall_date_str = payload.get("recall_date")
    try:
        recall_date = date.fromisoformat(recall_date_str)
    except (TypeError, ValueError):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid recall_date format. Use YYYY-MM-DD.")

    screening_id = payload.get("screening_id")
    recall = None
    if screening_id is not None:
        screening = (
            db.query(Screening)
            .filter(Screening.id == screening_id, Screening.patient_id == patient_id)
            .first()
        )
        if not screening:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Screening does not belong to this patient.",
            )
        recall = db.query(Recall).filter(Recall.screening_id == screening_id).first()

    if recall is None:
        recall = Recall(patient_id=patient_id, screening_id=screening_id)
        db.add(recall)
    recall.recall_date = recall_date
    recall.risk_stratification = payload.get("risk_stratification", "Low")
    recall.recommendation = payload.get("recommendation", "")
    recall.status = "Scheduled"
    db.commit()
    db.refresh(recall)
    return {"success": True, "message": "Recall scheduled successfully.", "recall_id": recall.id}
