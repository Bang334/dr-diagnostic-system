from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.core.database import get_db
from app.models.patient import Patient
from app.schemas.patient import PatientCreate, PatientUpdate, PatientResponse

router = APIRouter(
    prefix="/patients",
    tags=["Patients"]
)

@router.get("/", response_model=List[PatientResponse])
def get_patients(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Lấy danh sách bệnh nhân tiểu đường trong hệ thống sàng lọc.
    """
    patients = db.query(Patient).offset(skip).limit(limit).all()
    return patients

@router.post("/", response_model=PatientResponse, status_code=status.HTTP_201_CREATED)
def create_patient(patient_in: PatientCreate, db: Session = Depends(get_db)):
    """
    Tạo hồ sơ bệnh nhân tiểu đường mới.
    """
    # Kiểm tra xem mã bệnh nhân đã tồn tại chưa
    db_patient = db.query(Patient).filter(Patient.patient_code == patient_in.patient_code).first()
    if db_patient:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Mã bệnh nhân '{patient_in.patient_code}' đã tồn tại trong hệ thống."
        )
    
    new_patient = Patient(**patient_in.model_dump())
    db.add(new_patient)
    db.commit()
    db.refresh(new_patient)
    return new_patient

@router.get("/{patient_id}", response_model=PatientResponse)
def get_patient_by_id(patient_id: int, db: Session = Depends(get_db)):
    """
    Lấy chi tiết hồ sơ bệnh nhân theo ID.
    """
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy bệnh nhân có ID = {patient_id}."
        )
    return patient

@router.put("/{patient_id}", response_model=PatientResponse)
def update_patient(patient_id: int, patient_in: PatientUpdate, db: Session = Depends(get_db)):
    """
    Cập nhật thông tin hồ sơ bệnh nhân tiểu đường.
    """
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy bệnh nhân có ID = {patient_id}."
        )
    
    update_data = patient_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(patient, field, value)
        
    db.commit()
    db.refresh(patient)
    return patient

@router.delete("/{patient_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_patient(patient_id: int, db: Session = Depends(get_db)):
    """
    Xóa hồ sơ bệnh nhân khỏi hệ thống.
    """
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy bệnh nhân có ID = {patient_id}."
        )
    db.delete(patient)
    db.commit()
    return None
