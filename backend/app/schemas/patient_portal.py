from datetime import date, datetime
from typing import List, Optional

from pydantic import BaseModel

from app.schemas.patient import PatientResponse


class PatientPortalScreening(BaseModel):
    id: int
    screening_date: datetime
    status: str
    doctor_name: Optional[str] = None
    left_eye_grade: Optional[int] = None
    right_eye_grade: Optional[int] = None
    left_eye_notes: Optional[str] = None
    right_eye_notes: Optional[str] = None


class PatientPortalRecall(BaseModel):
    id: int
    screening_id: Optional[int] = None
    recall_date: date
    risk_stratification: str
    recommendation: Optional[str] = None
    status: str


class PatientPortalOverview(BaseModel):
    patient: PatientResponse
    screenings: List[PatientPortalScreening]
    recalls: List[PatientPortalRecall]

