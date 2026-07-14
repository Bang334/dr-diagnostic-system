from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class AIResultResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    eye: str
    dr_grade: int
    dr_label: str
    confidence: float
    probabilities: Dict[str, float]
    model_version: str


class LesionItem(BaseModel):
    key: str
    label: str
    detected: bool
    area_pct: float


class SegmentationResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    eye: str
    lesion_mask_url: Optional[str] = None
    lesions: List[LesionItem]
    model_version: str


class EyeAnalysisResponse(BaseModel):
    image_url: str
    disc_image_url: str
    quality: Dict[str, Any]
    ai_result: AIResultResponse
    segmentation: SegmentationResponse
    review_priority: str
    follow_up_window: str
    referral: str
    macular_status: str
    findings: List[str]
    actions: List[str]
    safety_flags: List[str]


class ScreeningUploadResponse(BaseModel):
    success: bool
    screening_id: int
    screening_date: datetime
    status: str
    left_eye: EyeAnalysisResponse
    right_eye: EyeAnalysisResponse
    risk_stratification: str
    clinical_recommendation: str
    review_status: str
    guideline_ids: List[str]
    disclaimer: str


class ScreeningHistoryItem(BaseModel):
    id: int
    patient_id: int
    doctor_id: Optional[int] = None
    doctor_name: Optional[str] = None
    screening_date: datetime
    left_eye_image_url: str
    right_eye_image_url: str
    left_disc_image_url: Optional[str] = None
    right_disc_image_url: Optional[str] = None
    left_eye_image_quality: str
    right_eye_image_quality: str
    status: str
    review_status: str = "draft"
    created_at: datetime

    class Config:
        from_attributes = True
