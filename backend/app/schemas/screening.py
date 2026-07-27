from datetime import date, datetime
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


class ClinicalSummaryResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    provider: str
    model: str
    overview: str
    diagnostic_impression: str
    diagnostic_basis: List[str]
    diagnostic_limitations: List[str]
    key_findings: List[str]
    risk_factors: List[str]
    recommended_actions: List[str]
    follow_up: str
    safety_note: str


class ScreeningUploadResponse(BaseModel):
    success: bool
    screening_id: int
    screening_date: datetime
    status: str
    left_eye: Optional[EyeAnalysisResponse] = None
    right_eye: Optional[EyeAnalysisResponse] = None
    risk_stratification: str
    clinical_recommendation: str
    clinical_summary: ClinicalSummaryResponse
    rule_summary: ClinicalSummaryResponse
    guideline_ids: List[str]
    disclaimer: str


class ScreeningHistoryItem(BaseModel):
    id: int
    patient_id: int
    doctor_id: Optional[int] = None
    doctor_name: Optional[str] = None
    screening_date: datetime
    left_eye_image_url: Optional[str] = None
    right_eye_image_url: Optional[str] = None
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class ScreeningAIResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    eye: str
    dr_grade: int
    dr_label: str
    confidence: float
    probabilities: Dict[str, float]
    model_version: str
    analyzed_at: Optional[datetime] = None


class ScreeningSegmentationDetail(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    eye: str
    lesion_mask_url: Optional[str] = None
    microaneurysm_detected: bool
    microaneurysm_area_pct: float
    hemorrhage_detected: bool
    hemorrhage_area_pct: float
    hard_exudate_detected: bool
    hard_exudate_area_pct: float
    model_version: str


class ScreeningDoctorReview(BaseModel):
    eye: str
    final_dr_grade: int
    final_dr_label: str
    is_agree_with_ai: bool
    clinical_notes: Optional[str] = None
    doctor_name: Optional[str] = None
    confirmed_at: Optional[datetime] = None


class ScreeningEyeDetail(BaseModel):
    eye: str
    image_url: Optional[str] = None
    ai_result: Optional[ScreeningAIResult] = None
    segmentation: Optional[ScreeningSegmentationDetail] = None
    doctor_review: Optional[ScreeningDoctorReview] = None


class ScreeningRecallDetail(BaseModel):
    recall_date: date
    risk_stratification: str
    recommendation: Optional[str] = None
    status: str


class ScreeningDetailResponse(BaseModel):
    id: int
    patient_id: int
    patient_code: str
    patient_name: str
    diabetes_type: Optional[str] = None
    diabetes_duration_years: Optional[float] = None
    latest_hba1c: Optional[float] = None
    screening_date: datetime
    status: str
    results_visible: bool
    visibility_message: Optional[str] = None
    doctor_name: Optional[str] = None
    left_eye: Optional[ScreeningEyeDetail] = None
    right_eye: Optional[ScreeningEyeDetail] = None
    recall: Optional[ScreeningRecallDetail] = None
