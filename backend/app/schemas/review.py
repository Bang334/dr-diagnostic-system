from pydantic import BaseModel, Field
from typing import Literal, Optional


class EyeReviewPayload(BaseModel):
    final_dr_grade: int = Field(..., ge=0, le=4)
    is_agree_with_ai: bool
    clinical_notes: Optional[str] = None
    image_quality: Literal["Good", "Fair", "Poor"]


class RecallSettingsPayload(BaseModel):
    recall_in_months: int = Field(12, ge=1, le=36)
    risk_stratification: str = "Low"
    recommendation: Optional[str] = None


class ReviewCreate(BaseModel):
    left_eye_review: EyeReviewPayload
    right_eye_review: EyeReviewPayload
    recall_settings: RecallSettingsPayload


class ReviewResponse(BaseModel):
    success: bool
    message: str
    screening_status: str
