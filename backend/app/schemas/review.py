from pydantic import BaseModel, Field, model_validator
from typing import Optional


class EyeReviewPayload(BaseModel):
    final_dr_grade: int = Field(..., ge=0, le=4)
    is_agree_with_ai: bool
    clinical_notes: Optional[str] = None


class RecallSettingsPayload(BaseModel):
    recall_in_months: int = Field(12, ge=1, le=36)
    risk_stratification: str = "Low"
    recommendation: Optional[str] = None


class ReviewCreate(BaseModel):
    left_eye_review: Optional[EyeReviewPayload] = None
    right_eye_review: Optional[EyeReviewPayload] = None
    recall_settings: RecallSettingsPayload

    @model_validator(mode="after")
    def require_at_least_one_eye(self):
        if self.left_eye_review is None and self.right_eye_review is None:
            raise ValueError("At least one eye review is required.")
        return self


class ReviewResponse(BaseModel):
    success: bool
    message: str
    screening_status: str
