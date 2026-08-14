from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

from pydantic import BaseModel, ConfigDict, Field


DR_LABELS = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}


@dataclass(frozen=True)
class EyeImageSet:
    """One gradable fundus image for one eye."""

    eye: str
    fundus_image: bytes


class PriorEyeFinding(BaseModel):
    eye: str
    dr_grade: int = Field(ge=0, le=4)
    dr_label: str
    confidence: float = Field(ge=0, le=1)
    detected_lesions: List[str] = Field(default_factory=list)


class PriorScreening(BaseModel):
    screening_date: datetime
    eyes: List[PriorEyeFinding] = Field(default_factory=list)


class ClinicalContext(BaseModel):
    patient_code: Optional[str] = None
    age_years: Optional[int] = Field(default=None, ge=0, le=130)
    gender: Optional[str] = None
    diabetes_type: Optional[str] = None
    diabetes_duration_years: Optional[float] = Field(default=None, ge=0)
    hba1c: Optional[float] = Field(default=None, ge=0, le=20)
    prior_screenings: List[PriorScreening] = Field(default_factory=list)


class ImageQuality(BaseModel):
    status: str
    technically_valid: bool
    requires_human_review: bool = True
    width: Optional[int] = None
    height: Optional[int] = None
    issues: List[str] = Field(default_factory=list)


class GradingResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    dr_grade: int = Field(ge=0, le=4)
    dr_label: str
    confidence: float = Field(ge=0, le=1)
    probabilities: Dict[str, float] = Field(default_factory=dict)
    model_version: str


class Lesion(BaseModel):
    key: str
    label: str
    detected: bool = False
    area_pct: float = Field(default=0, ge=0)
    confidence: Optional[float] = Field(default=None, ge=0, le=1)


class SegmentationResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    lesions: List[Lesion] = Field(default_factory=list)
    lesion_mask_url: Optional[str] = None
    model_version: str
    status: str = "ok"


class EyeClinicalAssessment(BaseModel):
    eye: str
    quality: Dict[str, ImageQuality]
    grading: GradingResult
    segmentation: SegmentationResult
    review_priority: str
    follow_up_window: str
    referral: str
    findings: List[str]
    actions: List[str]
    safety_flags: List[str]


class ScreeningAssessment(BaseModel):
    status: str = "ok"
    left_eye: Optional[EyeClinicalAssessment] = None
    right_eye: Optional[EyeClinicalAssessment] = None
    overall_priority: str
    clinical_recommendation: str
    guideline_ids: List[str]
    disclaimer: str


class GradingPort(Protocol):
    async def predict(self, image_bytes: bytes, eye: str) -> GradingResult: ...


class SegmentationPort(Protocol):
    async def predict(self, image_bytes: bytes, eye: str) -> SegmentationResult: ...
