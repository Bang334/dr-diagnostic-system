from __future__ import annotations

from dataclasses import dataclass
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
    """Minimum photographic set in QD 2557/QD-BYT: disc and posterior pole."""

    eye: str
    disc_image: bytes
    posterior_pole_image: bytes


class ClinicalContext(BaseModel):
    patient_code: Optional[str] = None
    diabetes_duration_years: Optional[float] = Field(default=None, ge=0)
    hba1c: Optional[float] = Field(default=None, ge=0, le=20)
    systolic_bp: Optional[int] = Field(default=None, ge=40, le=300)
    diastolic_bp: Optional[int] = Field(default=None, ge=20, le=200)
    visual_acuity_left: Optional[float] = Field(default=None, ge=0, le=2)
    visual_acuity_right: Optional[float] = Field(default=None, ge=0, le=2)
    sudden_vision_loss: bool = False
    pregnant: bool = False
    kidney_disease: bool = False

    def visual_acuity_for(self, eye: str) -> Optional[float]:
        return self.visual_acuity_left if eye == "L" else self.visual_acuity_right


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
    # Optional spatial evidence. Area alone must never be used to diagnose DME.
    distance_to_fovea_mm: Optional[float] = Field(default=None, ge=0)


class SegmentationResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    lesions: List[Lesion] = Field(default_factory=list)
    lesion_mask_url: Optional[str] = None
    model_version: str
    status: str = "ok"
    retinal_thickening_confirmed: Optional[bool] = None
    center_involved_confirmed_by_oct: Optional[bool] = None


class EyeClinicalAssessment(BaseModel):
    eye: str
    quality: Dict[str, ImageQuality]
    grading: GradingResult
    segmentation: SegmentationResult
    review_priority: str
    follow_up_window: str
    referral: str
    macular_status: str
    findings: List[str]
    actions: List[str]
    safety_flags: List[str]
    review_status: str = "draft"


class ScreeningAssessment(BaseModel):
    status: str = "ok"
    review_status: str = "draft"
    left_eye: EyeClinicalAssessment
    right_eye: EyeClinicalAssessment
    overall_priority: str
    clinical_recommendation: str
    guideline_ids: List[str]
    disclaimer: str


class GradingPort(Protocol):
    async def predict(self, image_bytes: bytes, eye: str) -> GradingResult: ...


class SegmentationPort(Protocol):
    async def predict(self, image_bytes: bytes, eye: str) -> SegmentationResult: ...
