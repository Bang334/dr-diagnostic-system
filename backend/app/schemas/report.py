from typing import List

from pydantic import BaseModel


class PrevalenceGroup(BaseModel):
    label: str
    total_screened: int
    dr_detected: int
    prevalence_rate: float


class GradeDistributionItem(BaseModel):
    final_dr_grade: int
    count: int


class EpidemiologyReport(BaseModel):
    success: bool
    by_age_group: List[PrevalenceGroup]
    by_diabetes_duration: List[PrevalenceGroup]
    grade_distribution: List[GradeDistributionItem]
    agreement_rate: float
