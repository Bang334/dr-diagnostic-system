"""Canonical ICDR grades and their ETDRS correspondence for DR grading."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DRGrade:
    grade: int
    icdr_label: str
    etdrs_range: str
    clinical_definition: str


DR_GRADES = (
    DRGrade(0, "No DR", "10", "No apparent diabetic retinopathy"),
    DRGrade(1, "Mild NPDR", "20", "Microaneurysms only"),
    DRGrade(2, "Moderate NPDR", "35, 43, 47", "More than microaneurysms, less than severe NPDR"),
    DRGrade(3, "Severe NPDR", "53", "Severe NPDR (4-2-1 rule), without proliferative signs"),
    DRGrade(4, "Proliferative DR", "61-85", "Neovascularization and/or preretinal/vitreous hemorrhage"),
)

NUM_CLASSES = len(DR_GRADES)
CLASS_NAMES = [item.icdr_label for item in DR_GRADES]


def grade_definition(grade: int) -> DRGrade:
    if not isinstance(grade, int) or not 0 <= grade < NUM_CLASSES:
        raise ValueError(f"DR grade must be an integer in [0, {NUM_CLASSES - 1}]")
    return DR_GRADES[grade]
