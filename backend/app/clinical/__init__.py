"""Clinical decision-support module for diabetic-retinopathy screening.

The public interface is :class:`ClinicalAnalysisModule`.  Everything returned
by this package is a screening draft that requires clinician confirmation.
"""

from app.clinical.analysis import ClinicalAnalysisModule
from app.clinical.models import ClinicalContext, EyeImageSet

__all__ = ["ClinicalAnalysisModule", "ClinicalContext", "EyeImageSet"]
