from __future__ import annotations

import asyncio
import unittest
from io import BytesIO

from PIL import Image

from app.clinical.analysis import ClinicalAnalysisModule, InvalidFundusSet
from app.clinical.models import (
    ClinicalContext,
    EyeImageSet,
    GradingResult,
    Lesion,
    SegmentationResult,
)
from app.clinical.report import clinical_report_pdf, screening_report_pdf


def valid_fundus_bytes() -> bytes:
    image = Image.effect_noise((600, 600), 60).convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class FakeGradingAdapter:
    def __init__(self, grades=None, confidence=0.92):
        self.grades = grades or {"L": 2, "R": 1}
        self.confidence = confidence

    async def predict(self, image_bytes: bytes, eye: str) -> GradingResult:
        grade = self.grades[eye]
        return GradingResult(
            dr_grade=grade,
            dr_label=f"Grade {grade}",
            confidence=self.confidence,
            probabilities={str(grade): self.confidence},
            model_version="test-grader",
        )


class FakeSegmentationAdapter:
    def __init__(self, *, hard_exudate=False):
        self.hard_exudate = hard_exudate

    async def predict(self, image_bytes: bytes, eye: str) -> SegmentationResult:
        return SegmentationResult(
            lesions=[
                Lesion(
                    key="hard_exudate",
                    label="Hard exudate",
                    detected=self.hard_exudate,
                    area_pct=5.0 if self.hard_exudate else 0,
                )
            ],
            model_version="test-segmenter",
        )


class ClinicalAnalysisTests(unittest.TestCase):
    def analyze(self, module, context=None):
        image = valid_fundus_bytes()
        return asyncio.run(
            module.analyze(
                EyeImageSet("L", image),
                EyeImageSet("R", image),
                context or ClinicalContext(),
            )
        )

    def test_requires_one_technically_valid_image_per_eye(self):
        module = ClinicalAnalysisModule(FakeGradingAdapter(), FakeSegmentationAdapter())
        image = valid_fundus_bytes()
        with self.assertRaises(InvalidFundusSet):
            asyncio.run(
                module.analyze(
                    EyeImageSet("L", b"not-an-image"),
                    EyeImageSet("R", image),
                    ClinicalContext(),
                )
            )

    def test_can_analyze_only_one_eye(self):
        image = valid_fundus_bytes()
        result = asyncio.run(
            ClinicalAnalysisModule(FakeGradingAdapter(), FakeSegmentationAdapter()).analyze(
                EyeImageSet("L", image),
                None,
                ClinicalContext(),
            )
        )
        self.assertIsNotNone(result.left_eye)
        self.assertIsNone(result.right_eye)
        self.assertEqual(result.left_eye.eye, "L")

    def test_rejects_screening_without_any_eye(self):
        with self.assertRaises(InvalidFundusSet):
            asyncio.run(
                ClinicalAnalysisModule(FakeGradingAdapter(), FakeSegmentationAdapter()).analyze(
                    None,
                    None,
                    ClinicalContext(),
                )
            )

    def test_quality_is_never_automatically_marked_good(self):
        result = self.analyze(ClinicalAnalysisModule(FakeGradingAdapter(), FakeSegmentationAdapter()))
        self.assertEqual(result.left_eye.quality["fundus"].status, "ReviewRequired")
        self.assertTrue(result.left_eye.quality["fundus"].requires_human_review)

    def test_hard_exudate_does_not_create_a_macular_assessment(self):
        result = self.analyze(
            ClinicalAnalysisModule(
                FakeGradingAdapter(grades={"L": 0, "R": 0}),
                FakeSegmentationAdapter(hard_exudate=True),
            )
        )
        self.assertEqual(result.left_eye.review_priority, "routine")
        serialized = str(result.left_eye.model_dump())
        self.assertNotIn("macular", serialized.lower())
        self.assertNotIn("hoàng điểm", serialized.lower())

    def test_pdr_does_not_use_hemorrhage_area_as_24_hour_proxy(self):
        result = self.analyze(
            ClinicalAnalysisModule(
                FakeGradingAdapter(grades={"L": 4, "R": 0}),
                FakeSegmentationAdapter(),
            )
        )
        self.assertEqual(result.left_eye.follow_up_window, "Dưới 1 tháng")
        self.assertNotIn("24", result.left_eye.referral)
        self.assertIn("specialist_confirmation_required", result.left_eye.safety_flags)

    def test_low_confidence_is_review_flag_not_diagnosis(self):
        result = self.analyze(
            ClinicalAnalysisModule(FakeGradingAdapter(confidence=0.4), FakeSegmentationAdapter())
        )
        self.assertIn("low_ai_confidence", result.left_eye.safety_flags)

    def test_automatic_pdf_contains_valid_pdf_header(self):
        result = self.analyze(ClinicalAnalysisModule(FakeGradingAdapter(), FakeSegmentationAdapter()))
        payload = result.model_dump(mode="json")
        pdf = clinical_report_pdf({"patient_code": "BN01", "full_name": "Nguyễn Văn A"}, payload)
        self.assertTrue(pdf.startswith(b"%PDF"))

    def test_stored_screening_pdf_contains_valid_pdf_header(self):
        pdf = screening_report_pdf({
            "screening_id": 22,
            "screening_date": "2026-08-20T18:25:35+07:00",
            "status": "Reviewed",
            "patient": {"patient_code": "BN01", "full_name": "Nguyễn Văn A"},
            "doctor_name": "Bác sĩ Nguyễn",
            "eyes": [{
                "eye": "L",
                "ai_result": {
                    "dr_grade": 2,
                    "dr_label": "DR không tăng sinh trung bình",
                    "confidence": 0.91,
                    "model_version": "grading-v1",
                },
                "doctor_review": {
                    "final_dr_grade": 2,
                    "final_dr_label": "DR không tăng sinh trung bình",
                    "is_agree_with_ai": True,
                    "clinical_notes": "Theo dõi định kỳ.",
                },
                "lesions": [{
                    "label": "Vi phình mạch (MA)",
                    "detected": True,
                    "area_pct": 0.12,
                }],
            }],
            "recall": {
                "recall_date": "2027-02-20",
                "risk_stratification": "Medium",
                "recommendation": "Tái khám sau 6 tháng.",
                "status": "Scheduled",
            },
        })

        self.assertTrue(pdf.startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
