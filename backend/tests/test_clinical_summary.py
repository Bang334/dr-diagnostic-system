from __future__ import annotations

import asyncio
import unittest

from app.clinical.models import (
    ClinicalContext,
    EyeClinicalAssessment,
    GradingResult,
    ImageQuality,
    ScreeningAssessment,
    SegmentationResult,
)
from app.clinical.summary import (
    GeminiClinicalSummaryAdapter,
    _safe_clinical_payload,
    build_rule_summary,
)


def assessment() -> ScreeningAssessment:
    def eye(side: str, grade: int) -> EyeClinicalAssessment:
        return EyeClinicalAssessment(
            eye=side,
            quality={"fundus": ImageQuality(status="ReviewRequired", technically_valid=True)},
            grading=GradingResult(
                dr_grade=grade,
                dr_label="Moderate NPDR" if grade == 2 else "Mild NPDR",
                confidence=0.91,
                probabilities={str(grade): 0.91},
                model_version="test",
            ),
            segmentation=SegmentationResult(model_version="not-configured", status="not_available"),
            review_priority="prompt",
            follow_up_window="3–6 tháng",
            referral="Bác sĩ xác nhận.",
            macular_status="not_assessed",
            findings=[],
            actions=["Bác sĩ xác nhận kết quả."],
            safety_flags=[],
        )

    return ScreeningAssessment(
        left_eye=eye("L", 2),
        right_eye=eye("R", 1),
        overall_priority="prompt",
        clinical_recommendation="Bác sĩ rà soát.",
        guideline_ids=["ICDR"],
        disclaimer="Dự thảo, không thay thế chẩn đoán của bác sĩ.",
    )


class ClinicalSummaryTests(unittest.TestCase):
    def test_outbound_payload_is_deidentified(self):
        context = ClinicalContext(
            patient_code="BN-SECRET",
            age_years=52,
            gender="Nam",
            diabetes_type="Type 2",
            diabetes_duration_years=4,
            hba1c=6.5,
        )
        payload = _safe_clinical_payload(context, assessment())
        serialized = str(payload)
        self.assertNotIn("BN-SECRET", serialized)
        self.assertNotIn("patient_code", serialized)
        self.assertNotIn("macular_status", serialized)
        self.assertEqual(payload["patient"]["hba1c_percent"], 6.5)

    def test_missing_key_returns_safe_local_draft(self):
        result = asyncio.run(
            GeminiClinicalSummaryAdapter(api_key="").generate(
                ClinicalContext(diabetes_type="Type 2", diabetes_duration_years=4, hba1c=6.5),
                assessment(),
            )
        )
        self.assertEqual(result.status, "fallback")
        self.assertEqual(result.provider, "local-rules")
        self.assertIn("HbA1c", " ".join(result.risk_factors))

    def test_rule_summary_states_recall_window_for_each_eye(self):
        result = build_rule_summary(
            ClinicalContext(diabetes_type="Type 2", diabetes_duration_years=4, hba1c=6.5),
            assessment(),
        )
        self.assertEqual(result.provider, "local-rules")
        self.assertIn("Tái khám mắt trái: 3–6 tháng", result.follow_up)
        self.assertIn("Tái khám mắt phải: 3–6 tháng", result.follow_up)
        self.assertIn("bệnh võng mạc đái tháo đường", result.diagnostic_impression.lower())
        self.assertTrue(result.diagnostic_basis)
        self.assertTrue(
            any(
                "không chẩn đoán hoặc loại trừ đái tháo đường" in item.lower()
                for item in result.diagnostic_limitations
            )
        )

    def test_gemini_prompt_requires_exact_rule_based_recall_windows(self):
        prompt = GeminiClinicalSummaryAdapter._prompt(
            _safe_clinical_payload(ClinicalContext(), assessment())
        )
        self.assertIn("sao chép nguyên văn", prompt)
        self.assertIn("follow_up_window", prompt)
        self.assertIn("không tự rút ngắn hoặc kéo dài", prompt)
        self.assertIn("chỉ hỗ trợ chẩn đoán bệnh võng mạc đái tháo đường", prompt)
        self.assertIn("không chẩn đoán đái tháo đường", prompt)


if __name__ == "__main__":
    unittest.main()
