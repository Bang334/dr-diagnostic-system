from __future__ import annotations

import asyncio
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import httpx

from app.clinical.models import (
    ClinicalContext,
    EyeClinicalAssessment,
    GradingResult,
    ImageQuality,
    PriorEyeFinding,
    PriorScreening,
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

    def test_outbound_payload_includes_at_most_two_prior_screenings(self):
        prior_screenings = [
            PriorScreening(
                screening_date=datetime(2026, month, 1, tzinfo=timezone.utc),
                eyes=[
                    PriorEyeFinding(
                        eye="L",
                        dr_grade=grade,
                        dr_label=f"Grade {grade}",
                        confidence=0.8,
                        detected_lesions=["MA"],
                    )
                ],
            )
            for month, grade in ((7, 2), (5, 1), (3, 0))
        ]
        payload = _safe_clinical_payload(
            ClinicalContext(patient_code="BN-SECRET", prior_screenings=prior_screenings),
            assessment(),
        )

        self.assertEqual(len(payload["prior_screenings"]), 2)
        self.assertEqual(payload["prior_screenings"][0]["eyes"][0]["dr_grade"], 2)
        self.assertNotIn("BN-SECRET", str(payload))

        summary = build_rule_summary(
            ClinicalContext(prior_screenings=prior_screenings),
            assessment(),
        )
        self.assertTrue(any("2026-07-01" in item for item in summary.diabetes_evidence))
        self.assertIn("suy luận gián tiếp", summary.diabetes_assessment)

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
        self.assertEqual(result.diabetes_assessment_level, "known_diabetes")
        self.assertIn("Type 2", result.diabetes_assessment)
        self.assertIn("4 năm", result.diabetes_assessment)
        self.assertIn("HbA1c gần nhất là 6.5%", result.diabetes_assessment)
        self.assertIn("Grade 2", result.diabetes_assessment)
        self.assertTrue(any("Grade 2" in item for item in result.diabetes_evidence))

    def test_invalid_proxy_url_returns_safe_local_draft(self):
        with patch(
            "app.clinical.summary.httpx.AsyncClient",
            side_effect=httpx.InvalidURL("Invalid port: ':1'"),
        ) as async_client:
            result = asyncio.run(
                GeminiClinicalSummaryAdapter(api_key="configured-key").generate(
                    ClinicalContext(
                        diabetes_type="Type 2",
                        diabetes_duration_years=4,
                        hba1c=6.5,
                    ),
                    assessment(),
                )
            )

        self.assertEqual(result.status, "fallback")
        self.assertEqual(result.provider, "local-rules")
        async_client.assert_called_once_with(timeout=45.0, trust_env=False)

    def test_hba1c_range_is_primary_and_grade_is_supporting_evidence(self):
        result = build_rule_summary(ClinicalContext(hba1c=6.7), assessment())
        self.assertEqual(result.diabetes_assessment_level, "high")
        self.assertIn("HbA1c 6.7%", result.diabetes_assessment)
        self.assertIn("Grade 2", result.diabetes_assessment)
        self.assertIn("bằng chứng", result.diabetes_assessment)
        self.assertTrue(
            any("không tự xác nhận" in item for item in result.diabetes_evidence)
        )

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
        self.assertIn("đánh giá tình trạng đái tháo đường chủ yếu", prompt)
        self.assertIn("làm bằng chứng bổ sung", prompt)
        self.assertIn("không dùng chúng độc lập", prompt)
        self.assertIn("một đoạn văn liền mạch", prompt)
        self.assertIn("diabetes_assessment_level đúng một trong các mã", prompt)
        self.assertIn("nguy cơ sàng lọc cao", prompt)
        self.assertIn("không tự suy đoán loại bệnh", prompt)
        self.assertIn("1–2 lần khám trước", prompt)
        self.assertIn("xét nghiệm xác nhận", prompt)
        self.assertIn("nhận định xu hướng kiểm soát đường huyết ở mức gợi ý", prompt)
        self.assertIn("đây là suy luận gián tiếp", prompt)


if __name__ == "__main__":
    unittest.main()
