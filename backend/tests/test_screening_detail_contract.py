import unittest
from types import SimpleNamespace

from fastapi import HTTPException
from main import app
from app.api.screenings import _require_screening_detail_access, _screening_eye_detail


class ScreeningDetailContractTests(unittest.TestCase):
    def test_openapi_exposes_model_selection_for_screening_upload(self):
        spec = app.openapi()
        self.assertIn("/api/v1/screenings/models", spec["paths"])
        upload = spec["paths"]["/api/v1/screenings/upload"]["post"]
        body_ref = upload["requestBody"]["content"]["multipart/form-data"]["schema"]["$ref"]
        body_schema = spec["components"]["schemas"][body_ref.rsplit("/", 1)[-1]]
        self.assertIn("grading_model", body_schema["properties"])

    def test_openapi_exposes_screening_detail_with_ai_and_doctor_results(self):
        spec = app.openapi()
        path = spec["paths"].get("/api/v1/screenings/{screening_id}")

        self.assertIsNotNone(path)
        response = path["get"]["responses"]["200"]["content"]["application/json"]["schema"]
        self.assertEqual(
            response["$ref"],
            "#/components/schemas/ScreeningDetailResponse",
        )

        schemas = spec["components"]["schemas"]
        eye_properties = schemas["ScreeningEyeDetail"]["properties"]
        self.assertIn("ai_result", eye_properties)
        self.assertIn("doctor_review", eye_properties)
        upload_eye_properties = schemas["EyeAnalysisResponse"]["properties"]
        self.assertNotIn("macular_status", upload_eye_properties)
        detail_properties = schemas["ScreeningDetailResponse"]["properties"]
        self.assertIn("results_visible", detail_properties)
        self.assertIn("diabetes_type", detail_properties)
        self.assertIn("diabetes_duration_years", detail_properties)
        self.assertIn("latest_hba1c", detail_properties)

    def test_patient_can_view_only_own_screening_at_any_status(self):
        account = SimpleNamespace(
            role="patient",
            patient_profile=SimpleNamespace(id=7),
        )

        _require_screening_detail_access(
            account,
            SimpleNamespace(patient_id=7, status="Reviewed"),
        )

        _require_screening_detail_access(
            account,
            SimpleNamespace(patient_id=7, status="AI_Analyzed"),
        )

        with self.assertRaises(HTTPException) as another_patient:
            _require_screening_detail_access(
                account,
                SimpleNamespace(patient_id=8, status="Reviewed"),
            )
        self.assertEqual(another_patient.exception.status_code, 404)

    def test_pending_patient_detail_keeps_image_but_hides_clinical_results(self):
        screening = SimpleNamespace(
            left_eye_image_url="https://example.test/left.png",
            right_eye_image_url=None,
            ai_results=[SimpleNamespace(eye="L")],
            segmentation_results=[SimpleNamespace(eye="L")],
            reviews=[],
        )

        detail = _screening_eye_detail(screening, "L", include_results=False)

        self.assertEqual(detail["image_url"], "https://example.test/left.png")
        self.assertIsNone(detail["ai_result"])
        self.assertIsNone(detail["segmentation"])
        self.assertIsNone(detail["doctor_review"])


if __name__ == "__main__":
    unittest.main()
