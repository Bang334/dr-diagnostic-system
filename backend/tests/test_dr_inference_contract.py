import os
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from main import app
from app.services.dr_inference import (
    DRInferenceService,
    InvalidFundusImage,
)


BACKEND_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = BACKEND_DIR / "checkpoint-best.pth"


class DRInferenceContractTests(unittest.TestCase):
    def test_invalid_image_is_rejected_before_model_loading(self):
        service = DRInferenceService(CHECKPOINT_PATH)

        with self.assertRaises(InvalidFundusImage):
            service.predict(b"not-an-image")

        self.assertFalse(service.loaded)

    def test_prediction_response_contains_all_five_icdr_probabilities(self):
        result = DRInferenceService.format_result(
            [0.05, 0.10, 0.60, 0.20, 0.05],
            preview_b64="data:image/png;base64,preview",
            model_version="test-model",
            device="cpu",
        )

        self.assertEqual(result["dr_grade"], 2)
        self.assertEqual(result["dr_label"], "Moderate NPDR")
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=5)
        self.assertEqual(len(result["probabilities"]), 5)
        self.assertEqual(result["model_version"], "test-model")

    @unittest.skipUnless(
        os.getenv("RUN_DR_MODEL_TEST") == "1",
        "Set RUN_DR_MODEL_TEST=1 to load the 1.2 GB production checkpoint.",
    )
    def test_real_checkpoint_loads_with_exact_architecture(self):
        service = DRInferenceService(CHECKPOINT_PATH, device="cpu")
        service.load()

        info = service.model_info()
        self.assertTrue(info["loaded"])
        self.assertEqual(info["architecture"], "vit_large_patch14_dinov2.lvd142m")
        self.assertEqual(info["image_size"], 224)
        self.assertEqual(info["classes"], 5)


class DiagnosisAPIContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_multipart_upload_returns_prediction_contract(self):
        expected = DRInferenceService.format_result(
            [0.05, 0.10, 0.60, 0.20, 0.05],
            preview_b64="data:image/png;base64,preview",
            model_version="fake-model",
            device="cpu",
        )

        class FakeService:
            def predict(self, content):
                self.content = content
                return expected

        transport = httpx.ASGITransport(app=app)
        with patch(
            "app.api.diagnosis.get_dr_inference_service",
            return_value=FakeService(),
        ):
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://test",
            ) as client:
                response = await client.post(
                    "/api/v1/diagnosis/analyze",
                    files={"file": ("fundus.jpg", b"jpeg-bytes", "image/jpeg")},
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["dr_grade"], 2)
        self.assertEqual(len(response.json()["probabilities"]), 5)


if __name__ == "__main__":
    unittest.main()
