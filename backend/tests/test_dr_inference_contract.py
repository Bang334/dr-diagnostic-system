import os
import unittest
from pathlib import Path

from app.services.dr_inference import (
    DRInferenceService,
    InvalidFundusImage,
    InvalidModelSelection,
    normalize_model_key,
)


BACKEND_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = BACKEND_DIR / "checkpoint-best.pth"


class DRInferenceContractTests(unittest.TestCase):
    def test_model_selection_accepts_only_allowlisted_models(self):
        self.assertEqual(normalize_model_key("checkpoint-best.pth"), "grading")
        self.assertEqual(normalize_model_key("best-fewshot.pth"), "fewshot")
        with self.assertRaises(InvalidModelSelection):
            normalize_model_key("../../untrusted.pth")

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

    @unittest.skipUnless(
        os.getenv("RUN_DR_MODEL_TEST") == "1",
        "Set RUN_DR_MODEL_TEST=1 to load the 1.3 GB few-shot checkpoint.",
    )
    def test_real_fewshot_checkpoint_loads_as_protonet(self):
        service = DRInferenceService(
            BACKEND_DIR / "best-fewshot.pth",
            model_key="fewshot",
            device="cpu",
        )
        service.load()

        info = service.model_info()
        self.assertTrue(info["loaded"])
        self.assertEqual(info["checkpoint_kind"], "fewshot_protonet")
        self.assertEqual(info["image_size"], 224)
        self.assertEqual(info["classes"], 5)

if __name__ == "__main__":
    unittest.main()
