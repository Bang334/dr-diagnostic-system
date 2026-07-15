"""Optional smoke test for the legacy Keras grading artifact."""

import os
import unittest
from pathlib import Path


class LegacyKerasWeightsTests(unittest.TestCase):
    def test_legacy_weights_load_through_predictor_adapter(self):
        model_path = Path(__file__).resolve().parent / "ai" / "weights" / "dr_grading_model.keras"
        if not model_path.is_file() or model_path.stat().st_size == 0:
            self.skipTest(f"Legacy Keras artifact is not installed: {model_path}")
        try:
            import tensorflow  # noqa: F401
        except ImportError:
            self.skipTest("TensorFlow is not installed in this environment")

        from ai.grading.predictor import KerasEfficientNetPredictor

        predictor = KerasEfficientNetPredictor(os.fspath(model_path))
        self.assertIsNotNone(predictor.model)
        self.assertEqual(len(predictor.info.class_names), 5)


if __name__ == "__main__":
    unittest.main()
