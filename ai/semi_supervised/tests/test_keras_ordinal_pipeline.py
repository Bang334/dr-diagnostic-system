import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd

from ai.keras_grading.ordinal import (
    DEFAULT_ORDINAL_THRESHOLDS,
    decode_ordinal,
    enforce_monotonic,
    ordinal_to_class_probabilities,
    validate_thresholds,
)
from ai.keras_grading.evaluate import evaluate_samples
from ai.preprocessing.fundus_prep import preprocess_rgb_crop_512_from_bgr
from ai.semi_supervised.keras_pseudo_labels import (
    ordinal_decision_confidence,
    select_pseudo_labels,
)
from ai.semi_supervised.keras_semi_supervised import (
    assert_external,
    parse_args,
    pseudo_cache_signature,
    read_matching_pseudo_cache,
)


class OrdinalDecodingTests(unittest.TestCase):
    def test_decodes_four_boundaries_into_five_grades(self):
        prediction = decode_ordinal([0.90, 0.80, 0.20, 0.10])
        self.assertEqual(prediction.grade, 2)
        self.assertEqual(prediction.label, "Moderate NPDR")
        np.testing.assert_allclose(
            prediction.class_probabilities,
            [0.10, 0.10, 0.60, 0.10, 0.10],
            atol=1e-6,
        )

    def test_repairs_non_monotonic_coral_output(self):
        repaired = enforce_monotonic([0.8, 0.9, 0.4, 0.5])
        np.testing.assert_allclose(repaired, [0.8, 0.8, 0.4, 0.4])
        probabilities = ordinal_to_class_probabilities(repaired)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)
        self.assertTrue(np.all(probabilities >= 0))

    def test_rejects_wrong_threshold_count(self):
        with self.assertRaisesRegex(ValueError, "4 ordinal thresholds"):
            validate_thresholds([0.5, 0.5, 0.5])


class PseudoLabelSelectionTests(unittest.TestCase):
    def test_uses_distance_from_all_calibrated_decisions(self):
        prediction = decode_ordinal([0.90, 0.80, 0.20, 0.10])
        confidence = ordinal_decision_confidence(prediction)
        self.assertGreater(confidence, 0.5)
        self.assertLess(confidence, 0.6)

    def test_filters_and_caps_each_class_deterministically(self):
        predictions = [
            (
                Path(f"grade2-{index}.jpg"),
                decode_ordinal([0.99, 0.99, 0.01, 0.01]),
            )
            for index in range(3)
        ]
        predictions.append(
            (Path("uncertain.jpg"), decode_ordinal(DEFAULT_ORDINAL_THRESHOLDS))
        )
        selected = select_pseudo_labels(
            predictions, minimum_confidence=0.5, max_per_class=2
        )
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(item.grade == 2 for item in selected))
        self.assertNotIn("uncertain.jpg", {item.image_path.name for item in selected})


class TeacherPreprocessingTests(unittest.TestCase):
    def test_rgb_crop_preserves_aspect_ratio_and_pads_to_512(self):
        image = np.zeros((300, 500, 3), dtype=np.uint8)
        cv2.circle(image, (250, 150), 130, (10, 80, 220), -1)
        prepared = preprocess_rgb_crop_512_from_bgr(image)
        self.assertEqual(prepared.shape, (512, 512, 3))
        self.assertEqual(prepared.dtype, np.uint8)
        self.assertGreater(int(prepared[..., 0].max()), int(prepared[..., 2].max()))


class KerasSemiSupervisedSafetyTests(unittest.TestCase):
    def test_defaults_keep_pseudo_labels_lower_weighted(self):
        args = parse_args(
            [
                "--model",
                "teacher.keras",
                "--dataset-dir",
                "dataset",
                "--unlabeled-dir",
                "unlabeled",
                "--output-dir",
                "output",
            ]
        )
        self.assertEqual(args.pseudo_weight, 0.25)
        self.assertEqual(args.pseudo_confidence, 0.50)
        self.assertEqual(args.learning_rate, 1e-5)

    def test_rejects_copied_fixed_split_image(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            labeled = root / "labeled" / "same.jpg"
            copied = root / "external" / "same.jpg"
            labeled.parent.mkdir()
            copied.parent.mkdir()
            labeled.write_bytes(b"identical bytes")
            copied.write_bytes(b"identical bytes")
            frames = {
                "train": pd.DataFrame({"image_path": [str(labeled)]}),
                "val": pd.DataFrame({"image_path": []}),
                "test": pd.DataFrame({"image_path": []}),
            }
            with self.assertRaisesRegex(ValueError, "byte-identical"):
                assert_external([copied], frames)

    def test_reuses_pseudo_labels_only_for_matching_threshold_signature(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            model = root / "teacher.keras"
            image = root / "unlabeled.jpg"
            model.write_bytes(b"model")
            image.write_bytes(b"image")
            args = SimpleNamespace(
                unlabeled_dir=root,
                pseudo_confidence=0.5,
                max_pseudo_per_class=0,
                no_tta=False,
            )
            grader = SimpleNamespace(
                model_path=model,
                thresholds=np.array([0.55, 0.50, 0.435, 0.31]),
            )
            signature = pseudo_cache_signature(args, grader, [image])
            pseudo_path = root / "pseudo_labels.csv"
            cache_path = root / "pseudo-cache.json"
            pd.DataFrame(
                {"image_path": [str(image)], "diagnosis": [2], "confidence": [0.9]}
            ).to_csv(pseudo_path, index=False)
            cache_path.write_text(
                json.dumps({"complete": True, "signature": signature}),
                encoding="utf-8",
            )
            self.assertIsNotNone(
                read_matching_pseudo_cache(cache_path, pseudo_path, signature)
            )
            changed = {**signature, "pseudo_confidence": 0.7}
            self.assertIsNone(
                read_matching_pseudo_cache(cache_path, pseudo_path, changed)
            )


class BaselineProgressLoggingTests(unittest.TestCase):
    def test_logs_immediate_progress_eta_and_running_accuracy(self):
        class FakeModel:
            input_shape = (None, 384, 384, 3)
            output_shape = (None, 4)

        class FakeGrader:
            model = FakeModel()
            load_mode = "fake"
            use_tta = True
            thresholds = DEFAULT_ORDINAL_THRESHOLDS

            @staticmethod
            def predict_path(path):
                return decode_ordinal([0.9, 0.8, 0.2, 0.1])

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            metrics, _ = evaluate_samples(
                FakeGrader(),
                [(Path("one.jpg"), 2), (Path("two.jpg"), 1)],
                progress_every=1,
            )
        log = output.getvalue()
        self.assertIn("BASELINE VALIDATION STARTED", log)
        self.assertIn("[baseline] 1/2", log)
        self.assertIn("ETA=", log)
        self.assertIn("running_accuracy=", log)
        self.assertIn("BASELINE COMPLETED", log)
        self.assertEqual(metrics["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
