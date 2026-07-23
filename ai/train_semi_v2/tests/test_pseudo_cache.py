import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ai.train_semi_v2.pseudo_cache import PseudoLabelCache, PseudoLabelCacheSpec
from ai.train_semi_v2.train import parse_args


class PseudoLabelCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_dir.name)
        self.checkpoint = self.root / "checkpoint-best.pth"
        self.checkpoint.write_bytes(b"teacher-v1")
        self.images = tuple(self.root / f"image-{index}.jpg" for index in range(2))
        for index, image in enumerate(self.images):
            image.write_bytes(f"image-{index}".encode())
        self.cache = PseudoLabelCache(self.root / "cache")

    def tearDown(self):
        self.temporary_dir.cleanup()

    def spec(self, threshold=0.95):
        return PseudoLabelCacheSpec(
            threshold=threshold,
            teacher_checkpoint=self.checkpoint,
            unlabeled_paths=self.images,
            preprocessing="rgb_crop",
            image_size=224,
            max_pseudo_per_class=0,
            enhance=False,
        )

    def frame(self):
        return pd.DataFrame(
            {
                "image_path": [str(self.images[0])],
                "pseudo_label": [2],
                "confidence": [0.98],
            }
        )

    def test_same_contract_reuses_predictions_without_calling_generator(self):
        calls = []

        def generate():
            calls.append("called")
            return self.frame()

        first = self.cache.load_or_generate(self.spec(), generate)
        second = self.cache.load_or_generate(self.spec(), generate)

        self.assertFalse(first.reused)
        self.assertTrue(second.reused)
        self.assertEqual(calls, ["called"])
        pd.testing.assert_frame_equal(first.frame, second.frame)

    def test_threshold_change_creates_a_new_cache_and_runs_prediction(self):
        calls = []

        def generate():
            calls.append("called")
            return self.frame()

        first = self.cache.load_or_generate(self.spec(0.95), generate)
        second = self.cache.load_or_generate(self.spec(0.90), generate)

        self.assertEqual(calls, ["called", "called"])
        self.assertNotEqual(first.cache_key, second.cache_key)

    def test_deleting_cache_csv_forces_prediction_again(self):
        calls = []

        def generate():
            calls.append("called")
            return self.frame()

        first = self.cache.load_or_generate(self.spec(), generate)
        first.csv_path.unlink()
        second = self.cache.load_or_generate(self.spec(), generate)

        self.assertEqual(calls, ["called", "called"])
        self.assertFalse(second.reused)

    def test_teacher_change_invalidates_cache_for_safety(self):
        calls = []

        def generate():
            calls.append("called")
            return self.frame()

        first = self.cache.load_or_generate(self.spec(), generate)
        self.checkpoint.write_bytes(b"teacher-v2-with-different-size")
        second = self.cache.load_or_generate(self.spec(), generate)

        self.assertEqual(calls, ["called", "called"])
        self.assertNotEqual(first.cache_key, second.cache_key)


class TrainV2ArgumentTests(unittest.TestCase):
    def test_default_cache_is_shared_next_to_run_directory(self):
        args = parse_args(
            [
                "--checkpoint",
                "grade/checkpoint-best.pth",
                "--dataset-dir",
                "dataset",
                "--unlabeled-dir",
                "unlabeled",
                "--output-dir",
                "runs/run-01",
            ]
        )
        self.assertEqual(
            args.pseudo_cache_dir,
            Path("runs/pseudo-label-cache-v2"),
        )

    def test_explicit_cache_directory_is_preserved(self):
        args = parse_args(
            [
                "--checkpoint",
                "grade/checkpoint-best.pth",
                "--dataset-dir",
                "dataset",
                "--unlabeled-dir",
                "unlabeled",
                "--output-dir",
                "runs/run-01",
                "--pseudo-cache-dir",
                "drive/shared-cache",
            ]
        )
        self.assertEqual(args.pseudo_cache_dir, Path("drive/shared-cache"))


if __name__ == "__main__":
    unittest.main()
