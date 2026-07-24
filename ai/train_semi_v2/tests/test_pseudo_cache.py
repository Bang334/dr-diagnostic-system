import json
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

    def test_recreated_colab_runtime_reuses_equivalent_cache(self):
        calls = []

        def generate():
            calls.append("called")
            return self.frame()

        first = self.cache.load_or_generate(self.spec(), generate)

        recreated_root = self.root / "new-colab-runtime"
        recreated_root.mkdir()
        recreated_checkpoint = recreated_root / self.checkpoint.name
        recreated_checkpoint.write_bytes(self.checkpoint.read_bytes())
        recreated_images = tuple(
            recreated_root / image.name for image in self.images
        )
        for source, destination in zip(self.images, recreated_images):
            destination.write_bytes(source.read_bytes())

        recreated_spec = PseudoLabelCacheSpec(
            threshold=0.95,
            teacher_checkpoint=recreated_checkpoint,
            unlabeled_paths=recreated_images,
            preprocessing="rgb_crop",
            image_size=224,
            max_pseudo_per_class=0,
            enhance=False,
        )
        second = self.cache.load_or_generate(recreated_spec, generate)

        self.assertFalse(first.reused)
        self.assertTrue(second.reused)
        self.assertEqual(calls, ["called"])

    def test_renamed_schema_v1_files_are_reused_and_migrated(self):
        cache_dir = self.root / "cache"
        cache_dir.mkdir()
        self.frame().to_csv(cache_dir / "labels.csv", index=False)
        old_contract = {
            "schema_version": 1,
            "threshold": 0.95,
            "teacher_checkpoint": {
                "path": str(self.checkpoint.resolve()),
                "size": self.checkpoint.stat().st_size,
                "mtime_ns": self.checkpoint.stat().st_mtime_ns,
            },
            "unlabeled_manifest_sha256": "old-runtime-dependent-digest",
            "unlabeled_image_count": len(self.images),
            "preprocessing": "rgb_crop",
            "image_size": 224,
            "max_pseudo_per_class": 0,
            "enhance": False,
        }
        (cache_dir / "meta.json").write_text(
            json.dumps({"cache_key": "old-key", "contract": old_contract}),
            encoding="utf-8",
        )

        def must_not_generate():
            self.fail("Schema v1 cache should be reused")

        result = self.cache.load_or_generate(self.spec(), must_not_generate)

        self.assertTrue(result.reused)
        migrated = json.loads(result.metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(migrated["contract"]["schema_version"], 2)
        self.assertEqual(result.csv_path.name, "labels.csv")
        self.assertEqual(result.metadata_path.name, "meta.json")


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
            Path("runs/cache"),
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
