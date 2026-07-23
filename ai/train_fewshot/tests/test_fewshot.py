import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ai.train_fewshot.train import (
    FixedSupportEpisodeSampler,
    parse_args,
    prepare_output_dir,
    select_fixed_support,
    validate_args,
)


class FewShotSelectionTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(
            {
                "image_path": [
                    f"{grade}_{index}.jpg"
                    for grade in range(5)
                    for index in range(10)
                ],
                "diagnosis": [
                    grade for grade in range(5) for _ in range(10)
                ],
            }
        )

    def test_selects_exactly_k_per_grade_deterministically(self):
        first = select_fixed_support(self.frame(), shots=5, seed=42)
        second = select_fixed_support(self.frame(), shots=5, seed=42)
        self.assertEqual(first["diagnosis"].value_counts().sort_index().tolist(), [5] * 5)
        self.assertEqual(first["image_path"].tolist(), second["image_path"].tolist())

    def test_episode_query_never_adds_images_outside_fixed_support(self):
        support = select_fixed_support(self.frame(), shots=5, seed=42)
        sampler = FixedSupportEpisodeSampler(support, seed=42)
        support_indices, _, query_indices, _ = sampler.sample(queries=1)
        self.assertTrue(set(support_indices + query_indices).issubset(set(support.index)))
        self.assertTrue(set(support_indices).isdisjoint(query_indices))


class FewShotArgumentTests(unittest.TestCase):
    def args(self, *extra):
        return parse_args(
            [
                "--checkpoint", "grade.pth",
                "--target-dataset-dir", "deepdrid",
                "--output-dir", "run",
                *extra,
            ]
        )

    def test_defaults_are_small_fixed_support_adaptation(self):
        args = self.args()
        self.assertEqual(args.shots, 5)
        self.assertEqual(args.queries, 1)
        self.assertEqual(args.unfreeze_last_blocks, 1)
        self.assertEqual(args.patience, 3)

    def test_eval_only_requires_resume(self):
        with self.assertRaisesRegex(ValueError, "requires --resume"):
            validate_args(self.args("--eval-only"))

    def test_resume_only_accepts_checkpoint_inside_output(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            output = root / "run"
            output.mkdir()
            checkpoint = output / "checkpoint-last.pth"
            checkpoint.write_bytes(b"checkpoint")
            self.assertEqual(prepare_output_dir(output, checkpoint), output.resolve())
            outside = root / "outside.pth"
            outside.write_bytes(b"checkpoint")
            with self.assertRaisesRegex(ValueError, "inside --output-dir"):
                prepare_output_dir(output, outside)


class DependencyTests(unittest.TestCase):
    def test_package_does_not_import_old_semi_modules(self):
        package = Path(__file__).resolve().parents[1]
        source = "\n".join(
            path.read_text(encoding="utf-8") for path in package.glob("*.py")
        )
        self.assertNotIn("ai.semi_supervised", source)
        self.assertNotIn("ai.train_semi_v2", source)


if __name__ == "__main__":
    unittest.main()
