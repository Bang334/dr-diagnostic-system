import argparse
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from ai.train_fewshot.train import (
    FixedSupportEpisodeSampler,
    parse_args,
    prepare_output_dir,
    save_adapted_checkpoint,
    save_metrics,
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
            checkpoint = output / "last.pth"
            checkpoint.write_bytes(b"checkpoint")
            self.assertEqual(prepare_output_dir(output, checkpoint), output.resolve())
            outside = root / "outside.pth"
            outside.write_bytes(b"checkpoint")
            with self.assertRaisesRegex(ValueError, "inside --output-dir"):
                prepare_output_dir(output, outside)


class FewShotArtifactTests(unittest.TestCase):
    def test_metrics_csv_keeps_epochs_and_replaces_before_after_scores(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "metrics.csv"
            save_metrics(
                path,
                [{"stage": "epoch", "epoch": 1, "loss": 0.5, "best_loss": 0.5}],
            )
            save_metrics(
                path,
                [{"stage": "epoch", "epoch": 2, "loss": 0.4, "best_loss": 0.4}],
            )
            save_metrics(
                path,
                [
                    {"stage": "before", "accuracy": 0.6, "macro_f1": 0.5},
                    {"stage": "after", "accuracy": 0.7, "macro_f1": 0.65},
                    {"stage": "delta", "accuracy": 0.1, "macro_f1": 0.15},
                ],
                replace_stages={"before", "after", "delta"},
            )
            save_metrics(
                path,
                [
                    {"stage": "before", "accuracy": 0.61, "macro_f1": 0.51},
                    {"stage": "after", "accuracy": 0.71, "macro_f1": 0.66},
                    {"stage": "delta", "accuracy": 0.1, "macro_f1": 0.15},
                ],
                replace_stages={"before", "after", "delta"},
            )

            metrics = pd.read_csv(path)
            self.assertEqual(
                metrics["stage"].tolist(),
                ["epoch", "epoch", "before", "after", "delta"],
            )
            self.assertEqual(metrics.loc[:1, "epoch"].tolist(), [1.0, 2.0])
            self.assertAlmostEqual(
                metrics.loc[metrics["stage"] == "after", "accuracy"].item(),
                0.71,
            )

    def test_checkpoint_embeds_support_instead_of_requiring_an_extra_file(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)

            class Model:
                encoder = nn.Linear(2, 2)
                projection = nn.Identity()
                embedding_dim = 2
                temperature = 0.1

            support = pd.DataFrame(
                {"image_path": ["0.jpg", "1.jpg"], "diagnosis": [0, 1]}
            )
            args = argparse.Namespace(checkpoint=root / "base.pth", shots=1)
            path = root / "last.pth"
            save_adapted_checkpoint(
                path,
                Model(),
                torch.zeros(2, 2),
                torch.tensor([0, 1]),
                args,
                argparse.Namespace(image_size=224),
                support,
                epoch=0,
                best_support_loss=0.5,
                stale_epochs=0,
            )

            state = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(state["support_rows"], support.to_dict(orient="records"))
            self.assertNotIn("support_manifest", state)


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
