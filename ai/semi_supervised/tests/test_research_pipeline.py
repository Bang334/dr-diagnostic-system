import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ai.semi_supervised.few_shot_demo import (
    EpisodeSampler,
    RetfoundProtoNet,
    parse_args as parse_few_shot_args,
)
from ai.semi_supervised.research_utils import (
    assert_unlabeled_is_external,
    load_grading_checkpoint,
    prepare_fresh_output_dir,
)
from ai.semi_supervised.semi_supervised_training import (
    generate_pseudo_labels,
    parse_args as parse_semi_args,
)


class ArgumentDefaultTests(unittest.TestCase):
    def test_semi_supervised_defaults_are_conservative(self):
        args = parse_semi_args(
            [
                "--checkpoint",
                "best.pth",
                "--dataset-dir",
                "dataset",
                "--unlabeled-dir",
                "unlabeled",
                "--output-dir",
                "output",
            ]
        )
        self.assertEqual(args.threshold, 0.95)
        self.assertEqual(args.pseudo_weight, 0.25)
        self.assertEqual(args.head_lr, 1e-5)
        self.assertEqual(args.backbone_lr, 1e-6)
        self.assertEqual(args.patience, 3)

    def test_few_shot_defaults_only_unfreeze_last_block(self):
        args = parse_few_shot_args(
            [
                "--checkpoint",
                "best.pth",
                "--dataset-dir",
                "dataset",
                "--output-dir",
                "output",
            ]
        )
        self.assertEqual(args.shots, 5)
        self.assertEqual(args.queries, 3)
        self.assertEqual(args.unfreeze_last_blocks, 1)
        self.assertEqual(args.encoder_lr, 1e-6)


class CheckpointLoadingTests(unittest.TestCase):
    def test_reconstructs_model_from_saved_args_without_pretrained_download(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "checkpoint-best.pth"
            original_model = nn.Linear(2, 5)
            torch.save(
                {
                    "model": original_model.state_dict(),
                    "args": {
                        "loss": "ce",
                        "model_source": "timm",
                        "model_name": "toy_model",
                        "image_size": 224,
                    },
                },
                checkpoint_path,
            )
            reconstructed = nn.Linear(2, 5)
            with patch(
                "ai.semi_supervised.research_utils.timm.create_model",
                return_value=reconstructed,
            ) as create_model:
                bundle = load_grading_checkpoint(
                    checkpoint_path, torch.device("cpu"), require_ce=True
                )

        create_model.assert_called_once_with(
            "toy_model", pretrained=False, num_classes=5
        )
        for expected, actual in zip(
            original_model.parameters(), bundle.model.parameters()
        ):
            torch.testing.assert_close(expected, actual)


class DataSeparationTests(unittest.TestCase):
    def test_rejects_unlabeled_path_from_any_fixed_split(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            train_image = root / "train" / "image.jpg"
            train_image.parent.mkdir()
            train_image.write_bytes(b"image")
            frames = {
                "train": pd.DataFrame({"image_path": [str(train_image)]}),
                "val": pd.DataFrame({"image_path": []}),
                "test": pd.DataFrame({"image_path": []}),
            }
            with self.assertRaisesRegex(ValueError, "overlaps"):
                assert_unlabeled_is_external([train_image], frames)

    def test_accepts_external_unlabeled_path(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            labeled = root / "train.jpg"
            external = root / "external.jpg"
            labeled.write_bytes(b"labeled")
            external.write_bytes(b"external")
            frames = {
                "train": pd.DataFrame({"image_path": [str(labeled)]}),
                "val": pd.DataFrame({"image_path": []}),
                "test": pd.DataFrame({"image_path": []}),
            }
            assert_unlabeled_is_external([external], frames)

    def test_rejects_copied_split_image_after_zip_extraction(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            labeled = root / "labeled" / "same.jpg"
            copied = root / "unzipped" / "same.jpg"
            labeled.parent.mkdir()
            copied.parent.mkdir()
            labeled.write_bytes(b"identical-image-bytes")
            copied.write_bytes(b"identical-image-bytes")
            frames = {
                "train": pd.DataFrame({"image_path": [str(labeled)]}),
                "val": pd.DataFrame({"image_path": []}),
                "test": pd.DataFrame({"image_path": []}),
            }
            with self.assertRaisesRegex(ValueError, "byte-identical"):
                assert_unlabeled_is_external([copied], frames)

    def test_refuses_nonempty_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir) / "run"
            output.mkdir()
            (output / "history.jsonl").write_text("old run", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "not empty"):
                prepare_fresh_output_dir(output)


class EpisodeSamplerTests(unittest.TestCase):
    def test_samples_balanced_disjoint_support_and_query_sets(self):
        frame = pd.DataFrame(
            {
                "diagnosis": [grade for grade in range(5) for _ in range(10)]
            }
        )
        sampler = EpisodeSampler(frame, seed=42)
        support, support_labels, query, query_labels = sampler.sample(2, 3)
        self.assertEqual(len(support), 10)
        self.assertEqual(len(query), 15)
        self.assertTrue(set(support).isdisjoint(query))
        self.assertEqual([support_labels.count(grade) for grade in range(5)], [2] * 5)
        self.assertEqual([query_labels.count(grade) for grade in range(5)], [3] * 5)

    def test_rejects_class_with_too_few_images(self):
        frame = pd.DataFrame(
            {"diagnosis": [grade for grade in range(5) for _ in range(2)]}
        )
        with self.assertRaisesRegex(ValueError, "episode needs"):
            EpisodeSampler(frame, seed=1).sample(2, 1)


class _ToyEncoder(nn.Module):
    num_features = 4

    def forward_features(self, images):
        return images.mean(dim=(2, 3))[:, :4]

    def forward_head(self, features, pre_logits=False):
        return features


class ProtoNetTests(unittest.TestCase):
    def test_returns_one_logit_per_support_class(self):
        model = RetfoundProtoNet(
            _ToyEncoder(), embedding_dim=3, temperature=0.1, forward_batch_size=2
        )
        support_images = torch.randn(10, 4, 2, 2)
        support_labels = torch.tensor([grade for grade in range(5) for _ in range(2)])
        query_images = torch.randn(5, 4, 2, 2)
        logits, class_ids = model.episode_logits(
            support_images, support_labels, query_images
        )
        self.assertEqual(tuple(logits.shape), (5, 5))
        self.assertEqual(class_ids.tolist(), list(range(5)))


class _PseudoDataset(Dataset):
    def __init__(self, root):
        self.root = root

    def __len__(self):
        return 2

    def __getitem__(self, index):
        value = 1.0 if index == 0 else 0.0
        return torch.full((3, 2, 2), value), str(self.root / f"{index}.jpg")


class _ConfidenceModel(nn.Module):
    def forward(self, images):
        logits = torch.zeros(images.size(0), 5, device=images.device)
        strong = images[:, 0, 0, 0] > 0.5
        logits[strong, 2] = 10.0
        return logits


class PseudoLabelTests(unittest.TestCase):
    def test_keeps_only_predictions_above_threshold(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            loader = DataLoader(_PseudoDataset(Path(temporary_dir)), batch_size=2)
            frame = generate_pseudo_labels(
                _ConfidenceModel(),
                loader,
                torch.device("cpu"),
                threshold=0.95,
                max_per_class=0,
                amp_enabled=False,
            )
        self.assertEqual(len(frame), 1)
        self.assertEqual(int(frame.iloc[0]["pseudo_label"]), 2)
        self.assertGreater(float(frame.iloc[0]["confidence"]), 0.95)


if __name__ == "__main__":
    unittest.main()
