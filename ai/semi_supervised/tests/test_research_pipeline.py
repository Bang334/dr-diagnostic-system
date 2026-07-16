import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ai.semi_supervised.few_shot_demo import (
    FixedSupportEpisodeSampler,
    RetfoundProtoNet,
    parse_args as parse_few_shot_args,
    select_fixed_support,
)
from ai.semi_supervised.research_utils import (
    assert_unlabeled_is_external,
    load_grading_checkpoint,
    load_split_frames,
    prepare_deepdrid_target,
    prepare_fresh_output_dir,
    save_classifier_checkpoint,
)
from ai.semi_supervised.semi_supervised_training import (
    generate_pseudo_labels,
    limit_labeled_replay,
    limit_unlabeled_paths,
    parse_args as parse_semi_args,
    prepare_output_dir,
    resume_training_state,
    trim_history_for_resume,
    validate_args as validate_semi_args,
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
        self.assertEqual(args.max_unlabeled_images, 20_000)
        self.assertEqual(args.max_labeled_per_class, 1_000)
        self.assertIsNone(args.resume)
        self.assertFalse(args.eval_only)

    def test_eval_only_requires_resume_checkpoint(self):
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
                "--eval-only",
            ]
        )
        with self.assertRaisesRegex(ValueError, "requires --resume"):
            validate_semi_args(args)

    def test_few_shot_defaults_only_unfreeze_last_block(self):
        args = parse_few_shot_args(
            [
                "--checkpoint",
                "best.pth",
                "--target-dataset-dir",
                "dataset",
                "--output-dir",
                "output",
            ]
        )
        self.assertEqual(args.shots, 5)
        self.assertEqual(args.queries, 1)
        self.assertEqual(args.embedding_dim, 0)
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

    def test_resume_accepts_nonempty_matching_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir) / "run"
            output.mkdir()
            checkpoint = output / "checkpoint-last.pth"
            checkpoint.write_bytes(b"checkpoint")
            prepared = prepare_output_dir(output, checkpoint)
        self.assertEqual(prepared, output.resolve())


class ReplaySamplingTests(unittest.TestCase):
    def test_limits_labeled_replay_per_class_deterministically(self):
        frame = pd.DataFrame(
            {
                "image_path": [
                    f"{grade}_{index}.jpg"
                    for grade in range(5)
                    for index in range(7)
                ],
                "diagnosis": [grade for grade in range(5) for _ in range(7)],
            }
        )
        first = limit_labeled_replay(frame, max_per_class=3, seed=42)
        second = limit_labeled_replay(frame, max_per_class=3, seed=42)
        self.assertEqual(len(first), 15)
        self.assertEqual(
            first["diagnosis"].value_counts().sort_index().tolist(), [3] * 5
        )
        self.assertEqual(first["image_path"].tolist(), second["image_path"].tolist())

    def test_limits_unlabeled_scan_deterministically(self):
        paths = [Path(f"image_{index:05d}.jpg") for index in range(100)]
        first = limit_unlabeled_paths(paths, max_images=20, seed=42)
        second = limit_unlabeled_paths(paths, max_images=20, seed=42)
        self.assertEqual(len(first), 20)
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 20)


class ResumeTests(unittest.TestCase):
    def test_restores_epoch_patience_best_epoch_and_training_states(self):
        model = nn.Linear(2, 5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

        class _Scaler:
            def __init__(self):
                self.loaded = None

            def state_dict(self):
                return {"scale": 128.0}

            def load_state_dict(self, state):
                self.loaded = state

        scaler = _Scaler()
        state = {
            "epoch": 2,
            "best_qwk": 0.91,
            "best_epoch": 1,
            "stale_epochs": 1,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": {"scale": 64.0},
        }
        start_epoch, best_qwk, best_epoch, stale_epochs = resume_training_state(
            state, optimizer, scheduler, scaler
        )
        self.assertEqual(start_epoch, 3)
        self.assertEqual(best_qwk, 0.91)
        self.assertEqual(best_epoch, 1)
        self.assertEqual(stale_epochs, 1)
        self.assertEqual(scaler.loaded, {"scale": 64.0})

    def test_trims_duplicate_or_future_history_before_resume(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            history = Path(temporary_dir) / "history.jsonl"
            history.write_text(
                "\n".join(
                    json.dumps({"epoch": epoch}) for epoch in (0, 1, 2, 2, 3)
                )
                + "\n",
                encoding="utf-8",
            )
            trim_history_for_resume(history, start_epoch=3)
            epochs = [
                json.loads(line)["epoch"]
                for line in history.read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(epochs, [0, 1, 2])

    def test_last_checkpoint_contains_complete_resume_state(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            model = nn.Linear(2, 5)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

            class _Scaler:
                def state_dict(self):
                    return {"scale": 128.0}

            path = Path(temporary_dir) / "checkpoint-last.pth"
            save_classifier_checkpoint(
                path,
                model,
                {"loss": "ce"},
                epoch=2,
                best_qwk=0.91,
                best_epoch=1,
                stale_epochs=1,
                parent_checkpoint=Path(temporary_dir) / "parent.pth",
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=_Scaler(),
            )
            state = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(state["best_epoch"], 1)
        self.assertEqual(state["stale_epochs"], 1)
        self.assertEqual(state["scaler"], {"scale": 128.0})
        self.assertIn("optimizer", state)
        self.assertIn("scheduler", state)


class DeepDRiDPreparationTests(unittest.TestCase):
    def test_converts_official_metadata_to_fixed_class_splits(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            regular = root / "archive" / "regular_fundus_images"

            for folder_name, csv_name, patient_offset in (
                ("regular-fundus-training", "regular-fundus-training.csv", 100),
                ("regular-fundus-validation", "regular-fundus-validation.csv", 200),
            ):
                folder = regular / folder_name
                rows = []
                for grade in range(5):
                    patient_id = patient_offset + grade
                    image_id = f"{patient_id}_l1"
                    image = folder / "Images" / str(patient_id) / f"{image_id}.jpg"
                    image.parent.mkdir(parents=True, exist_ok=True)
                    image.write_bytes(f"image-{folder_name}-{grade}".encode())
                    rows.append(
                        {
                            "image_id": image_id,
                            "left_eye_DR_Level": grade,
                            "right_eye_DR_Level": None,
                        }
                    )
                folder.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(folder / csv_name, index=False)

            evaluation = regular / "Online-Challenge1&2-Evaluation"
            test_rows = []
            for grade in range(5):
                patient_id = 300 + grade
                image_id = f"{patient_id}_r1"
                image = evaluation / "Images" / str(patient_id) / f"{image_id}.jpg"
                image.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(f"test-image-{grade}".encode())
                test_rows.append({"image_id": image_id, "DR_Levels": grade})
            pd.DataFrame(test_rows).to_excel(
                evaluation / "Challenge1_labels.xlsx", index=False
            )

            output = prepare_deepdrid_target(root, root / "prepared")
            frames, _ = load_split_frames(output)

        self.assertEqual({name: len(frame) for name, frame in frames.items()}, {
            "train": 5,
            "val": 5,
            "test": 5,
        })
        for frame in frames.values():
            self.assertEqual(sorted(frame["diagnosis"].tolist()), list(range(5)))


class FixedSupportTests(unittest.TestCase):
    def test_selects_exactly_k_images_per_class_once(self):
        frame = pd.DataFrame(
            {
                "image_path": [f"{grade}_{index}.jpg" for grade in range(5) for index in range(10)],
                "diagnosis": [grade for grade in range(5) for _ in range(10)],
            }
        )
        first = select_fixed_support(frame, shots=2, seed=42)
        second = select_fixed_support(frame, shots=2, seed=42)
        self.assertEqual(len(first), 10)
        self.assertEqual(first["diagnosis"].value_counts().sort_index().tolist(), [2] * 5)
        self.assertEqual(first["image_path"].tolist(), second["image_path"].tolist())

    def test_episode_only_reuses_fixed_support_and_has_disjoint_query(self):
        fixed = pd.DataFrame(
            {"diagnosis": [grade for grade in range(5) for _ in range(5)]}
        )
        sampler = FixedSupportEpisodeSampler(fixed, seed=42)
        support, support_labels, query, query_labels = sampler.sample(1)
        self.assertEqual(len(support), 20)
        self.assertEqual(len(query), 5)
        self.assertTrue(set(support).isdisjoint(query))
        self.assertEqual([support_labels.count(grade) for grade in range(5)], [4] * 5)
        self.assertEqual([query_labels.count(grade) for grade in range(5)], [1] * 5)
        self.assertTrue(set(support + query).issubset(set(fixed.index)))

    def test_rejects_class_with_too_few_images(self):
        frame = pd.DataFrame(
            {"diagnosis": [grade for grade in range(5) for _ in range(2)]}
        )
        with self.assertRaisesRegex(ValueError, "needs at least"):
            select_fixed_support(frame, shots=3, seed=1)


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
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
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
        self.assertIn("Pseudo-label progress: image 1/2", output.getvalue())


if __name__ == "__main__":
    unittest.main()
