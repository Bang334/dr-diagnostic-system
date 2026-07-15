import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn

from ai.grading.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    CLASS_NAMES,
    ArtifactMetadata,
    ModelSpec,
    PreprocessingSpec,
    load_torch_checkpoint,
    validate_resume_metadata,
)
from ai.grading.data_audit import audit_splits
from ai.grading.predictor import PyTorchRETFoundPredictor
from ai.grading.train import (
    FundusDataset,
    WarmupCosineScheduler,
    build_optimizer,
    build_transforms,
    calculate_metrics,
    configure_trainable,
    create_scaler,
    save_checkpoint,
)


class _TinyViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(4, 4)
        self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(4)])
        self.norm = nn.LayerNorm(4)
        self.head = nn.Linear(4, 5)


class AdaptationTests(unittest.TestCase):
    def test_last_two_blocks_leave_earlier_backbone_frozen(self):
        model = _TinyViT()
        configure_trainable(model, "last_n_blocks", last_n_blocks=2)
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

        self.assertNotIn("blocks.1.weight", trainable)
        self.assertIn("blocks.2.weight", trainable)
        self.assertIn("blocks.3.weight", trainable)
        self.assertIn("norm.weight", trainable)
        self.assertIn("head.weight", trainable)

    def test_layerwise_optimizer_orders_lr_and_excludes_bias_from_decay(self):
        model = _TinyViT()
        configure_trainable(model, "full_finetune")
        args = SimpleNamespace(peak_lr=1e-4, layer_decay=0.75, weight_decay=0.05)
        optimizer = build_optimizer(model, args)

        self.assertAlmostEqual(max(group["base_lr"] for group in optimizer.param_groups), 1e-4)
        self.assertLess(min(group["base_lr"] for group in optimizer.param_groups), 1e-4)
        self.assertTrue(any(group["weight_decay"] == 0.0 for group in optimizer.param_groups))
        self.assertTrue(any(group["weight_decay"] == 0.05 for group in optimizer.param_groups))


class SchedulerTests(unittest.TestCase):
    def test_warmup_then_cosine_reaches_scaled_floor(self):
        parameter = nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW(
            [{"params": [parameter], "base_lr": 1e-4, "lr_scale": 0.5, "lr": 1e-4}]
        )
        scheduler = WarmupCosineScheduler(
            optimizer, total_updates=10, warmup_updates=2, min_lr=1e-6
        )
        scheduler.step_update(0)
        first = optimizer.param_groups[0]["lr"]
        scheduler.step_update(1)
        peak = optimizer.param_groups[0]["lr"]
        scheduler.step_update(9)
        final = optimizer.param_groups[0]["lr"]

        self.assertLess(first, peak)
        self.assertAlmostEqual(peak, 1e-4)
        self.assertAlmostEqual(final, 5e-7)


class MetricTests(unittest.TestCase):
    def test_probability_metrics_are_perfect_for_one_hot_predictions(self):
        targets = [0, 1, 2, 3, 4]
        probabilities = np.eye(5).tolist()
        metrics = calculate_metrics(targets, targets, probabilities)

        self.assertEqual(metrics["qwk"], 1.0)
        self.assertEqual(metrics["macro_auroc"], 1.0)
        self.assertEqual(metrics["macro_auprc"], 1.0)
        self.assertEqual(metrics["ece_15_bin"], 0.0)
        self.assertEqual(metrics["multiclass_brier"], 0.0)


class LeakageAuditTests(unittest.TestCase):
    def test_exact_duplicate_across_splits_is_blocking(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            image = np.full((24, 24, 3), 128, dtype=np.uint8)
            train_path = root / "train.png"
            val_path = root / "val.png"
            cv2.imwrite(str(train_path), image)
            val_path.write_bytes(train_path.read_bytes())
            splits = {
                "train": pd.DataFrame(
                    [{"image_id": "train.png", "image_path": str(train_path), "diagnosis": 0}]
                ),
                "val": pd.DataFrame(
                    [{"image_id": "val.png", "image_path": str(val_path), "diagnosis": 0}]
                ),
                "test": pd.DataFrame(
                    [{"image_id": "test.png", "image_path": str(root / "test.png"), "diagnosis": 1}]
                ),
            }
            cv2.imwrite(str(root / "test.png"), np.zeros((24, 24, 3), dtype=np.uint8))
            args = SimpleNamespace(
                image_column="image_id",
                label_column="diagnosis",
                images_dir=root,
                image_extension=".png",
            )

            result = audit_splits(splits, args, root / "audit", fail_on_leakage=False)

            self.assertGreater(result.report["blocking_cross_split_violations"], 0)
            self.assertTrue((root / "audit" / "leakage_violations.csv").is_file())


class PredictorContractTests(unittest.TestCase):
    def test_pytorch_adapter_loads_versioned_artifact_and_predicts(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "tiny.pth"
            model = timm.create_model("resnet18", pretrained=False, num_classes=5)
            metadata = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "model": ModelSpec(
                    framework="pytorch",
                    model_source="timm",
                    architecture="resnet18",
                ).__dict__,
                "preprocessing": PreprocessingSpec(image_size=32).__dict__,
                "class_names": list(CLASS_NAMES),
                "grading_scale": "ICDR",
                "training": {"loss": "ce"},
                "provenance": {"test": True},
            }
            torch.save({"model": model.state_dict(), "metadata": metadata}, path)

            predictor = PyTorchRETFoundPredictor(path, device="cpu")
            image = np.full((40, 40, 3), 100, dtype=np.uint8)
            image_path = Path(temporary_dir) / "sample.png"
            cv2.imwrite(str(image_path), image)
            prediction = predictor.predict(image)

            _, eval_transform = build_transforms(32)
            dataset = FundusDataset(
                pd.DataFrame(
                    [{"image_id": "sample.png", "image_path": str(image_path), "diagnosis": 0}]
                ),
                SimpleNamespace(
                    image_size=32,
                    enhance=False,
                    crop_tolerance=7,
                    label_column="diagnosis",
                    image_column="image_id",
                    images_dir=Path(temporary_dir),
                    image_extension=".png",
                ),
                eval_transform,
            )
            tensor, _, _ = dataset[0]
            model.eval()
            with torch.inference_mode():
                expected = torch.softmax(model(tensor.unsqueeze(0)), dim=1)[0].numpy()

            self.assertEqual(len(prediction.probabilities), 5)
            self.assertAlmostEqual(sum(prediction.probabilities), 1.0, places=5)
            self.assertEqual(predictor.info.class_names, CLASS_NAMES)
            np.testing.assert_allclose(prediction.probabilities, expected, rtol=1e-5, atol=1e-6)


class ResumeContractTests(unittest.TestCase):
    def test_rejects_changed_preprocessing(self):
        common = dict(
            schema_version=ARTIFACT_SCHEMA_VERSION,
            model=ModelSpec("pytorch", "timm", "resnet18"),
            class_names=CLASS_NAMES,
            grading_scale="ICDR",
            training={"loss": "ce", "adaptation": "linear_probe"},
            provenance={"train_manifest_sha256": "same"},
        )
        saved = ArtifactMetadata(
            preprocessing=PreprocessingSpec(image_size=224, enhance=False),
            **common,
        )
        current = ArtifactMetadata(
            preprocessing=PreprocessingSpec(image_size=224, enhance=True),
            **common,
        )

        with self.assertRaisesRegex(ValueError, "preprocessing"):
            validate_resume_metadata(saved, current)

    def test_last_checkpoint_contains_resumable_state(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            model = _TinyViT()
            configure_trainable(model, "linear_probe")
            args = SimpleNamespace(peak_lr=1e-4, layer_decay=0.75, weight_decay=0.05)
            optimizer = build_optimizer(model, args)
            scheduler = WarmupCosineScheduler(
                optimizer, total_updates=10, warmup_updates=2, min_lr=1e-6
            )
            scheduler.step_update(3)
            metadata = ArtifactMetadata(
                schema_version=ARTIFACT_SCHEMA_VERSION,
                model=ModelSpec("pytorch", "timm", "tiny"),
                preprocessing=PreprocessingSpec(32),
                class_names=CLASS_NAMES,
                grading_scale="ICDR",
                training={"loss": "ce", "adaptation": "linear_probe"},
                provenance={},
            )
            generator = torch.Generator().manual_seed(42)
            path = Path(temporary_dir) / "checkpoint-last.pth"

            save_checkpoint(
                path,
                model,
                optimizer,
                create_scaler(False),
                2,
                0.5,
                1,
                SimpleNamespace(test=True),
                metadata,
                scheduler,
                4,
                generator,
                include_training_state=True,
            )
            checkpoint = load_torch_checkpoint(path)

            self.assertIn("optimizer", checkpoint)
            self.assertIn("scheduler", checkpoint)
            self.assertIn("rng_state", checkpoint)
            self.assertEqual(checkpoint["global_update"], 4)


if __name__ == "__main__":
    unittest.main()
