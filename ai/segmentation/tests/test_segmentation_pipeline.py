from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from ai.segmentation.data import (
    IDRiDLesionDataset,
    LESION_MASK_DIRS,
    build_idrid_manifest,
    create_idrid_splits,
)
from ai.segmentation.metrics import SegmentationMeter, calibrate_dice_thresholds
from ai.segmentation.train import DiceFocalLoss, parse_args, validate_args


def create_idrid_fixture(root: Path) -> Path:
    segmentation = root / "bundle" / "Segmentation"
    for split, start, stop in (
        ("Training Set", 1, 55),
        ("Testing Set", 55, 82),
    ):
        image_dir = segmentation / "Original_Images" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        for mask_dir, _ in LESION_MASK_DIRS.values():
            (segmentation / "Segmentation_Groundtruths" / split / mask_dir).mkdir(
                parents=True, exist_ok=True
            )
        for number in range(start, stop):
            image_id = f"IDRiD_{number:02d}"
            image = np.zeros((24, 32, 3), dtype=np.uint8)
            cv2.circle(image, (16, 12), 10, (30, 100, 180), -1)
            if not cv2.imwrite(str(image_dir / f"{image_id}.jpg"), image):
                raise RuntimeError("Could not create test image")
            for lesion, (mask_dir, suffix) in LESION_MASK_DIRS.items():
                # The official set has one hemorrhage-negative image without a HE file.
                if lesion == "hemorrhage" and number == 1:
                    continue
                mask = np.zeros((24, 32), dtype=np.uint8)
                mask[10:13, 14:18] = 255
                mask_path = (
                    segmentation
                    / "Segmentation_Groundtruths"
                    / split
                    / mask_dir
                    / f"{image_id}_{suffix}.tif"
                )
                if not cv2.imwrite(str(mask_path), mask):
                    raise RuntimeError("Could not create test mask")
    return root


class IDRiDDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = create_idrid_fixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_manifest_validates_official_layout_and_counts(self) -> None:
        frame = build_idrid_manifest(self.root)
        self.assertEqual(len(frame), 81)
        self.assertEqual(
            frame["official_split"].value_counts().to_dict(),
            {"Training Set": 54, "Testing Set": 27},
        )
        self.assertEqual(int((frame["hemorrhage_mask"] == "").sum()), 1)

    def test_splits_are_reproducible_and_keep_official_test_untouched(self) -> None:
        split_a = create_idrid_splits(self.root, Path(self.temporary.name) / "split-a")
        split_b = create_idrid_splits(self.root, Path(self.temporary.name) / "split-b")
        self.assertEqual({key: len(value) for key, value in split_a.items()}, {
            "train": 43,
            "val": 11,
            "test": 27,
        })
        self.assertListEqual(split_a["val"]["image_id"].tolist(), split_b["val"]["image_id"].tolist())
        train_ids = set(split_a["train"]["image_id"])
        val_ids = set(split_a["val"]["image_id"])
        test_ids = set(split_a["test"]["image_id"])
        self.assertFalse(train_ids & val_ids)
        self.assertFalse((train_ids | val_ids) & test_ids)
        self.assertTrue(all(value >= "IDRiD_55" for value in test_ids))

    def test_dataset_returns_three_binary_masks_and_missing_he_is_negative(self) -> None:
        frame = build_idrid_manifest(self.root)
        row = frame[frame["image_id"] == "IDRiD_01"]
        image, mask, image_id = IDRiDLesionDataset(
            row, image_size=32, training=False
        )[0]
        self.assertEqual(image.shape, (3, 32, 32))
        self.assertEqual(mask.shape, (3, 32, 32))
        self.assertEqual(image_id, "IDRiD_01")
        self.assertEqual(float(mask[1].sum()), 0.0)
        self.assertGreater(float(mask[0].sum()), 0.0)
        self.assertTrue(set(torch.unique(mask).tolist()).issubset({0.0, 1.0}))


class MetricAndLossTests(unittest.TestCase):
    def test_perfect_prediction_has_perfect_metrics(self) -> None:
        target = torch.tensor(
            [[[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]]],
            dtype=torch.float32,
        )
        meter = SegmentationMeter([0.5, 0.5, 0.5])
        meter.update(target, target)
        result = meter.compute()
        self.assertAlmostEqual(result["macro"]["dice"], 1.0)
        self.assertAlmostEqual(result["macro"]["iou"], 1.0)

    def test_threshold_calibration_uses_each_lesion_independently(self) -> None:
        targets = torch.tensor(
            [[[[1.0, 0.0]], [[1.0, 0.0]], [[1.0, 0.0]]]],
            dtype=torch.float32,
        )
        probabilities = torch.tensor(
            [[[[0.40, 0.30]], [[0.70, 0.60]], [[0.90, 0.80]]]],
            dtype=torch.float32,
        )
        thresholds = calibrate_dice_thresholds(
            [probabilities], [targets], grid=(0.35, 0.65, 0.85)
        )
        self.assertEqual(thresholds, [0.35, 0.65, 0.85])

    def test_loss_prefers_correct_logits(self) -> None:
        criterion = DiceFocalLoss(
            alpha=0.75,
            gamma=2.0,
            dice_weight=0.6,
            focal_weight=0.4,
        )
        target = torch.tensor([[[[1.0, 0.0]]] * 3])
        correct = torch.tensor([[[[6.0, -6.0]]] * 3])
        wrong = -correct
        self.assertLess(float(criterion(correct, target)), float(criterion(wrong, target)))

    def test_cli_research_defaults(self) -> None:
        args = parse_args(["--dataset-dir", "data", "--output-dir", "run"])
        validate_args(args)
        self.assertEqual(args.architecture, "unet")
        self.assertEqual(args.encoder_name, "resnet34")
        self.assertEqual(args.encoder_weights, "imagenet")
        self.assertEqual(args.image_size, 768)


if __name__ == "__main__":
    unittest.main()
