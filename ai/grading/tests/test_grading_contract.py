import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from ai.grading.benchmark_backbones import training_command
from ai.grading.preprocessing import PREPROCESSING_RECIPES, PreprocessingSpec, preprocess_fundus
from ai.grading.predictor import Prediction
from ai.grading.taxonomy import CLASS_NAMES, DR_GRADES, grade_definition
from ai.grading.backbones import BACKBONE_PRESETS


class TaxonomyTests(unittest.TestCase):
    def test_icdr_and_etdrs_cover_exactly_five_ordered_grades(self):
        self.assertEqual([item.grade for item in DR_GRADES], list(range(5)))
        self.assertEqual(len(CLASS_NAMES), 5)
        self.assertEqual(grade_definition(4).icdr_label, "Proliferative DR")
        self.assertEqual(grade_definition(3).etdrs_range, "53")

    def test_prediction_contract_exposes_both_clinical_scales(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        result = Prediction(
            grade=2,
            probabilities=(0.05, 0.1, 0.7, 0.1, 0.05),
            model_version="test",
            preprocessing=PreprocessingSpec("rgb_crop", 16),
            preprocessed_bgr=image,
        ).to_api_dict()
        self.assertEqual(result["dr_label"], "Moderate NPDR")
        self.assertEqual(result["etdrs_level_range"], "35, 43, 47")


class PreprocessingTests(unittest.TestCase):
    def test_all_specialized_recipes_return_model_ready_bgr(self):
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.circle(image, (80, 60), 52, (30, 100, 180), -1)
        for recipe in PREPROCESSING_RECIPES:
            with self.subTest(recipe=recipe):
                result = preprocess_fundus(image, PreprocessingSpec(recipe, 64))
                self.assertEqual(result.shape, (64, 64, 3))
                self.assertEqual(result.dtype, np.uint8)


class BackboneBenchmarkTests(unittest.TestCase):
    def test_controlled_benchmark_contains_required_architectures(self):
        self.assertEqual(set(BACKBONE_PRESETS), {"efficientnet", "resnet", "convnext"})
        args = SimpleNamespace(
            dataset_dir=Path("dataset"), output_dir=Path("runs"), epochs=2,
            seed=7, preprocessing="clahe",
        )
        for architecture in BACKBONE_PRESETS:
            command = training_command(args, architecture)
            self.assertIn(architecture, command)
            self.assertIn("clahe", command)


if __name__ == "__main__":
    unittest.main()
