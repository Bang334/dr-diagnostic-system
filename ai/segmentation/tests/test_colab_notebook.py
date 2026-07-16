from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path


NOTEBOOK = Path(__file__).parents[1] / "Lesion_Segmentation_Colab.ipynb"


class ColabNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.source = "\n".join(
            "".join(cell.get("source", [])) for cell in cls.notebook["cells"]
        )

    def test_notebook_is_valid_json_and_python_cells_parse(self) -> None:
        self.assertEqual(self.notebook["nbformat"], 4)
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))

    def test_colab_download_and_training_contract(self) -> None:
        required = (
            "drive.mount('/content/drive')",
            "KAGGLE_API_TOKEN",
            "dankok/diabetic-retinopathy-image-dataset",
            "build_idrid_manifest",
            "ai.segmentation.train",
            "unetplusplus",
            "manet",
            "resnet34",
            "imagenet",
            "summary.json",
        )
        for value in required:
            self.assertIn(value, self.source)

    def test_notebook_does_not_embed_credentials_or_touch_test_selection(self) -> None:
        self.assertNotIn("kaggle.json", self.source)
        self.assertNotIn("--test-dir", self.source)
        self.assertNotIn("test_used_for_model_selection = True", self.source)


if __name__ == "__main__":
    unittest.main()
