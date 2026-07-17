import json
import unittest
from pathlib import Path


NOTEBOOK = Path(__file__).with_name("DR_Test_Only_Colab.ipynb")


class TestOnlyNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.source = "\n".join(
            "".join(cell.get("source", [])) for cell in cls.notebook["cells"]
        )

    def test_is_a_valid_test_only_colab_notebook(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertEqual(self.notebook["metadata"]["accelerator"], "GPU")
        self.assertIn("drive.mount('/content/drive')", self.source)
        self.assertIn("checkpoint-best.pth", self.source)
        self.assertIn("tanzinabdul/fundus-patientwise-split", self.source)
        self.assertIn("split_paths['test']", self.source)
        self.assertIn("ai.grading.evaluate_test", self.source)

    def test_contains_no_training_or_resume_command(self):
        self.assertNotIn("ai.grading.train'", self.source)
        self.assertNotIn("ai.semi_supervised.semi_supervised_training", self.source)
        self.assertNotIn("'--resume'", self.source)
        self.assertNotIn("'--epochs'", self.source)
        self.assertNotIn("pseudo_labels.csv", self.source)

    def test_saves_and_displays_all_evaluation_artifacts(self):
        self.assertIn("test_metrics.json", self.source)
        self.assertIn("test_predictions.csv", self.source)
        self.assertIn("confusion_matrix_normalized.png", self.source)
        self.assertIn("LIMIT_PER_CLASS", self.source)
        self.assertIn("subprocess.Popen(", self.source)

    def test_does_not_embed_access_tokens(self):
        raw = NOTEBOOK.read_text(encoding="utf-8")
        self.assertNotIn("ghp_", raw)
        self.assertNotIn("hf_", raw)
        self.assertNotIn("kaggle.json", raw)


if __name__ == "__main__":
    unittest.main()
