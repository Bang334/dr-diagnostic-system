import json
import unittest
from pathlib import Path


NOTEBOOK = (
    Path(__file__).resolve().parents[1] / "Semi_Supervised_Few_Shot_Colab.ipynb"
)


class ResearchNotebookTests(unittest.TestCase):
    def test_selects_drive_checkpoint_and_both_research_modes(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn("drive.mount('/content/drive')", source)
        self.assertIn("checkpoint-best.pth", source)
        self.assertIn(
            "/content/drive/MyDrive/retfound_merged_seed42/checkpoint-best.pth",
            source,
        )
        self.assertIn("load_grading_checkpoint", source)
        self.assertIn("ai.semi_supervised.semi_supervised_training", source)
        self.assertIn("ai.semi_supervised.few_shot_demo", source)
        self.assertIn("--unlabeled-dir", source)
        self.assertIn("test_split_used", source)
        self.assertIn(
            "kaggle:sehastrajits/fundus-aptosddridirdeyepacsmessidor", source
        )
        self.assertIn("datasets', 'download'", source)
        self.assertIn("userdata.get('KAGGLE_API_TOKEN')", source)
        self.assertIn("Labeled replay:", source)
        self.assertIn("Target labeled:", source)
        self.assertIn("Unlabeled pool:", source)
        self.assertIn("labeled replay + pseudo-label mới", source)
        self.assertIn("--target-dataset-dir", source)
        self.assertIn("support_manifest.csv", source)
        self.assertIn("target_test_used_for_model_selection", source)
        self.assertIn("github:deepdrdoc/DeepDRiD@v1.1", source)
        self.assertIn("prepare_deepdrid_target", source)
        self.assertNotIn("--val-episodes", source)

    def test_does_not_embed_access_tokens(self):
        raw = NOTEBOOK.read_text(encoding="utf-8")
        self.assertNotIn("ghp_", raw)
        self.assertNotIn("hf_", raw)


if __name__ == "__main__":
    unittest.main()
