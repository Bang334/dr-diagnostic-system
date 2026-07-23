import json
import unittest
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "Train_Semi_V2_Colab.ipynb"


class TrainSemiV2NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.source = "\n".join(
            "".join(cell.get("source", [])) for cell in cls.notebook["cells"]
        )

    def test_contains_complete_new_resume_and_test_workflow(self):
        self.assertIn("ai.train_semi_v2.train", self.source)
        self.assertIn("# TRAIN NEW", self.source)
        self.assertIn("# RESUME SEMI V2", self.source)
        self.assertIn("--resume", self.source)
        self.assertIn("--eval-only", self.source)
        self.assertIn("checkpoint-last.pth", self.source)
        self.assertIn("checkpoint-best.pth", self.source)

    def test_uses_shared_cache_and_explains_invalidation(self):
        self.assertIn("--pseudo-cache-dir", self.source)
        self.assertIn("PSEUDO_CACHE_DIR", self.source)
        self.assertIn("Đổi threshold", self.source)
        self.assertIn("không xóa", self.source)

    def test_downloads_the_two_kaggle_datasets_instead_of_assuming_drive_folders(self):
        self.assertIn(
            "sehastrajits/fundus-aptosddridirdeyepacsmessidor", self.source
        )
        self.assertIn(
            "griffchristenson/unlabeled-retinal-image-dataset", self.source
        )
        self.assertIn("kaggle_cli", self.source)
        self.assertIn("split_dataset", self.source)
        self.assertNotIn(
            "Path('/content/drive/MyDrive/fundus_merged')", self.source
        )
        self.assertNotIn(
            "Path('/content/drive/MyDrive/fundus_unlabeled')", self.source
        )

    def test_logs_child_process_and_handles_interrupt(self):
        self.assertIn("subprocess.Popen", self.source)
        self.assertIn("stderr=subprocess.STDOUT", self.source)
        self.assertIn("KeyboardInterrupt", self.source)
        self.assertIn("checkpoint-last.pth của epoch hoàn tất gần nhất", self.source)

    def test_does_not_embed_tokens(self):
        raw = NOTEBOOK.read_text(encoding="utf-8")
        self.assertNotIn("ghp_", raw)
        self.assertNotIn("hf_", raw)

    def test_v2_source_has_no_dependency_on_legacy_semi_package(self):
        package = NOTEBOOK.parent
        python_source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in package.glob("*.py")
        )
        self.assertNotIn("ai.semi_supervised", python_source)


if __name__ == "__main__":
    unittest.main()
