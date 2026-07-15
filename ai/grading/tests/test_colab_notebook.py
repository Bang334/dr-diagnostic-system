import json
import unittest
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "DR_Training_Colab.ipynb"


class ColabNotebookDownloadTests(unittest.TestCase):
    def test_downloads_merged_dataset_and_preserves_predefined_splits(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        clone_source = "".join(notebook["cells"][6]["source"])
        source = "".join(notebook["cells"][12]["source"])
        train_source = "".join(notebook["cells"][16]["source"])
        self.assertIn('feat/merged-dataset-training', clone_source)
        self.assertIn("sehastrajits/fundus-aptosddridirdeyepacsmessidor", source)
        self.assertIn('["datasets", "download"', source)
        self.assertIn("subprocess.run", source)
        self.assertIn("check=True", source)
        self.assertIn("403", source)
        self.assertIn("zip_path.exists()", source)
        self.assertIn('"--dataset-dir"', train_source)
        self.assertNotIn('"--images-dir"', train_source)


if __name__ == "__main__":
    unittest.main()
