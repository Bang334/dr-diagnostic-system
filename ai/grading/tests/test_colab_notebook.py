import json
import unittest
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "DR_Training_Colab.ipynb"


class ColabNotebookDownloadTests(unittest.TestCase):
    def test_kaggle_download_fails_fast_before_unzip(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "".join(notebook["cells"][12]["source"])
        self.assertIn("subprocess.run", source)
        self.assertIn("check=True", source)
        self.assertIn("403", source)
        self.assertIn("zip_path.exists()", source)
        self.assertNotIn("!kaggle competitions download", source)


if __name__ == "__main__":
    unittest.main()
