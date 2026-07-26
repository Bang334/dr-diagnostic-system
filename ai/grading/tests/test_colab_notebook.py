import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch


NOTEBOOK = Path(__file__).resolve().parents[1] / "DR_Training_Colab.ipynb"


class ColabNotebookDownloadTests(unittest.TestCase):
    def test_hugging_face_auth_cell_survives_temporary_gateway_timeout(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = next(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if "from huggingface_hub import login, whoami" in "".join(
                cell.get("source", [])
            )
        )
        fake_google = types.ModuleType("google")
        fake_colab = types.ModuleType("google.colab")
        fake_colab.userdata = types.SimpleNamespace(get=lambda _name: "test-token")
        fake_google.colab = fake_colab
        fake_huggingface = types.ModuleType("huggingface_hub")
        fake_huggingface.login = lambda **_kwargs: None

        def raise_gateway_timeout(**_kwargs):
            raise RuntimeError("504 Gateway Timeout")

        fake_huggingface.whoami = raise_gateway_timeout
        output = io.StringIO()
        with patch.dict(
            sys.modules,
            {
                "google": fake_google,
                "google.colab": fake_colab,
                "huggingface_hub": fake_huggingface,
            },
        ):
            with patch.dict(os.environ, {}, clear=False):
                with contextlib.redirect_stdout(output):
                    exec(
                        compile(source, "DR_Training_Colab.ipynb:hf-auth", "exec"),
                        {},
                    )
                self.assertEqual(os.environ["HF_TOKEN"], "test-token")
        self.assertIn("504 Gateway Timeout", output.getvalue())
        self.assertIn("tiếp tục", output.getvalue())

    def test_downloads_merged_dataset_and_preserves_predefined_splits(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        clone_source = "".join(notebook["cells"][6]["source"])
        source = "".join(notebook["cells"][12]["source"])
        train_source = "".join(notebook["cells"][16]["source"])
        self.assertIn('feat/keras-grade-semi-supervised', clone_source)
        self.assertNotIn('feat/merged-dataset-training', clone_source)
        self.assertIn("sehastrajits/fundus-aptosddridirdeyepacsmessidor", source)
        self.assertIn('["datasets", "download"', source)
        self.assertIn("subprocess.run", source)
        self.assertIn("check=True", source)
        self.assertIn("403", source)
        self.assertIn("zip_path.exists()", source)
        self.assertIn("find_predefined_splits", source)
        self.assertNotIn("split_names = ('train', 'validation', 'test')", source)
        self.assertIn('"--dataset-dir"', train_source)
        self.assertNotIn('"--images-dir"', train_source)
        self.assertIn("PREPROCESSING = 'rgb_crop'", train_source)
        self.assertIn("ARCHITECTURE = 'convnext'", train_source)
        self.assertIn('"--preprocessing"', train_source)
        self.assertIn("notebook_training.log", train_source)
        self.assertIn("timestamp_utc", train_source)
        self.assertIn("gpu_memory_gb", train_source)
        self.assertIn("dataset_images", train_source)

    def test_download_cell_accepts_kaggle_val_directory(self):
        from ai.grading.train import find_predefined_splits

        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "".join(notebook["cells"][12]["source"])
        with tempfile.TemporaryDirectory() as temporary_dir:
            download_dir = Path(temporary_dir) / "fundus_merged"
            download_dir.mkdir()
            archive_path = download_dir / "fundus-aptosddridirdeyepacsmessidor.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for split_name in ("train", "val", "test"):
                    archive.writestr(
                        f"split_dataset/{split_name}/0/sample.jpg",
                        b"image",
                    )

            source = source.replace("/content/fundus_merged", download_dir.as_posix())
            fake_google = types.ModuleType("google")
            fake_colab = types.ModuleType("google.colab")
            fake_colab.userdata = types.SimpleNamespace(get=lambda _name: "test-token")
            fake_google.colab = fake_colab
            fake_train = types.ModuleType("ai.grading.train")
            fake_train.find_predefined_splits = find_predefined_splits
            globals_for_cell = {
                "Path": Path,
                "os": os,
                "subprocess": subprocess,
                "sys": sys,
                "zipfile": zipfile,
            }
            with patch.dict(
                sys.modules,
                {
                    "google": fake_google,
                    "google.colab": fake_colab,
                    "ai.grading.train": fake_train,
                },
            ):
                with patch.object(subprocess, "run", return_value=None):
                    with patch.dict(os.environ, {}, clear=False):
                        with contextlib.redirect_stdout(io.StringIO()):
                            exec(
                                compile(source, "DR_Training_Colab.ipynb:cell12", "exec"),
                                globals_for_cell,
                            )


if __name__ == "__main__":
    unittest.main()
