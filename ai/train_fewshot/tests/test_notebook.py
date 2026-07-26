import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ai.train_fewshot.runtime import prepare_fresh_output_dir


NOTEBOOK = Path(__file__).resolve().parents[1] / "Train_FewShot_DeepDRiD_Colab.ipynb"


class FewShotNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.source = "\n".join(
            "".join(cell.get("source", [])) for cell in cls.notebook["cells"]
        )

    def test_uses_official_pinned_deepdrid_release(self):
        self.assertIn("zenodo.org/records/8248825", self.source)
        self.assertIn("3379e2fd7a2dd398545a67148420a5d3", self.source)
        self.assertIn("prepare_deepdrid_target", self.source)
        self.assertNotIn("kaggle:", self.source.lower())

    def test_downloads_dataset_to_colab_local_storage(self):
        self.assertIn("DATASET_ROOT = Path('/content/datasets')", self.source)
        self.assertIn("shutil.unpack_archive", self.source)
        self.assertNotIn("DRIVE_ROOT / 'datasets'", self.source)

    def test_checks_out_the_fewshot_fix_branch(self):
        self.assertIn(
            "GITHUB_BRANCH = 'fix/semi-pseudo-weight-batch4'",
            self.source,
        )
        self.assertNotIn("feat/keras-grade-semi-supervised", self.source)

    def test_has_separate_train_resume_and_test_cells(self):
        self.assertIn("# TRAIN NEW", self.source)
        self.assertIn("# RESUME FEWSHOT", self.source)
        self.assertIn("# TEST HELD-OUT TARGET", self.source)
        self.assertIn("--resume", self.source)
        self.assertIn("--eval-only", self.source)

    def test_streams_logs_and_handles_interrupt(self):
        self.assertIn("subprocess.Popen", self.source)
        self.assertIn("stderr=subprocess.STDOUT", self.source)
        self.assertIn("KeyboardInterrupt", self.source)
        self.assertIn("checkpoint-last.pth", self.source)

    def test_streaming_log_does_not_make_new_run_directory_nonempty(self):
        streaming_cell = next(
            "".join(cell.get("source", []))
            for cell in self.notebook["cells"]
            if "def run_streaming" in "".join(cell.get("source", []))
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "runs" / "new-run"

            class TrainingStdout:
                def __iter__(self):
                    prepare_fresh_output_dir(output_dir)
                    return iter(())

            process = mock.Mock()
            process.stdout = TrainingStdout()
            process.returncode = 0
            namespace = {
                "OUTPUT_DIR": output_dir,
                "subprocess": mock.Mock(
                    Popen=mock.Mock(return_value=process),
                    PIPE=object(),
                    STDOUT=object(),
                ),
            }
            exec(streaming_cell, namespace)

            namespace["run_streaming"](["python", "-m", "train"], "notebook.log")

    def test_train_new_recovers_log_left_by_old_streaming_bug(self):
        train_cell = next(
            "".join(cell.get("source", []))
            for cell in self.notebook["cells"]
            if "# TRAIN NEW" in "".join(cell.get("source", []))
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "runs" / "new-run"
            output_dir.mkdir(parents=True)
            legacy_log = output_dir / "notebook-train-live.log"
            legacy_log.write_text("old traceback", encoding="utf-8")
            run_streaming = mock.Mock()
            namespace = {
                "OUTPUT_DIR": output_dir,
                "build_command": mock.Mock(return_value=["python", "-m", "train"]),
                "run_streaming": run_streaming,
                "print": mock.Mock(),
            }

            exec(train_cell, namespace)

            run_streaming.assert_called_once()
            recovered_log = (
                output_dir.parent
                / "_notebook_logs"
                / f"{output_dir.name}-{legacy_log.name}"
            )
            self.assertEqual(
                recovered_log.read_text(encoding="utf-8"),
                "old traceback",
            )

    def test_all_code_cells_parse(self):
        for index, cell in enumerate(self.notebook["cells"]):
            if cell.get("cell_type") == "code":
                ast.parse("".join(cell.get("source", [])), filename=f"cell-{index}")

    def test_does_not_embed_tokens(self):
        raw = NOTEBOOK.read_text(encoding="utf-8")
        self.assertNotIn("ghp_", raw)
        self.assertNotIn("hf_", raw)


if __name__ == "__main__":
    unittest.main()
