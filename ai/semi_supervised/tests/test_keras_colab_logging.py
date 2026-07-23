import json
import unittest
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "Keras_Ordinal_Semi_Supervised_Colab.ipynb"


class KerasColabLoggingTests(unittest.TestCase):
    def test_streams_child_output_and_configures_heartbeats(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertIn("PSEUDO_LOG_EVERY_IMAGES = 10", source)
        self.assertIn("PHASE_HEARTBEAT_SECONDS = 30", source)
        self.assertIn("--phase-heartbeat-seconds", source)
        self.assertIn("--pseudo-log-every-images", source)
        self.assertIn("subprocess.Popen(", source)
        self.assertIn("stderr=subprocess.STDOUT", source)
        self.assertIn("for line in process.stdout", source)
        self.assertIn("flush=True", source)
        self.assertIn("notebook-live.log", source)
        self.assertIn("progress.jsonl", source)
        self.assertNotIn("subprocess.run(command, check=True)", source)


if __name__ == "__main__":
    unittest.main()
