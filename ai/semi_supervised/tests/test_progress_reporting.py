import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from ai.semi_supervised.progress import ProgressReporter


class ProgressReportingTests(unittest.TestCase):
    def test_long_phase_emits_start_progress_and_completion(self):
        output = io.StringIO()
        clock_values = iter([100.0, 105.0, 110.0, 115.0])
        with tempfile.TemporaryDirectory() as temporary_dir:
            with contextlib.redirect_stdout(output):
                reporter = ProgressReporter(
                    "pseudo-label-inference",
                    total=10,
                    log_path=Path(temporary_dir) / "progress.jsonl",
                    every_items=2,
                    clock=lambda: next(clock_values),
                )
                reporter.start(detail="warming up model")
                reporter.advance(1)
                reporter.advance(2)
                reporter.complete(detail="selected=2")
        text = output.getvalue()
        self.assertIn("PHASE START", text)
        self.assertIn("warming up model", text)
        self.assertIn("2/10", text)
        self.assertIn("ETA", text)
        self.assertIn("PHASE COMPLETE", text)


if __name__ == "__main__":
    unittest.main()
