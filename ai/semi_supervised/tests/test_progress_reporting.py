import contextlib
import io
import itertools
import tempfile
import time
import unittest
from pathlib import Path

from ai.semi_supervised.progress import ProgressReporter


class ProgressReportingTests(unittest.TestCase):
    def test_long_phase_emits_start_progress_and_completion(self):
        output = io.StringIO()
        clock_values = itertools.count(100.0, 5.0)
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

    def test_heartbeat_appears_while_an_item_is_still_blocked(self):
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temporary_dir:
            with contextlib.redirect_stdout(output):
                reporter = ProgressReporter(
                    "model-warmup",
                    total=1,
                    log_path=Path(temporary_dir) / "progress.jsonl",
                    heartbeat_seconds=0.02,
                )
                reporter.start(detail="first prediction")
                time.sleep(0.055)
                reporter.complete()
        self.assertIn("HEARTBEAT", output.getvalue())


if __name__ == "__main__":
    unittest.main()
