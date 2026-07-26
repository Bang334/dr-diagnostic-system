import os
import unittest
from unittest.mock import patch

from app.core.config import BASE_DIR, _backend_path, _first_env


class ConfigTests(unittest.TestCase):
    def test_relative_model_paths_are_resolved_from_backend_directory(self):
        self.assertEqual(
            _backend_path("best-fewshot.pth"),
            str((BASE_DIR / "best-fewshot.pth").resolve()),
        )

    def test_gemini_key_accepts_plural_environment_alias(self):
        with patch.dict(
            os.environ,
            {"GEMINI_API_KEY": "", "GEMINI_API_KEYS": "plural-key"},
            clear=False,
        ):
            self.assertEqual(
                _first_env("GEMINI_API_KEY", "GEMINI_API_KEYS"),
                "plural-key",
            )


if __name__ == "__main__":
    unittest.main()
