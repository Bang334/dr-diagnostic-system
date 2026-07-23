import os
import unittest
from unittest.mock import patch

from app.core.config import _first_env


class ConfigTests(unittest.TestCase):
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
