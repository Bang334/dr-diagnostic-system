import os
import unittest
from unittest.mock import patch

from ai.grading.colab_secrets import load_runtime_secret


class MissingSecretError(Exception):
    pass


class FakeUserdata:
    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error

    def get(self, _name):
        if self.error:
            raise self.error
        return self.value


class RuntimeSecretTests(unittest.TestCase):
    def test_uses_existing_environment_value_without_prompt(self):
        with patch.dict(os.environ, {"HF_TOKEN": "from-environment"}, clear=False):
            value = load_runtime_secret(
                "HF_TOKEN",
                userdata=FakeUserdata(value="from-colab"),
                prompt_fn=lambda _: self.fail("prompt should not be used"),
            )
        self.assertEqual(value, "from-environment")

    def test_falls_back_to_hidden_prompt_when_colab_secret_is_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            value = load_runtime_secret(
                "HF_TOKEN",
                userdata=FakeUserdata(error=MissingSecretError("missing")),
                prompt_fn=lambda _: "from-hidden-prompt",
            )
            self.assertEqual(os.environ["HF_TOKEN"], "from-hidden-prompt")
        self.assertEqual(value, "from-hidden-prompt")

    def test_rejects_empty_prompt_value(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                load_runtime_secret(
                    "HF_TOKEN",
                    userdata=FakeUserdata(error=MissingSecretError("missing")),
                    prompt_fn=lambda _: "",
                )


if __name__ == "__main__":
    unittest.main()
