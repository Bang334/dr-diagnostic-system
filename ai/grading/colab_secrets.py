"""Load notebook credentials without writing them to source or output cells."""

from __future__ import annotations

import getpass
import os
from typing import Any, Callable, Optional


_RECOVERABLE_COLAB_ERRORS = {
    "SecretNotFoundError",
    "NotebookAccessError",
    "TimeoutException",
}


def load_runtime_secret(
    name: str,
    *,
    prompt: Optional[str] = None,
    userdata: Any = None,
    prompt_fn: Optional[Callable[[str], str]] = None,
) -> str:
    """Return a secret from env/Colab Secrets or request hidden session input.

    Resolution order is intentionally deterministic:

    1. Existing environment variable (useful on resume and outside Colab).
    2. ``google.colab.userdata`` secret.
    3. A hidden ``getpass`` prompt that only lives in the current runtime.
    """
    existing = os.environ.get(name)
    if existing:
        return existing

    if userdata is None:
        try:
            from google.colab import userdata as colab_userdata

            userdata = colab_userdata
        except ImportError:
            userdata = None

    value = None
    if userdata is not None:
        try:
            value = userdata.get(name)
        except Exception as exc:
            if exc.__class__.__name__ not in _RECOVERABLE_COLAB_ERRORS:
                # Test doubles and compatible Colab clients may use a custom
                # missing-secret exception. Treat an explicit "missing" error
                # as recoverable, while preserving unrelated failures.
                if "missing" not in str(exc).lower() and "does not exist" not in str(exc).lower():
                    raise

    if not value:
        reader = prompt_fn or getpass.getpass
        value = reader(prompt or f"Enter {name} (hidden input): ").strip()
    if not value:
        raise RuntimeError(f"{name} was not provided")

    os.environ[name] = value
    return value
