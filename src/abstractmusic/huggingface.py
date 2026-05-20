"""
Helpers for Hugging Face model identifier validation.
"""

from __future__ import annotations

import re
from pathlib import Path

_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\\\/]")


def is_local_model_reference(value: str) -> bool:
    """Return True when a string should be treated as a local filesystem reference."""

    candidate = str(value or "").strip()
    if not candidate:
        return False

    lower = candidate.lower()
    if lower.startswith(("file://", "http://", "https://")):
        return True

    if candidate in {".", "..", "~"}:
        return True

    if candidate.startswith(("/", "./", "../", "~/", ".\\", "..\\", "~\\", "\\\\")):
        return True

    if _WINDOWS_DRIVE_RE.match(candidate):
        return True

    try:
        return Path(candidate).expanduser().exists()
    except Exception:
        return False


def require_hf_repo_id(value: str, *, field_name: str = "model_id") -> str:
    """Validate that a model selector is a Hugging Face repo id, not a local path."""

    candidate = str(value or "").strip()
    if not candidate:
        raise ValueError(f"{field_name} must be a Hugging Face repo id.")

    if is_local_model_reference(candidate):
        raise ValueError(
            f"{field_name} must be a Hugging Face repo id, not a local filesystem path or URL. "
            "Install model weights in the default Hugging Face cache and pass the repo id instead."
        )

    return candidate
