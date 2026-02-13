"""
Pytest bootstrap for AbstractMusic.

This repo uses a `src/` layout. For local test runs (without an editable install),
we add `src/` to `sys.path` so `import abstractmusic` works reliably.
"""

from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if src.exists() and src.is_dir():
        sys.path.insert(0, str(src))
