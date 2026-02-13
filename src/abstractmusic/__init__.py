"""
AbstractMusic.

Local-first text-to-music / text-to-audio primitives intended to integrate with
AbstractCore via the capability plugin system (`abstractcore.capabilities_plugins`).
"""

from __future__ import annotations

import os

if os.environ.get("DIFFUSERS_SLOW_IMPORT", "").strip().upper() in {"1", "ON", "YES", "TRUE"}:
    os.environ["DIFFUSERS_SLOW_IMPORT"] = "0"

from .music_manager import MusicManager

__all__ = [
    "__version__",
    "MusicManager",
]

__version__ = "0.1.0"
