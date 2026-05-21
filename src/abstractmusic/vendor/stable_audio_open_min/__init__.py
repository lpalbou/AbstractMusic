"""
Minimal vendored Stable Audio Open runtime pieces.

This package vendors the Stable Audio Open model definitions (from
`stable-audio-tools==0.0.19`, MIT license) so AbstractMusic can run the gated
`stabilityai/stable-audio-open-small` checkpoint without requiring the upstream
`stable-audio-tools` package (which pulls heavy UI/training dependencies).

Only the model/config loading surface is exposed here; AbstractMusic owns the
minimal inference loop in `abstractmusic.backends.stable_audio`.
"""

from __future__ import annotations

from .models.factory import create_model_from_config, create_model_from_config_path
from .models.pretrained import get_pretrained_model

__all__ = [
    "create_model_from_config",
    "create_model_from_config_path",
    "get_pretrained_model",
]

