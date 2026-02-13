"""
AbstractMusic backend implementations.
"""

from __future__ import annotations

from .base_backend import MusicBackend
from .acestep_v15 import AceStepV15Backend, AceStepV15BackendConfig

__all__ = [
    "MusicBackend",
    "AceStepV15Backend",
    "AceStepV15BackendConfig",
    "DiffusersAudioBackend",
    "DiffusersAudioBackendConfig",
]


def __getattr__(name: str):
    if name in {"DiffusersAudioBackend", "DiffusersAudioBackendConfig"}:
        from .diffusers_audio import DiffusersAudioBackend, DiffusersAudioBackendConfig

        return DiffusersAudioBackend if name == "DiffusersAudioBackend" else DiffusersAudioBackendConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
