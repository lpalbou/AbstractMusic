"""
AbstractMusic backend implementations.
"""

from __future__ import annotations

from .base_backend import MusicBackend

__all__ = [
    "MusicBackend",
    "AceStepV15Backend",
    "AceStepV15BackendConfig",
    "AceStepDiffusersBackend",
    "AceStepDiffusersBackendConfig",
    "DiffusersAudioBackend",
    "DiffusersAudioBackendConfig",
    "MusicGenBackend",
    "MusicGenBackendConfig",
    "StableAudioBackend",
    "StableAudioBackendConfig",
]


def __getattr__(name: str):
    if name in {"AceStepV15Backend", "AceStepV15BackendConfig"}:
        from .acestep_v15 import AceStepV15Backend, AceStepV15BackendConfig

        return AceStepV15Backend if name == "AceStepV15Backend" else AceStepV15BackendConfig
    if name in {"AceStepDiffusersBackend", "AceStepDiffusersBackendConfig"}:
        from .acestep_diffusers import AceStepDiffusersBackend, AceStepDiffusersBackendConfig

        return AceStepDiffusersBackend if name == "AceStepDiffusersBackend" else AceStepDiffusersBackendConfig
    if name in {"DiffusersAudioBackend", "DiffusersAudioBackendConfig"}:
        from .diffusers_audio import DiffusersAudioBackend, DiffusersAudioBackendConfig

        return DiffusersAudioBackend if name == "DiffusersAudioBackend" else DiffusersAudioBackendConfig
    if name in {"MusicGenBackend", "MusicGenBackendConfig"}:
        from .musicgen import MusicGenBackend, MusicGenBackendConfig

        return MusicGenBackend if name == "MusicGenBackend" else MusicGenBackendConfig
    if name in {"StableAudioBackend", "StableAudioBackendConfig"}:
        from .stable_audio import StableAudioBackend, StableAudioBackendConfig

        return StableAudioBackend if name == "StableAudioBackend" else StableAudioBackendConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
