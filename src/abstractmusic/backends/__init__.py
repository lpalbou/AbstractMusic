"""
AbstractMusic backend implementations.
"""

from __future__ import annotations

from .base_backend import MusicBackend

__all__ = [
    "MusicBackend",
    "AceMusicBackend",
    "AceMusicBackendConfig",
    "ElevenLabsMusicBackend",
    "ElevenLabsMusicBackendConfig",
    "AceStepBackend",
    "AceStepBackendConfig",
    "DiffusersAudioBackend",
    "DiffusersAudioBackendConfig",
    "MusicGenBackend",
    "MusicGenBackendConfig",
    "StableAudioBackend",
    "StableAudioBackendConfig",
    "StableAudio3Backend",
    "StableAudio3BackendConfig",
]


def __getattr__(name: str):
    if name in {"AceMusicBackend", "AceMusicBackendConfig"}:
        from .acemusic import AceMusicBackend, AceMusicBackendConfig

        return AceMusicBackend if name == "AceMusicBackend" else AceMusicBackendConfig
    if name in {"ElevenLabsMusicBackend", "ElevenLabsMusicBackendConfig"}:
        from .elevenlabs_music import ElevenLabsMusicBackend, ElevenLabsMusicBackendConfig

        return ElevenLabsMusicBackend if name == "ElevenLabsMusicBackend" else ElevenLabsMusicBackendConfig
    if name in {"AceStepBackend", "AceStepBackendConfig"}:
        from .acestep import AceStepBackend, AceStepBackendConfig

        return AceStepBackend if name == "AceStepBackend" else AceStepBackendConfig
    if name in {"DiffusersAudioBackend", "DiffusersAudioBackendConfig"}:
        from .diffusers_audio import DiffusersAudioBackend, DiffusersAudioBackendConfig

        return DiffusersAudioBackend if name == "DiffusersAudioBackend" else DiffusersAudioBackendConfig
    if name in {"MusicGenBackend", "MusicGenBackendConfig"}:
        from .musicgen import MusicGenBackend, MusicGenBackendConfig

        return MusicGenBackend if name == "MusicGenBackend" else MusicGenBackendConfig
    if name in {"StableAudioBackend", "StableAudioBackendConfig"}:
        from .stable_audio import StableAudioBackend, StableAudioBackendConfig

        return StableAudioBackend if name == "StableAudioBackend" else StableAudioBackendConfig
    if name in {"StableAudio3Backend", "StableAudio3BackendConfig"}:
        from .stable_audio_3 import StableAudio3Backend, StableAudio3BackendConfig

        return StableAudio3Backend if name == "StableAudio3Backend" else StableAudio3BackendConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
