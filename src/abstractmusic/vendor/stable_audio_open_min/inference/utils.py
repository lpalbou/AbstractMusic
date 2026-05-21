"""Minimal helpers required by vendored Stable Audio Open model code."""

from __future__ import annotations

from typing import Any


def set_audio_channels(audio: Any, channels: int) -> Any:
    """Best-effort channel adapter for tensors shaped [B,C,T] (or [C,T]).

    Stable Audio Open conditioners may feed audio into a pretransform with a
    fixed expected channel count. The upstream helper performs simple
    mono/stereo reshaping; we keep the same intent here with minimal logic.
    """

    target = int(channels)
    if target <= 0:
        return audio

    try:
        import torch  # type: ignore
    except Exception:
        return audio
    if not isinstance(audio, torch.Tensor):
        return audio

    t = audio
    squeeze = False
    if t.ndim == 2:
        t = t.unsqueeze(0)
        squeeze = True
    if t.ndim != 3:
        return audio

    batch, current, samples = t.shape
    if current == target:
        return t.squeeze(0) if squeeze else t

    if target == 1:
        out = t.mean(dim=1, keepdim=True)
    elif current == 1 and target > 1:
        out = t.repeat(1, target, 1)
    elif current > target:
        out = t[:, :target, :]
    else:
        pad = target - current
        zeros = t.new_zeros((batch, pad, samples))
        out = torch.cat([t, zeros], dim=1)

    return out.squeeze(0) if squeeze else out


__all__ = ["set_audio_channels"]


def prepare_audio(
    audio: Any,
    *,
    in_sr: int,
    target_sr: int,
    target_length: int,
    target_channels: int,
    device: str,
) -> Any:
    """Minimal resample + pad/crop helper (vendored contract).

    This mirrors the upstream stable-audio-tools helper closely enough for the
    subset of model code we vendor.
    """

    try:
        import torch  # type: ignore
    except Exception:
        return audio
    if not isinstance(audio, torch.Tensor):
        return audio

    audio = audio.to(device)

    if int(in_sr) != int(target_sr):
        from torchaudio import transforms as T  # type: ignore

        resample_tf = T.Resample(int(in_sr), int(target_sr)).to(device)
        audio = resample_tf(audio)

    from ..data.utils import PadCrop  # type: ignore

    audio = PadCrop(int(target_length), randomize=False)(audio)

    if audio.dim() == 1:
        audio = audio.unsqueeze(0).unsqueeze(0)
    elif audio.dim() == 2:
        audio = audio.unsqueeze(0)

    audio = set_audio_channels(audio, int(target_channels))
    return audio


__all__.append("prepare_audio")
