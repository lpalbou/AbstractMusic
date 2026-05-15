"""
Small WAV inspection helpers for smoke tests and manual validation.

The basic WAV checks intentionally use only the Python standard library. The
music-likeness checks load NumPy lazily so a base install stays import-light.
"""

from __future__ import annotations

import io
import math
import struct
import sys
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Union


@dataclass(frozen=True)
class WavAudioStats:
    """Objective stats for a PCM WAV file."""

    sample_rate_hz: int
    channels: int
    sample_width_bytes: int
    frames: int
    duration_s: float
    peak: float
    rms: float
    dc_offset: float
    clipped_ratio: float
    zero_crossing_rate: float

    @property
    def is_probably_silent(self) -> bool:
        return self.peak < 1e-4 or self.rms < 1e-5

    @property
    def has_severe_dc_offset(self) -> bool:
        return abs(self.dc_offset) > 0.10

    @property
    def is_probably_noise_or_invalid(self) -> bool:
        if self.is_probably_silent or self.has_severe_dc_offset:
            return True
        if self.clipped_ratio > 0.05:
            return True
        # White-noise-like signals often have very high ZCR. This is not a
        # complete music classifier, just a useful smoke-test warning.
        return self.zero_crossing_rate > 0.45


@dataclass(frozen=True)
class MusicSignalStats:
    """Objective music smoke metrics for a WAV file.

    This is not a perceptual model and should not be treated as a quality score.
    It is a targeted guardrail for obvious failures: silence, clipping, white
    noise, and short rotor-like pulses that can pass basic WAV validation.
    """

    wav: WavAudioStats
    harmonic_ratio_mean: float
    harmonic_ratio_p75: float
    tonal_peak_concentration_mean: float
    voiced_frame_ratio: float
    f0_median_hz: float
    f0_iqr_hz: float
    envelope_repetition_ratio: float
    beat_modulation_ratio: float
    dominant_envelope_hz: float

    @property
    def has_harmonic_structure(self) -> bool:
        return (
            self.voiced_frame_ratio >= 0.20
            and self.harmonic_ratio_p75 >= 0.22
            and self.tonal_peak_concentration_mean >= 0.12
        )

    @property
    def is_probably_repetitive_noise(self) -> bool:
        fast_envelope_dominates = (
            self.envelope_repetition_ratio >= 0.65
            and 6.0 <= self.dominant_envelope_hz <= 35.0
            and self.beat_modulation_ratio <= 0.35
        )
        narrow_pitch_motion = self.voiced_frame_ratio >= 0.80 and 0.0 < self.f0_iqr_hz <= 40.0
        return (
            fast_envelope_dominates
            and (self.harmonic_ratio_p75 < 0.22 or narrow_pitch_motion)
        )

    @property
    def is_probably_music_like(self) -> bool:
        if self.wav.is_probably_noise_or_invalid:
            return False
        if self.is_probably_repetitive_noise:
            return False
        return self.has_harmonic_structure


def inspect_wav_file(path: Union[str, Path]) -> WavAudioStats:
    """Inspect a WAV file path."""

    with open(Path(path).expanduser(), "rb") as f:
        return inspect_wav_stream(f)


def inspect_wav_bytes(data: bytes) -> WavAudioStats:
    """Inspect WAV bytes."""

    return inspect_wav_stream(io.BytesIO(bytes(data)))


def inspect_wav_stream(stream: BinaryIO) -> WavAudioStats:
    """Inspect a PCM or IEEE-float WAV stream and return objective stats."""

    data = stream.read()
    channels, sample_width, sample_rate, frames, samples = _read_wav_samples(data)

    if channels <= 0:
        raise ValueError("Invalid WAV: channel count must be positive")
    if sample_rate <= 0:
        raise ValueError("Invalid WAV: sample rate must be positive")
    if sample_width not in {1, 2, 4, 8}:
        raise ValueError(f"Unsupported PCM sample width: {sample_width} bytes")
    total = len(samples)
    if total == 0:
        peak = rms = dc = clipped = zcr = 0.0
    else:
        peak = max(abs(x) for x in samples)
        square_sum = math.fsum(float(x) * float(x) for x in samples)
        rms = math.sqrt(square_sum / float(total))
        dc = math.fsum(float(x) for x in samples) / float(total)
        clipped = sum(1 for x in samples if abs(x) >= 0.99) / float(total)
        zcr = _zero_crossing_rate(samples)

    return WavAudioStats(
        sample_rate_hz=sample_rate,
        channels=channels,
        sample_width_bytes=sample_width,
        frames=frames,
        duration_s=float(frames) / float(sample_rate),
        peak=float(peak),
        rms=float(rms),
        dc_offset=float(dc),
        clipped_ratio=float(clipped),
        zero_crossing_rate=float(zcr),
    )


def inspect_music_signal_file(path: Union[str, Path]) -> MusicSignalStats:
    """Inspect a WAV path with harmonic and envelope smoke metrics.

    Requires NumPy at runtime. NumPy remains optional because the base package
    should still be usable for registry and abstraction code without local model
    runtimes.
    """

    expanded = Path(path).expanduser()
    with open(expanded, "rb") as f:
        data = f.read()
    wav_stats = inspect_wav_bytes(data)
    samples = _wav_bytes_to_mono_float_array(data)
    return _inspect_music_samples(samples, wav_stats)


def inspect_music_signal_bytes(data: bytes) -> MusicSignalStats:
    """Inspect WAV bytes with harmonic and envelope smoke metrics."""

    wav_stats = inspect_wav_bytes(data)
    samples = _wav_bytes_to_mono_float_array(data)
    return _inspect_music_samples(samples, wav_stats)


def _pcm_to_float_samples(raw: bytes, *, sample_width: int) -> list[float]:
    if not raw:
        return []
    if sample_width == 1:
        return [(float(b) - 128.0) / 128.0 for b in raw]
    if sample_width == 2:
        values = array("h")
        values.frombytes(raw)
        if values.itemsize != 2:
            raise ValueError("Unexpected platform int16 array size")
        return [float(v) / 32768.0 for v in values]
    values = array("i")
    values.frombytes(raw)
    if values.itemsize != 4:
        raise ValueError("Unexpected platform int32 array size")
    return [float(v) / 2147483648.0 for v in values]


def _ieee_float_to_float_samples(raw: bytes, *, sample_width: int) -> list[float]:
    if not raw:
        return []
    if sample_width == 4:
        values = array("f")
        values.frombytes(raw)
        if values.itemsize != 4:
            raise ValueError("Unexpected platform float32 array size")
    elif sample_width == 8:
        values = array("d")
        values.frombytes(raw)
        if values.itemsize != 8:
            raise ValueError("Unexpected platform float64 array size")
    else:
        raise ValueError(f"Unsupported IEEE float sample width: {sample_width} bytes")
    if sys.byteorder != "little":
        values.byteswap()
    return [float(v) for v in values]


def _wav_bytes_to_mono_float_array(data: bytes) -> Any:
    np = _lazy_import_numpy()
    channels, _sample_width, _sample_rate, _frames, float_samples = _read_wav_samples(data)
    samples = np.asarray(float_samples, dtype=np.float64)
    if channels > 1 and samples.size:
        samples = samples.reshape((-1, channels)).mean(axis=1)
    return samples


def _read_wav_samples(data: bytes) -> tuple[int, int, int, int, list[float]]:
    try:
        with wave.open(io.BytesIO(bytes(data)), "rb") as wf:
            channels = int(wf.getnchannels())
            sample_width = int(wf.getsampwidth())
            sample_rate = int(wf.getframerate())
            frames = int(wf.getnframes())
            raw = wf.readframes(frames)
        return channels, sample_width, sample_rate, frames, _pcm_to_float_samples(raw, sample_width=sample_width)
    except wave.Error as exc:
        if "unknown format: 3" not in str(exc):
            raise

    fmt, raw = _parse_riff_wave(data)
    audio_format, channels, sample_rate, _byte_rate, block_align, bits_per_sample = fmt
    if audio_format != 3:
        raise ValueError(f"Unsupported WAV format tag: {audio_format}")
    sample_width = int(bits_per_sample) // 8
    if sample_width not in {4, 8}:
        raise ValueError(f"Unsupported IEEE float WAV sample width: {sample_width} bytes")
    if channels <= 0 or block_align <= 0:
        raise ValueError("Invalid WAV: channel count and block align must be positive")
    frames = len(raw) // int(block_align)
    return channels, sample_width, sample_rate, frames, _ieee_float_to_float_samples(raw, sample_width=sample_width)


def _parse_riff_wave(data: bytes) -> tuple[tuple[int, int, int, int, int, int], bytes]:
    if len(data) < 12 or data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Invalid WAV: missing RIFF/WAVE header")

    offset = 12
    fmt: tuple[int, int, int, int, int, int] | None = None
    audio_data: bytes | None = None
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        chunk_size = int.from_bytes(data[offset + 4 : offset + 8], byteorder="little", signed=False)
        chunk_start = offset + 8
        chunk_end = min(chunk_start + chunk_size, len(data))
        chunk = data[chunk_start:chunk_end]
        if chunk_id == b"fmt ":
            if len(chunk) < 16:
                raise ValueError("Invalid WAV: fmt chunk too short")
            fmt = struct.unpack("<HHIIHH", chunk[:16])
        elif chunk_id == b"data":
            audio_data = bytes(chunk)
        offset = chunk_end + (chunk_size % 2)

    if fmt is None:
        raise ValueError("Invalid WAV: missing fmt chunk")
    if audio_data is None:
        raise ValueError("Invalid WAV: missing data chunk")
    return fmt, audio_data


def _inspect_music_samples(samples: Any, wav_stats: WavAudioStats) -> MusicSignalStats:
    np = _lazy_import_numpy()
    if samples.size == 0 or wav_stats.sample_rate_hz <= 0:
        return MusicSignalStats(
            wav=wav_stats,
            harmonic_ratio_mean=0.0,
            harmonic_ratio_p75=0.0,
            tonal_peak_concentration_mean=0.0,
            voiced_frame_ratio=0.0,
            f0_median_hz=0.0,
            f0_iqr_hz=0.0,
            envelope_repetition_ratio=0.0,
            beat_modulation_ratio=0.0,
            dominant_envelope_hz=0.0,
        )

    sr = int(wav_stats.sample_rate_hz)
    samples = np.asarray(samples, dtype=np.float64)
    samples = samples - float(np.mean(samples))

    frame_size = min(4096, max(1024, _previous_power_of_two(max(1, samples.size))))
    hop = max(256, frame_size // 4)
    window = np.hanning(frame_size)
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / float(sr))
    audible_mask = (freqs >= 40.0) & (freqs <= 5000.0)
    f0_min_hz = 50.0
    f0_max_hz = 1000.0
    min_lag = max(1, int(sr / f0_max_hz))
    max_lag = min(frame_size - 1, int(sr / f0_min_hz))

    harmonic_ratios: list[float] = []
    peak_concentrations: list[float] = []
    f0_values: list[float] = []
    voiced = 0
    total_frames = 0
    rms_floor = max(float(wav_stats.rms) * 0.08, 1e-5)

    for start in range(0, max(1, samples.size - frame_size + 1), hop):
        frame = samples[start : start + frame_size]
        if frame.size < frame_size:
            frame = np.pad(frame, (0, frame_size - frame.size))
        frame_rms = float(np.sqrt(np.mean(np.square(frame))))
        if frame_rms < rms_floor:
            continue
        total_frames += 1

        centered = frame - float(np.mean(frame))
        spectrum = np.abs(np.fft.rfft(centered * window))
        power = np.square(spectrum)
        audible_power = power[audible_mask]
        total_power = float(np.sum(audible_power))
        if total_power <= 0.0:
            continue

        peak_concentrations.append(_top_energy_ratio(np, audible_power, total_power))

        corr_fft_size = _next_power_of_two(frame_size * 2)
        corr_spec = np.fft.rfft(centered, n=corr_fft_size)
        corr = np.fft.irfft(corr_spec * np.conj(corr_spec), n=corr_fft_size)[:frame_size]
        if corr.size <= max_lag or float(corr[0]) <= 1e-12:
            continue
        lag_window = corr[min_lag : max_lag + 1]
        best_relative = int(np.argmax(lag_window))
        best_lag = min_lag + best_relative
        normalized_corr = float(lag_window[best_relative] / corr[0])
        if normalized_corr < 0.18:
            continue

        f0 = float(sr) / float(best_lag)
        ratio = _harmonic_energy_ratio(np, power, freqs, f0, total_power)
        harmonic_ratios.append(ratio)
        f0_values.append(f0)
        voiced += 1

    env_repetition, beat_ratio, dominant_env = _envelope_metrics(np, samples, sr)
    harmonic_array = np.asarray(harmonic_ratios, dtype=np.float64)
    peak_array = np.asarray(peak_concentrations, dtype=np.float64)
    f0_array = np.asarray(f0_values, dtype=np.float64)

    return MusicSignalStats(
        wav=wav_stats,
        harmonic_ratio_mean=float(np.mean(harmonic_array)) if harmonic_array.size else 0.0,
        harmonic_ratio_p75=float(np.percentile(harmonic_array, 75)) if harmonic_array.size else 0.0,
        tonal_peak_concentration_mean=float(np.mean(peak_array)) if peak_array.size else 0.0,
        voiced_frame_ratio=float(voiced) / float(total_frames) if total_frames else 0.0,
        f0_median_hz=float(np.median(f0_array)) if f0_array.size else 0.0,
        f0_iqr_hz=float(np.percentile(f0_array, 75) - np.percentile(f0_array, 25)) if f0_array.size else 0.0,
        envelope_repetition_ratio=env_repetition,
        beat_modulation_ratio=beat_ratio,
        dominant_envelope_hz=dominant_env,
    )


def _harmonic_energy_ratio(np: Any, power: Any, freqs: Any, f0_hz: float, total_power: float) -> float:
    harmonic_power = 0.0
    max_freq = min(5000.0, float(freqs[-1]))
    harmonic = 1
    while f0_hz * harmonic <= max_freq:
        center = f0_hz * harmonic
        width = max(12.0, center * 0.025)
        mask = (freqs >= center - width) & (freqs <= center + width)
        harmonic_power += float(np.sum(power[mask]))
        harmonic += 1
    return float(min(1.0, harmonic_power / max(total_power, 1e-12)))


def _top_energy_ratio(np: Any, audible_power: Any, total_power: float) -> float:
    if audible_power.size == 0:
        return 0.0
    top_n = min(16, int(audible_power.size))
    if top_n <= 0:
        return 0.0
    top = np.partition(audible_power, -top_n)[-top_n:]
    return float(np.sum(top) / max(total_power, 1e-12))


def _envelope_metrics(np: Any, samples: Any, sample_rate_hz: int) -> tuple[float, float, float]:
    if samples.size < 2 or sample_rate_hz <= 0:
        return 0.0, 0.0, 0.0

    hop = max(1, int(sample_rate_hz / 100.0))
    win = max(hop, int(sample_rate_hz / 50.0))
    values: list[float] = []
    for start in range(0, max(1, samples.size - win + 1), hop):
        frame = samples[start : start + win]
        if frame.size < win:
            break
        values.append(float(np.sqrt(np.mean(np.square(frame)))))
    if len(values) < 8:
        return 0.0, 0.0, 0.0

    env = np.asarray(values, dtype=np.float64)
    env = env - float(np.mean(env))
    if float(np.max(np.abs(env))) <= 1e-12:
        return 0.0, 0.0, 0.0

    window = np.hanning(env.size)
    spec = np.square(np.abs(np.fft.rfft(env * window)))
    freqs = np.fft.rfftfreq(env.size, d=float(hop) / float(sample_rate_hz))
    mod_mask = (freqs >= 0.5) & (freqs <= 35.0)
    if not bool(np.any(mod_mask)):
        return 0.0, 0.0, 0.0

    total = float(np.sum(spec[mod_mask]))
    if total <= 1e-12:
        return 0.0, 0.0, 0.0
    fast_mask = (freqs >= 6.0) & (freqs <= 35.0)
    beat_mask = (freqs >= 0.5) & (freqs < 6.0)
    dominant_index = int(np.argmax(spec[mod_mask]))
    dominant_hz = float(freqs[mod_mask][dominant_index])
    return (
        float(np.sum(spec[fast_mask]) / total),
        float(np.sum(spec[beat_mask]) / total),
        dominant_hz,
    )


def _previous_power_of_two(value: int) -> int:
    value = int(value)
    if value <= 1:
        return 1
    return 1 << (value.bit_length() - 1)


def _next_power_of_two(value: int) -> int:
    value = int(value)
    if value <= 1:
        return 1
    return 1 << ((value - 1).bit_length())


def _lazy_import_numpy() -> Any:
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as e:  # pragma: no cover - exercised without optional deps.
        raise RuntimeError(
            "Music signal inspection requires NumPy. Install an AbstractMusic local model extra "
            "or install numpy explicitly."
        ) from e
    return np


def _zero_crossing_rate(samples: list[float]) -> float:
    if len(samples) < 2:
        return 0.0
    crossings = 0
    previous = samples[0] >= 0.0
    for sample in samples[1:]:
        current = sample >= 0.0
        if current != previous:
            crossings += 1
        previous = current
    return float(crossings) / float(len(samples) - 1)
