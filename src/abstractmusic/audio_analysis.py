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
class EnergyContinuityStats:
    """Frame-energy continuity metrics for detecting long quiet gaps."""

    wav: WavAudioStats
    frame_s: float
    hop_s: float
    low_energy_floor: float
    low_energy_fraction: float
    max_low_energy_s: float
    max_low_energy_start_s: float
    leading_low_energy_s: float
    trailing_low_energy_s: float
    rms_p10: float
    rms_p50: float
    rms_p75: float

    @property
    def has_long_low_energy_gap(self) -> bool:
        return self.max_low_energy_s >= 6.0

    @property
    def has_long_trailing_fade(self) -> bool:
        return self.trailing_low_energy_s >= 6.0

    @property
    def is_probably_continuous(self) -> bool:
        return not self.has_long_low_energy_gap


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


@dataclass(frozen=True)
class HarmonicDiversityStats:
    """Time-local harmonic and spectral-motion metrics for generated music.

    These metrics are meant to catch collapsed ACE-Step style failures where a
    WAV is non-silent and harmonic but repeats one pitch or one narrow texture.
    They are not a general musical-quality score.
    """

    wav: WavAudioStats
    voiced_frame_ratio: float
    f0_median_hz: float
    f0_iqr_hz: float
    midi_pitch_iqr: float
    midi_unique_notes: int
    spectral_centroid_mean_hz: float
    spectral_centroid_cv: float
    spectral_entropy_mean: float
    spectral_entropy_cv: float
    chroma_effective_rank: float
    chroma_delta_mean: float

    @property
    def has_pitch_diversity(self) -> bool:
        return self.midi_unique_notes >= 8 or self.midi_pitch_iqr >= 3.0

    @property
    def has_spectral_motion(self) -> bool:
        return self.spectral_centroid_cv >= 0.12 and self.spectral_entropy_cv >= 0.08

    @property
    def is_probably_low_pitch_variety(self) -> bool:
        return self.voiced_frame_ratio >= 0.20 and self.midi_unique_notes <= 16 and self.midi_pitch_iqr < 2.0

    @property
    def is_probably_single_note_collapse(self) -> bool:
        if self.wav.is_probably_noise_or_invalid or self.wav.is_probably_silent:
            return False
        narrow_pitch = self.is_probably_low_pitch_variety
        static_spectrum = self.spectral_centroid_cv < 0.20 or self.spectral_entropy_cv < 0.12
        return narrow_pitch and static_spectrum


@dataclass(frozen=True)
class HarmonicDiversityComparison:
    """Reference-relative harmonic diversity comparison."""

    reference: HarmonicDiversityStats
    candidate: HarmonicDiversityStats
    midi_unique_ratio: float
    midi_pitch_iqr_ratio: float
    spectral_centroid_cv_ratio: float
    spectral_entropy_cv_ratio: float
    chroma_rank_ratio: float

    @property
    def passes_reference_floor(self) -> bool:
        if self.candidate.wav.is_probably_noise_or_invalid:
            return False
        if self.candidate.is_probably_single_note_collapse or self.candidate.is_probably_low_pitch_variety:
            return False
        if self.reference.midi_unique_notes >= 8 and self.midi_unique_ratio < 0.45:
            return False
        if self.reference.midi_pitch_iqr >= 3.0 and self.midi_pitch_iqr_ratio < 0.30:
            return False
        if self.reference.spectral_centroid_cv >= 0.12 and self.spectral_centroid_cv_ratio < 0.35:
            return False
        if self.reference.spectral_entropy_cv >= 0.08 and self.spectral_entropy_cv_ratio < 0.35:
            return False
        return True


@dataclass(frozen=True)
class SpectroTemporalModulationStats:
    """Subband envelope modulation metrics for clock-like generated artifacts."""

    wav: WavAudioStats
    subband_edges_hz: tuple[tuple[float, float], ...]
    subband_energy_share: tuple[float, ...]
    subband_peak_hz: tuple[float, ...]
    subband_peak_ratio: tuple[float, ...]
    aligned_peak_hz: float
    aligned_band_count: int
    aligned_band_fraction: float
    aligned_peak_ratio_mean: float
    broadband_repetition_score: float
    broadband_fast_modulation_ratio_mean: float
    high_band_fast_modulation_ratio_mean: float
    modulation_entropy_p25: float
    long_lag_similarity_mean: float
    long_lag_similarity_min: float
    highband_dominant_modulation_hz: float
    highband_fast_modulation_ratio: float
    highband_peak_iqr_hz: float
    highband_peak_bin_occupancy: float

    @property
    def is_probably_broadband_repetition_artifact(self) -> bool:
        slow_pump_artifact = (
            self.aligned_band_count >= 4
            and self.aligned_peak_ratio_mean >= 0.45
            and self.broadband_repetition_score >= 0.35
            and 1.8 <= self.aligned_peak_hz <= 7.0
        )
        fast_clock_artifact = (
            self.broadband_fast_modulation_ratio_mean >= 0.50
            and self.high_band_fast_modulation_ratio_mean >= 0.70
            and self.modulation_entropy_p25 <= 0.35
        )
        return (
            slow_pump_artifact
            or fast_clock_artifact
            or self.is_probably_repeated_high_frequency_pulse
            or self.is_probably_static_spectral_loop
        )

    @property
    def is_probably_slow_pump_artifact(self) -> bool:
        return (
            self.aligned_band_count >= 4
            and self.aligned_peak_ratio_mean >= 0.45
            and self.broadband_repetition_score >= 0.35
            and 1.8 <= self.aligned_peak_hz <= 7.0
        )

    @property
    def is_probably_fast_clock_artifact(self) -> bool:
        return (
            self.broadband_fast_modulation_ratio_mean >= 0.50
            and self.high_band_fast_modulation_ratio_mean >= 0.70
            and self.modulation_entropy_p25 <= 0.35
        )

    @property
    def is_probably_repeated_high_frequency_pulse(self) -> bool:
        return (
            self.long_lag_similarity_mean >= 0.85
            and self.long_lag_similarity_min >= 0.75
            and self.highband_fast_modulation_ratio >= 0.75
            and 6.0 <= self.highband_dominant_modulation_hz <= 20.0
            and self.highband_peak_iqr_hz <= 1200.0
            and self.highband_peak_bin_occupancy >= 0.50
        )

    @property
    def is_probably_static_spectral_loop(self) -> bool:
        return (
            self.long_lag_similarity_mean >= 0.70
            and self.long_lag_similarity_min >= 0.65
            and self.highband_peak_iqr_hz <= 750.0
            and self.highband_peak_bin_occupancy >= 0.70
        )


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


def inspect_energy_continuity_file(path: Union[str, Path]) -> EnergyContinuityStats:
    """Inspect a WAV path for long low-energy gaps or pre-ending fades."""

    expanded = Path(path).expanduser()
    data = expanded.read_bytes()
    return inspect_energy_continuity_bytes(data)


def inspect_energy_continuity_bytes(data: bytes) -> EnergyContinuityStats:
    """Inspect WAV bytes for long low-energy gaps or pre-ending fades."""

    wav_stats = inspect_wav_bytes(data)
    samples = _wav_bytes_to_mono_float_array(data)
    return _inspect_energy_continuity_samples(samples, wav_stats)


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


def inspect_harmonic_diversity_file(path: Union[str, Path]) -> HarmonicDiversityStats:
    """Inspect a WAV path for pitch diversity and time-local spectral motion."""

    expanded = Path(path).expanduser()
    data = expanded.read_bytes()
    return inspect_harmonic_diversity_bytes(data)


def inspect_harmonic_diversity_bytes(data: bytes) -> HarmonicDiversityStats:
    """Inspect WAV bytes for pitch diversity and time-local spectral motion."""

    wav_stats = inspect_wav_bytes(data)
    samples = _wav_bytes_to_mono_float_array(data)
    return _inspect_harmonic_diversity_samples(samples, wav_stats)


def compare_harmonic_diversity_files(
    reference_path: Union[str, Path],
    candidate_path: Union[str, Path],
) -> HarmonicDiversityComparison:
    """Compare a candidate WAV against a reference using collapse-sensitive metrics."""

    return compare_harmonic_diversity(
        inspect_harmonic_diversity_file(reference_path),
        inspect_harmonic_diversity_file(candidate_path),
    )


def compare_harmonic_diversity(
    reference: HarmonicDiversityStats,
    candidate: HarmonicDiversityStats,
) -> HarmonicDiversityComparison:
    """Compare two harmonic-diversity stat blocks."""

    return HarmonicDiversityComparison(
        reference=reference,
        candidate=candidate,
        midi_unique_ratio=float(candidate.midi_unique_notes) / max(float(reference.midi_unique_notes), 1.0),
        midi_pitch_iqr_ratio=float(candidate.midi_pitch_iqr) / max(float(reference.midi_pitch_iqr), 1e-9),
        spectral_centroid_cv_ratio=float(candidate.spectral_centroid_cv) / max(float(reference.spectral_centroid_cv), 1e-9),
        spectral_entropy_cv_ratio=float(candidate.spectral_entropy_cv) / max(float(reference.spectral_entropy_cv), 1e-9),
        chroma_rank_ratio=float(candidate.chroma_effective_rank) / max(float(reference.chroma_effective_rank), 1e-9),
    )


def inspect_spectrotemporal_modulation_file(path: Union[str, Path]) -> SpectroTemporalModulationStats:
    """Inspect subband modulation spectra for broadband repeated artifacts."""

    expanded = Path(path).expanduser()
    data = expanded.read_bytes()
    return inspect_spectrotemporal_modulation_bytes(data)


def inspect_spectrotemporal_modulation_bytes(data: bytes) -> SpectroTemporalModulationStats:
    """Inspect WAV bytes for broadband spectrotemporal modulation dominance."""

    wav_stats = inspect_wav_bytes(data)
    samples = _wav_bytes_to_mono_float_array(data)
    return _inspect_spectrotemporal_modulation_samples(samples, wav_stats)


def _inspect_energy_continuity_samples(samples: Any, wav_stats: WavAudioStats) -> EnergyContinuityStats:
    np = _lazy_import_numpy()
    frame_s = 0.25
    hop_s = 0.10
    if samples.size == 0 or wav_stats.sample_rate_hz <= 0:
        return EnergyContinuityStats(
            wav=wav_stats,
            frame_s=frame_s,
            hop_s=hop_s,
            low_energy_floor=0.0,
            low_energy_fraction=1.0,
            max_low_energy_s=float(wav_stats.duration_s),
            max_low_energy_start_s=0.0,
            leading_low_energy_s=float(wav_stats.duration_s),
            trailing_low_energy_s=float(wav_stats.duration_s),
            rms_p10=0.0,
            rms_p50=0.0,
            rms_p75=0.0,
        )

    sr = int(wav_stats.sample_rate_hz)
    samples = np.asarray(samples, dtype=np.float64)
    frame_size = max(1, int(round(frame_s * float(sr))))
    hop = max(1, int(round(hop_s * float(sr))))
    if samples.size < frame_size:
        frame_rms = np.asarray([float(np.sqrt(np.mean(np.square(samples))))], dtype=np.float64)
    else:
        values = []
        for start in range(0, int(samples.size) - frame_size + 1, hop):
            frame = samples[start : start + frame_size]
            values.append(float(np.sqrt(np.mean(np.square(frame)))))
        frame_rms = np.asarray(values, dtype=np.float64)

    if frame_rms.size == 0:
        frame_rms = np.asarray([0.0], dtype=np.float64)

    rms_p10, rms_p50, rms_p75 = [float(v) for v in np.percentile(frame_rms, [10, 50, 75])]
    low_energy_floor = max(rms_p75 * 0.08, float(wav_stats.rms) * 0.06, 1e-5)
    low = frame_rms < float(low_energy_floor)
    runs: list[tuple[int, int]] = []
    run_start = None
    run_len = 0
    for idx, is_low in enumerate(low.tolist()):
        if bool(is_low):
            if run_start is None:
                run_start = idx
            run_len += 1
            continue
        if run_start is not None:
            runs.append((int(run_start), int(run_len)))
        run_start = None
        run_len = 0
    if run_start is not None:
        runs.append((int(run_start), int(run_len)))

    def _run_duration(start_idx: int, count: int, *, is_trailing: bool = False) -> float:
        start_s = float(start_idx) * hop_s
        if is_trailing:
            return max(0.0, float(wav_stats.duration_s) - start_s)
        return frame_s + max(0, int(count) - 1) * hop_s

    max_start = 0.0
    max_duration = 0.0
    for start_idx, count in runs:
        is_trailing = start_idx + count >= int(low.size)
        duration = _run_duration(start_idx, count, is_trailing=is_trailing)
        if duration > max_duration:
            max_duration = duration
            max_start = float(start_idx) * hop_s

    leading = 0.0
    if runs and runs[0][0] == 0:
        leading = _run_duration(runs[0][0], runs[0][1], is_trailing=(runs[0][0] + runs[0][1] >= int(low.size)))
    trailing = 0.0
    if runs and runs[-1][0] + runs[-1][1] >= int(low.size):
        trailing = _run_duration(runs[-1][0], runs[-1][1], is_trailing=True)

    return EnergyContinuityStats(
        wav=wav_stats,
        frame_s=frame_s,
        hop_s=hop_s,
        low_energy_floor=float(low_energy_floor),
        low_energy_fraction=float(np.mean(low)) if low.size else 0.0,
        max_low_energy_s=float(max_duration),
        max_low_energy_start_s=float(max_start),
        leading_low_energy_s=float(leading),
        trailing_low_energy_s=float(trailing),
        rms_p10=rms_p10,
        rms_p50=rms_p50,
        rms_p75=rms_p75,
    )


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


def _inspect_harmonic_diversity_samples(samples: Any, wav_stats: WavAudioStats) -> HarmonicDiversityStats:
    np = _lazy_import_numpy()
    empty = HarmonicDiversityStats(
        wav=wav_stats,
        voiced_frame_ratio=0.0,
        f0_median_hz=0.0,
        f0_iqr_hz=0.0,
        midi_pitch_iqr=0.0,
        midi_unique_notes=0,
        spectral_centroid_mean_hz=0.0,
        spectral_centroid_cv=0.0,
        spectral_entropy_mean=0.0,
        spectral_entropy_cv=0.0,
        chroma_effective_rank=0.0,
        chroma_delta_mean=0.0,
    )
    if samples.size == 0 or wav_stats.sample_rate_hz <= 0:
        return empty

    sr = int(wav_stats.sample_rate_hz)
    samples = np.asarray(samples, dtype=np.float64)
    samples = samples - float(np.mean(samples))
    if samples.size < 64:
        return empty

    frame_size = min(4096, max(1024, _previous_power_of_two(max(1, samples.size))))
    hop = max(256, frame_size // 4)
    window = np.hanning(frame_size)
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / float(sr))
    audible_mask = (freqs >= 40.0) & (freqs <= 5000.0)
    audible_freqs = freqs[audible_mask]
    if audible_freqs.size == 0:
        return empty

    f0_min_hz = 50.0
    f0_max_hz = 1000.0
    min_lag = max(1, int(sr / f0_max_hz))
    max_lag = min(frame_size - 1, int(sr / f0_min_hz))
    corr_fft_size = _next_power_of_two(frame_size * 2)
    rms_floor = max(float(wav_stats.rms) * 0.08, 1e-5)

    centroids: list[float] = []
    entropies: list[float] = []
    chroma_frames: list[Any] = []
    f0_values: list[float] = []
    total_frames = 0
    voiced = 0

    for start in range(0, max(1, samples.size - frame_size + 1), hop):
        frame = samples[start : start + frame_size]
        if frame.size < frame_size:
            frame = np.pad(frame, (0, frame_size - frame.size))
        frame_rms = float(np.sqrt(np.mean(np.square(frame))))
        if frame_rms < rms_floor:
            continue
        total_frames += 1

        centered = frame - float(np.mean(frame))
        power = np.square(np.abs(np.fft.rfft(centered * window)))
        audible_power = power[audible_mask]
        total_power = float(np.sum(audible_power))
        if total_power <= 1e-12:
            continue

        centroids.append(float(np.sum(audible_freqs * audible_power) / total_power))
        probability = audible_power / total_power
        entropy = -float(np.sum(probability * np.log(probability + 1e-20)))
        entropy /= math.log(float(max(int(probability.size), 2)))
        entropies.append(float(entropy))
        chroma_frames.append(_chroma_frame(np, audible_freqs, audible_power, total_power))

        if max_lag <= min_lag:
            continue
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
        f0_values.append(float(sr) / float(best_lag))
        voiced += 1

    centroid_array = np.asarray(centroids, dtype=np.float64)
    entropy_array = np.asarray(entropies, dtype=np.float64)
    f0_array = np.asarray(f0_values, dtype=np.float64)
    midi = _f0_to_midi(np, f0_array)
    rounded_midi = np.rint(midi).astype(np.int64) if midi.size else np.asarray([], dtype=np.int64)
    chroma_matrix = np.vstack(chroma_frames) if chroma_frames else np.zeros((0, 12), dtype=np.float64)

    return HarmonicDiversityStats(
        wav=wav_stats,
        voiced_frame_ratio=float(voiced) / float(total_frames) if total_frames else 0.0,
        f0_median_hz=float(np.median(f0_array)) if f0_array.size else 0.0,
        f0_iqr_hz=float(np.percentile(f0_array, 75) - np.percentile(f0_array, 25)) if f0_array.size else 0.0,
        midi_pitch_iqr=float(np.percentile(midi, 75) - np.percentile(midi, 25)) if midi.size else 0.0,
        midi_unique_notes=int(np.unique(rounded_midi).size) if rounded_midi.size else 0,
        spectral_centroid_mean_hz=float(np.mean(centroid_array)) if centroid_array.size else 0.0,
        spectral_centroid_cv=_coefficient_of_variation(np, centroid_array),
        spectral_entropy_mean=float(np.mean(entropy_array)) if entropy_array.size else 0.0,
        spectral_entropy_cv=_coefficient_of_variation(np, entropy_array),
        chroma_effective_rank=_effective_rank(np, chroma_matrix),
        chroma_delta_mean=_chroma_delta_mean(np, chroma_matrix),
    )


def _inspect_spectrotemporal_modulation_samples(samples: Any, wav_stats: WavAudioStats) -> SpectroTemporalModulationStats:
    np = _lazy_import_numpy()
    bands = (
        (80.0, 500.0),
        (500.0, 1500.0),
        (1500.0, 3500.0),
        (3500.0, 8000.0),
        (8000.0, 16000.0),
    )
    empty = SpectroTemporalModulationStats(
        wav=wav_stats,
        subband_edges_hz=bands,
        subband_energy_share=tuple(0.0 for _ in bands),
        subband_peak_hz=tuple(0.0 for _ in bands),
        subband_peak_ratio=tuple(0.0 for _ in bands),
        aligned_peak_hz=0.0,
        aligned_band_count=0,
        aligned_band_fraction=0.0,
        aligned_peak_ratio_mean=0.0,
        broadband_repetition_score=0.0,
        broadband_fast_modulation_ratio_mean=0.0,
        high_band_fast_modulation_ratio_mean=0.0,
        modulation_entropy_p25=0.0,
        long_lag_similarity_mean=0.0,
        long_lag_similarity_min=0.0,
        highband_dominant_modulation_hz=0.0,
        highband_fast_modulation_ratio=0.0,
        highband_peak_iqr_hz=0.0,
        highband_peak_bin_occupancy=0.0,
    )
    if samples.size == 0 or wav_stats.sample_rate_hz <= 0:
        return empty

    sr = int(wav_stats.sample_rate_hz)
    samples = np.asarray(samples, dtype=np.float64)
    samples = samples - float(np.mean(samples))
    frame_size = min(4096, max(1024, _previous_power_of_two(max(1, samples.size))))
    hop = max(256, frame_size // 8)
    if samples.size < frame_size:
        return empty

    window = np.hanning(frame_size)
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / float(sr))
    total_mask = (freqs >= bands[0][0]) & (freqs <= bands[-1][1])
    powers: list[Any] = []
    for start in range(0, samples.size - frame_size + 1, hop):
        frame = samples[start : start + frame_size]
        powers.append(np.square(np.abs(np.fft.rfft(frame * window))))
    if not powers:
        return empty
    power_matrix = np.vstack(powers).T
    frame_rate = float(sr) / float(hop)
    total_power = np.sum(power_matrix[total_mask], axis=0) + 1e-18

    energy_shares: list[float] = []
    peak_hz: list[float] = []
    peak_ratios: list[float] = []
    fast_ratios: list[float] = []
    modulation_entropies: list[float] = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        if not bool(np.any(mask)):
            energy_shares.append(0.0)
            peak_hz.append(0.0)
            peak_ratios.append(0.0)
            fast_ratios.append(0.0)
            modulation_entropies.append(0.0)
            continue
        band_power = np.sum(power_matrix[mask], axis=0)
        energy_shares.append(float(np.mean(band_power / total_power)))
        envelope = np.log1p(band_power)
        envelope = envelope - float(np.mean(envelope))
        if envelope.size < 8 or float(np.max(np.abs(envelope))) <= 1e-12:
            peak_hz.append(0.0)
            peak_ratios.append(0.0)
            fast_ratios.append(0.0)
            modulation_entropies.append(0.0)
            continue
        modulation_power = np.square(np.abs(np.fft.rfft(envelope * np.hanning(envelope.size))))
        modulation_freqs = np.fft.rfftfreq(envelope.size, d=1.0 / frame_rate)
        modulation_mask = (modulation_freqs >= 0.5) & (modulation_freqs <= 8.0)
        wide_modulation_mask = (modulation_freqs >= 0.5) & (modulation_freqs <= 20.0)
        if not bool(np.any(modulation_mask)) or not bool(np.any(wide_modulation_mask)):
            peak_hz.append(0.0)
            peak_ratios.append(0.0)
            fast_ratios.append(0.0)
            modulation_entropies.append(0.0)
            continue
        local_power = modulation_power[modulation_mask]
        local_freqs = modulation_freqs[modulation_mask]
        total_modulation_power = float(np.sum(local_power))
        wide_power = modulation_power[wide_modulation_mask]
        total_wide_power = float(np.sum(wide_power))
        if total_modulation_power <= 1e-18 or total_wide_power <= 1e-18:
            peak_hz.append(0.0)
            peak_ratios.append(0.0)
            fast_ratios.append(0.0)
            modulation_entropies.append(0.0)
            continue
        idx = int(np.argmax(local_power))
        peak_hz.append(float(local_freqs[idx]))
        peak_ratios.append(float(local_power[idx] / total_modulation_power))
        fast_mask = (modulation_freqs >= 8.0) & (modulation_freqs <= 20.0)
        fast_ratios.append(float(np.sum(modulation_power[fast_mask]) / total_wide_power))
        probability = wide_power / total_wide_power
        entropy = -float(np.sum(probability * np.log(probability + 1e-20)))
        entropy /= math.log(float(max(int(probability.size), 2)))
        modulation_entropies.append(float(entropy))

    aligned_peak, aligned_count, aligned_mean = _aligned_modulation_peak(np, peak_hz, peak_ratios)
    band_count = len(bands)
    aligned_fraction = float(aligned_count) / float(band_count) if band_count else 0.0
    score = aligned_fraction * aligned_mean
    fast_array = np.asarray(fast_ratios, dtype=np.float64)
    entropy_array = np.asarray(modulation_entropies, dtype=np.float64)
    long_lag_mean, long_lag_min = _long_lag_spectral_similarity(np, power_matrix, freqs, frame_rate)
    high_dom_hz, high_fast_ratio, high_peak_iqr, high_peak_occupancy = _highband_pulse_metrics(
        np,
        power_matrix,
        freqs,
        frame_rate,
    )
    return SpectroTemporalModulationStats(
        wav=wav_stats,
        subband_edges_hz=bands,
        subband_energy_share=tuple(energy_shares),
        subband_peak_hz=tuple(peak_hz),
        subband_peak_ratio=tuple(peak_ratios),
        aligned_peak_hz=float(aligned_peak),
        aligned_band_count=int(aligned_count),
        aligned_band_fraction=aligned_fraction,
        aligned_peak_ratio_mean=float(aligned_mean),
        broadband_repetition_score=float(score),
        broadband_fast_modulation_ratio_mean=float(np.mean(fast_array)) if fast_array.size else 0.0,
        high_band_fast_modulation_ratio_mean=float(np.mean(fast_array[-2:])) if fast_array.size >= 2 else 0.0,
        modulation_entropy_p25=float(np.percentile(entropy_array, 25)) if entropy_array.size else 0.0,
        long_lag_similarity_mean=long_lag_mean,
        long_lag_similarity_min=long_lag_min,
        highband_dominant_modulation_hz=high_dom_hz,
        highband_fast_modulation_ratio=high_fast_ratio,
        highband_peak_iqr_hz=high_peak_iqr,
        highband_peak_bin_occupancy=high_peak_occupancy,
    )


def _aligned_modulation_peak(np: Any, peak_hz: list[float], peak_ratios: list[float]) -> tuple[float, int, float]:
    if not peak_hz:
        return 0.0, 0, 0.0
    rates = np.asarray(peak_hz, dtype=np.float64)
    ratios = np.asarray(peak_ratios, dtype=np.float64)
    candidates = np.where((rates >= 0.5) & (ratios >= 0.30))[0]
    if candidates.size == 0:
        return 0.0, 0, 0.0
    best_rate = 0.0
    best_count = 0
    best_ratio = 0.0
    for idx in candidates:
        close = np.where((np.abs(rates - rates[idx]) <= 0.35) & (ratios >= 0.30))[0]
        count = int(close.size)
        ratio = float(np.mean(ratios[close])) if close.size else 0.0
        if count > best_count or (count == best_count and ratio > best_ratio):
            best_rate = float(np.mean(rates[close])) if close.size else float(rates[idx])
            best_count = count
            best_ratio = ratio
    return best_rate, best_count, best_ratio


def _long_lag_spectral_similarity(np: Any, power_matrix: Any, freqs: Any, frame_rate: float) -> tuple[float, float]:
    mask = (freqs >= 80.0) & (freqs <= 16000.0)
    if not bool(np.any(mask)):
        return 0.0, 0.0
    features = np.log1p(power_matrix[mask].T)
    if features.ndim != 2 or features.shape[0] < 4:
        return 0.0, 0.0
    features = features - np.mean(features, axis=1, keepdims=True)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.maximum(norms, 1e-12)
    similarities: list[float] = []
    for seconds in (1.0, 2.0, 4.0, 8.0):
        lag = int(round(float(seconds) * float(frame_rate)))
        if lag <= 0 or lag >= features.shape[0]:
            continue
        sims = np.sum(features[:-lag] * features[lag:], axis=1)
        if sims.size:
            similarities.append(float(np.mean(sims)))
    if not similarities:
        return 0.0, 0.0
    values = np.asarray(similarities, dtype=np.float64)
    return float(np.mean(values)), float(np.min(values))


def _highband_pulse_metrics(np: Any, power_matrix: Any, freqs: Any, frame_rate: float) -> tuple[float, float, float, float]:
    mask = (freqs >= 2500.0) & (freqs <= 12000.0)
    if not bool(np.any(mask)):
        return 0.0, 0.0, 0.0, 0.0
    high_power = power_matrix[mask]
    high_freqs = freqs[mask]
    band_power = np.sum(high_power, axis=0)
    envelope = np.log1p(band_power)
    envelope = envelope - float(np.mean(envelope))
    dominant_hz = 0.0
    fast_ratio = 0.0
    if envelope.size >= 8 and float(np.max(np.abs(envelope))) > 1e-12:
        modulation_power = np.square(np.abs(np.fft.rfft(envelope * np.hanning(envelope.size))))
        modulation_freqs = np.fft.rfftfreq(envelope.size, d=1.0 / frame_rate)
        wide_mask = (modulation_freqs >= 0.5) & (modulation_freqs <= 35.0)
        fast_mask = (modulation_freqs >= 6.0) & (modulation_freqs <= 35.0)
        if bool(np.any(wide_mask)):
            wide_power = modulation_power[wide_mask]
            wide_total = float(np.sum(wide_power))
            if wide_total > 1e-18:
                wide_freqs = modulation_freqs[wide_mask]
                dominant_hz = float(wide_freqs[int(np.argmax(wide_power))])
                fast_ratio = float(np.sum(modulation_power[fast_mask]) / wide_total) if bool(np.any(fast_mask)) else 0.0

    active = band_power > max(float(np.percentile(band_power, 25)), 1e-18)
    if not bool(np.any(active)):
        return dominant_hz, fast_ratio, 0.0, 0.0
    peak_indices = np.argmax(high_power[:, active], axis=0)
    peak_freqs = high_freqs[peak_indices]
    if peak_freqs.size == 0:
        return dominant_hz, fast_ratio, 0.0, 0.0
    peak_iqr = float(np.percentile(peak_freqs, 75) - np.percentile(peak_freqs, 25))
    bin_width = 500.0
    bins = np.floor((peak_freqs - 2500.0) / bin_width).astype(np.int64)
    counts = np.bincount(bins - int(np.min(bins))) if bins.size else np.asarray([], dtype=np.int64)
    occupancy = float(np.max(counts) / peak_freqs.size) if counts.size and peak_freqs.size else 0.0
    return dominant_hz, fast_ratio, peak_iqr, occupancy


def _f0_to_midi(np: Any, f0_hz: Any) -> Any:
    values = np.asarray(f0_hz, dtype=np.float64)
    values = values[values > 0.0]
    if values.size == 0:
        return values
    return 69.0 + 12.0 * np.log2(values / 440.0)


def _coefficient_of_variation(np: Any, values: Any) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    mean = float(np.mean(arr))
    if abs(mean) <= 1e-12:
        return 0.0
    return float(np.std(arr) / abs(mean))


def _chroma_frame(np: Any, freqs: Any, power: Any, total_power: float) -> Any:
    chroma = np.zeros(12, dtype=np.float64)
    usable = freqs >= 27.5
    if not bool(np.any(usable)):
        return chroma
    midi = np.rint(69.0 + 12.0 * np.log2(freqs[usable] / 440.0)).astype(np.int64)
    np.add.at(chroma, np.mod(midi, 12), power[usable])
    total = float(np.sum(chroma))
    if total <= 1e-12:
        return chroma
    return chroma / max(total_power, total, 1e-12)


def _effective_rank(np: Any, matrix: Any) -> float:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return 0.0
    try:
        singular_values = np.linalg.svd(arr, compute_uv=False)
    except Exception:
        return 0.0
    total = float(np.sum(singular_values))
    if total <= 1e-12:
        return 0.0
    probability = singular_values / total
    entropy = -float(np.sum(probability * np.log(probability + 1e-20)))
    return float(np.exp(entropy))


def _chroma_delta_mean(np: Any, matrix: Any) -> float:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return 0.0
    return float(np.mean(np.linalg.norm(np.diff(arr, axis=0), ord=1, axis=1)))


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
