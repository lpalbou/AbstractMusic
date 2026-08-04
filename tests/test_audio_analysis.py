import io
import struct
import wave

import pytest


def _wav_bytes(samples, sample_rate=48000):
    np = pytest.importorskip("numpy")
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return out.getvalue()


def _float_wav_bytes(samples, sample_rate=48000):
    np = pytest.importorskip("numpy")
    pcm = np.asarray(samples, dtype="<f4")
    data = pcm.tobytes()
    channels = 1
    sample_width = 4
    byte_rate = sample_rate * channels * sample_width
    block_align = channels * sample_width
    fmt = struct.pack("<HHIIHH", 3, channels, sample_rate, byte_rate, block_align, sample_width * 8)
    return (
        b"RIFF"
        + struct.pack("<I", 4 + (8 + len(fmt)) + (8 + len(data)))
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )


@pytest.mark.unit
def test_music_signal_detects_harmonic_tone_as_music_like():
    np = pytest.importorskip("numpy")

    from abstractmusic.audio_analysis import inspect_music_signal_bytes

    sr = 48000
    t = np.arange(0, sr * 3, dtype=np.float64) / float(sr)
    chord = (
        0.22 * np.sin(2 * np.pi * 220.0 * t)
        + 0.17 * np.sin(2 * np.pi * 330.0 * t)
        + 0.12 * np.sin(2 * np.pi * 440.0 * t)
        + 0.08 * np.sin(2 * np.pi * 660.0 * t)
    )
    envelope = 0.72 + 0.12 * np.sin(2 * np.pi * 1.2 * t)
    stats = inspect_music_signal_bytes(_wav_bytes(chord * envelope, sample_rate=sr))

    assert stats.has_harmonic_structure
    assert stats.is_probably_music_like
    assert not stats.is_probably_repetitive_noise


@pytest.mark.unit
def test_wav_inspection_supports_ieee_float_wav():
    np = pytest.importorskip("numpy")

    from abstractmusic.audio_analysis import inspect_wav_bytes

    sr = 48000
    t = np.arange(0, sr, dtype=np.float64) / float(sr)
    stats = inspect_wav_bytes(_float_wav_bytes(0.25 * np.sin(2 * np.pi * 220.0 * t), sample_rate=sr))

    assert stats.sample_rate_hz == sr
    assert stats.channels == 1
    assert stats.sample_width_bytes == 4
    assert stats.duration_s == pytest.approx(1.0)
    assert stats.peak == pytest.approx(0.25, rel=0.01)


@pytest.mark.unit
def test_music_signal_rejects_fast_repetitive_low_harmonic_audio():
    np = pytest.importorskip("numpy")

    from abstractmusic.audio_analysis import inspect_music_signal_bytes

    sr = 48000
    t = np.arange(0, sr * 3, dtype=np.float64) / float(sr)
    rng = np.random.default_rng(123)
    carrier = 0.18 * rng.normal(0.0, 1.0, size=t.shape)
    pulses = (np.sin(2 * np.pi * 12.0 * t) > 0.72).astype(np.float64)
    envelope = 0.08 + 0.92 * pulses
    stats = inspect_music_signal_bytes(_wav_bytes(carrier * envelope, sample_rate=sr))

    assert stats.is_probably_repetitive_noise
    assert not stats.has_harmonic_structure
    assert not stats.is_probably_music_like


@pytest.mark.unit
def test_energy_continuity_flags_long_trailing_pause():
    np = pytest.importorskip("numpy")

    from abstractmusic.audio_analysis import inspect_energy_continuity_bytes

    sr = 48000
    t = np.arange(0, sr * 10, dtype=np.float64) / float(sr)
    tone = 0.25 * np.sin(2 * np.pi * 220.0 * t)
    tone[int(sr * 3.0) :] = 0.0
    stats = inspect_energy_continuity_bytes(_wav_bytes(tone, sample_rate=sr))

    assert stats.has_long_low_energy_gap
    assert stats.has_long_trailing_fade
    assert stats.trailing_low_energy_s >= 6.5
    assert not stats.is_probably_continuous


@pytest.mark.unit
def test_energy_continuity_accepts_continuous_music_bed():
    np = pytest.importorskip("numpy")

    from abstractmusic.audio_analysis import inspect_energy_continuity_bytes

    sr = 48000
    t = np.arange(0, sr * 10, dtype=np.float64) / float(sr)
    bed = 0.18 * np.sin(2 * np.pi * 110.0 * t) + 0.08 * np.sin(2 * np.pi * 330.0 * t)
    bed *= 0.65 + 0.20 * np.sin(2 * np.pi * 2.0 * t)
    stats = inspect_energy_continuity_bytes(_wav_bytes(bed, sample_rate=sr))

    assert stats.max_low_energy_s < 1.0
    assert stats.trailing_low_energy_s < 1.0
    assert stats.is_probably_continuous


def _harmonic_progression(np, sample_rate=48000, duration_s=8.0):
    t = np.arange(0, int(sample_rate * duration_s), dtype=np.float64) / float(sample_rate)
    roots = np.asarray([196.0, 246.94, 293.66, 369.99, 329.63, 261.63, 220.0, 392.0], dtype=np.float64)
    segment = max(1, int(t.size / roots.size))
    root_track = np.repeat(roots, segment)
    if root_track.size < t.size:
        root_track = np.pad(root_track, (0, t.size - root_track.size), mode="edge")
    root_track = root_track[: t.size]
    phase = 2.0 * np.pi * np.cumsum(root_track) / float(sample_rate)
    melody = np.sin(phase) + 0.52 * np.sin(2.0 * phase) + 0.34 * np.sin(3.0 * phase)
    envelope = 0.58 + 0.22 * np.sin(2.0 * np.pi * 0.55 * t) + 0.10 * np.sin(2.0 * np.pi * 1.3 * t)
    return 0.25 * melody * envelope


def _single_note_repetition(np, sample_rate=48000, duration_s=8.0):
    t = np.arange(0, int(sample_rate * duration_s), dtype=np.float64) / float(sample_rate)
    carrier = 0.30 * np.sin(2.0 * np.pi * 220.0 * t) + 0.08 * np.sin(2.0 * np.pi * 440.0 * t)
    gate = 0.18 + 0.82 * (np.sin(2.0 * np.pi * 7.0 * t) > 0.35).astype(np.float64)
    return carrier * gate


def _broadband_clocked_artifact(np, sample_rate=48000, duration_s=8.0):
    t = np.arange(0, int(sample_rate * duration_s), dtype=np.float64) / float(sample_rate)
    base = _harmonic_progression(np, sample_rate=sample_rate, duration_s=duration_s)
    bright = (
        0.035 * np.sin(2.0 * np.pi * 2400.0 * t)
        + 0.025 * np.sin(2.0 * np.pi * 5100.0 * t)
        + 0.018 * np.sin(2.0 * np.pi * 9300.0 * t)
    )
    clock = 0.20 + 0.80 * (np.sin(2.0 * np.pi * 5.0 * t) > 0.15).astype(np.float64)
    return (base + bright) * clock


def _fast_clocked_artifact(np, sample_rate=48000, duration_s=8.0):
    t = np.arange(0, int(sample_rate * duration_s), dtype=np.float64) / float(sample_rate)
    base = _harmonic_progression(np, sample_rate=sample_rate, duration_s=duration_s)
    bright = (
        0.030 * np.sin(2.0 * np.pi * 3200.0 * t)
        + 0.026 * np.sin(2.0 * np.pi * 7400.0 * t)
        + 0.020 * np.sin(2.0 * np.pi * 11800.0 * t)
    )
    clock = 0.18 + 0.82 * (np.sin(2.0 * np.pi * 10.0 * t) > 0.0).astype(np.float64)
    return (base + bright) * clock


def _repeating_high_frequency_pulse_loop(np, sample_rate=48000, duration_s=8.0):
    one_second = np.arange(0, sample_rate, dtype=np.float64) / float(sample_rate)
    carrier = (
        0.08 * np.sin(2.0 * np.pi * 220.0 * one_second)
        + 0.06 * np.sin(2.0 * np.pi * 330.0 * one_second)
        + 0.04 * np.sin(2.0 * np.pi * 3715.0 * one_second)
    )
    clock = 0.12 + 0.88 * (np.sin(2.0 * np.pi * 10.0 * one_second) > 0.0).astype(np.float64)
    repeated = np.tile(carrier * clock, max(1, int(duration_s)))
    return repeated[: int(sample_rate * duration_s)]


def _static_spectral_loop(np, sample_rate=48000, duration_s=8.0):
    one_second = np.arange(0, sample_rate, dtype=np.float64) / float(sample_rate)
    carrier = (
        0.16 * np.sin(2.0 * np.pi * 220.0 * one_second)
        + 0.11 * np.sin(2.0 * np.pi * 330.0 * one_second)
        + 0.07 * np.sin(2.0 * np.pi * 440.0 * one_second)
        + 0.035 * np.sin(2.0 * np.pi * 3715.0 * one_second)
    )
    envelope = 0.58 + 0.18 * np.sin(2.0 * np.pi * 1.0 * one_second)
    repeated = np.tile(carrier * envelope, max(1, int(duration_s)))
    return repeated[: int(sample_rate * duration_s)]


@pytest.mark.unit
def test_harmonic_diversity_flags_single_note_collapse():
    np = pytest.importorskip("numpy")

    from abstractmusic.audio_analysis import inspect_harmonic_diversity_bytes

    stats = inspect_harmonic_diversity_bytes(_wav_bytes(_single_note_repetition(np), sample_rate=48000))

    assert stats.voiced_frame_ratio > 0.20
    assert stats.midi_unique_notes <= 3
    assert stats.midi_pitch_iqr < 1.0
    assert stats.is_probably_single_note_collapse


@pytest.mark.unit
def test_harmonic_diversity_comparison_rejects_repeated_note_against_progression():
    np = pytest.importorskip("numpy")

    from abstractmusic.audio_analysis import compare_harmonic_diversity
    from abstractmusic.audio_analysis import inspect_harmonic_diversity_bytes

    reference = inspect_harmonic_diversity_bytes(_wav_bytes(_harmonic_progression(np), sample_rate=48000))
    candidate = inspect_harmonic_diversity_bytes(_wav_bytes(_single_note_repetition(np), sample_rate=48000))
    comparison = compare_harmonic_diversity(reference, candidate)

    assert reference.midi_unique_notes >= 6
    assert reference.midi_pitch_iqr >= 3.0
    assert candidate.is_probably_single_note_collapse
    assert not comparison.passes_reference_floor


@pytest.mark.unit
def test_harmonic_diversity_comparison_accepts_similar_progression():
    np = pytest.importorskip("numpy")

    from abstractmusic.audio_analysis import compare_harmonic_diversity
    from abstractmusic.audio_analysis import inspect_harmonic_diversity_bytes

    reference_audio = _harmonic_progression(np)
    candidate_audio = 0.92 * _harmonic_progression(np) + 0.015 * np.sin(
        2.0 * np.pi * 880.0 * np.arange(reference_audio.size, dtype=np.float64) / 48000.0
    )
    reference = inspect_harmonic_diversity_bytes(_wav_bytes(reference_audio, sample_rate=48000))
    candidate = inspect_harmonic_diversity_bytes(_wav_bytes(candidate_audio, sample_rate=48000))
    comparison = compare_harmonic_diversity(reference, candidate)

    assert not candidate.is_probably_single_note_collapse
    assert comparison.passes_reference_floor


@pytest.mark.unit
def test_spectrotemporal_modulation_flags_broadband_clock_artifact():
    np = pytest.importorskip("numpy")

    from abstractmusic.audio_analysis import inspect_spectrotemporal_modulation_bytes

    stats = inspect_spectrotemporal_modulation_bytes(
        _wav_bytes(_broadband_clocked_artifact(np), sample_rate=48000)
    )

    assert stats.aligned_band_count >= 3
    assert 4.5 <= stats.aligned_peak_hz <= 5.5
    assert stats.is_probably_broadband_repetition_artifact


@pytest.mark.unit
def test_spectrotemporal_modulation_accepts_unclocked_progression():
    np = pytest.importorskip("numpy")

    from abstractmusic.audio_analysis import inspect_spectrotemporal_modulation_bytes

    stats = inspect_spectrotemporal_modulation_bytes(_wav_bytes(_harmonic_progression(np), sample_rate=48000))

    assert not stats.is_probably_broadband_repetition_artifact


@pytest.mark.unit
def test_spectrotemporal_modulation_flags_fast_high_band_clock_artifact():
    np = pytest.importorskip("numpy")

    from abstractmusic.audio_analysis import inspect_spectrotemporal_modulation_bytes

    stats = inspect_spectrotemporal_modulation_bytes(_wav_bytes(_fast_clocked_artifact(np), sample_rate=48000))

    assert stats.broadband_fast_modulation_ratio_mean >= 0.50
    assert stats.high_band_fast_modulation_ratio_mean >= 0.70
    assert stats.is_probably_fast_clock_artifact
    assert stats.is_probably_broadband_repetition_artifact


@pytest.mark.unit
def test_spectrotemporal_modulation_flags_repeated_high_frequency_pulse_loop():
    np = pytest.importorskip("numpy")

    from abstractmusic.audio_analysis import inspect_spectrotemporal_modulation_bytes

    stats = inspect_spectrotemporal_modulation_bytes(
        _wav_bytes(_repeating_high_frequency_pulse_loop(np), sample_rate=48000)
    )

    assert stats.long_lag_similarity_mean >= 0.85
    assert stats.highband_fast_modulation_ratio >= 0.75
    assert stats.highband_peak_bin_occupancy >= 0.50
    assert stats.is_probably_repeated_high_frequency_pulse
    assert stats.is_probably_broadband_repetition_artifact


@pytest.mark.unit
def test_spectrotemporal_modulation_flags_static_spectral_loop():
    np = pytest.importorskip("numpy")

    from abstractmusic.audio_analysis import inspect_spectrotemporal_modulation_bytes

    stats = inspect_spectrotemporal_modulation_bytes(_wav_bytes(_static_spectral_loop(np), sample_rate=48000))

    assert stats.long_lag_similarity_mean >= 0.70
    assert stats.highband_peak_bin_occupancy >= 0.70
    assert stats.highband_peak_iqr_hz <= 750.0
    assert stats.is_probably_static_spectral_loop
    assert stats.is_probably_broadband_repetition_artifact


def _wind_noise_sweep(np, seconds=12.0, rate=48000, seed=11):
    """Wind/whoosh texture: a dense cluster of slowly drifting inharmonic
    partials with slow amplitude swells and no fast transients — the smooth
    generative failure mode, not raw white noise (which the basic ZCR check
    already catches). The airy high layer uses fixed, widely spaced partials so
    that no inter-partial beating produces spurious fast modulation."""

    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    t = np.arange(n) / rate
    out = np.zeros(n)
    for _ in range(60):
        f0 = rng.uniform(150, 2500)
        drift = 0.008 * f0 * np.sin(2 * np.pi * rng.uniform(0.05, 0.3) * t + rng.uniform(0, 6.28))
        swell = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.08, 0.5) * t + rng.uniform(0, 6.28))
        phase = 2 * np.pi * np.cumsum(f0 + drift) / rate
        out += rng.uniform(0.3, 1.0) * swell * np.sin(phase)
    for k in range(8):  # deterministic airy high layer, slow swells only
        f0 = 4200.0 + 620.0 * k
        swell = 0.5 + 0.5 * np.sin(2 * np.pi * (0.1 + 0.04 * k) * t + 0.7 * k)
        out += 0.05 * swell * np.sin(2 * np.pi * f0 * t)
    return 0.6 * out / np.max(np.abs(out))


def _drumless_chord_progression(np, seconds=12.0, rate=48000):
    """Legitimate percussion-free music: moving triads with evolving brightness."""

    t = np.arange(int(seconds * rate)) / rate
    chords = [(220.0, 277.18, 329.63), (246.94, 311.13, 369.99), (261.63, 329.63, 392.0), (196.0, 246.94, 293.66)]
    out = np.zeros_like(t)
    seg = len(t) // len(chords)
    for i, chord in enumerate(chords):
        idx = slice(i * seg, (i + 1) * seg)
        seg_t = t[idx]
        brightness = 1.0 + 0.8 * np.sin(2.0 * np.pi * 0.25 * seg_t + i)
        for f in chord:
            for harmonic in (1, 2, 3, 4):
                out[idx] += (0.5 / harmonic) * (brightness / (1 + 0.3 * harmonic)) * np.sin(
                    2.0 * np.pi * f * harmonic * seg_t
                )
    return 0.5 * out / np.max(np.abs(out))


@pytest.mark.unit
@pytest.mark.parametrize("seconds", [12.0, 24.0])
@pytest.mark.parametrize("seed", [11, 13, 47])
def test_noise_texture_detector_flags_wind_like_sweep(seconds, seed):
    """The flag must hold across durations and synthesis seeds, not at one
    hand-picked configuration."""

    np = pytest.importorskip("numpy")

    from abstractmusic.audio_analysis import (
        inspect_harmonic_diversity_bytes,
        inspect_spectrotemporal_modulation_bytes,
        is_probably_noise_texture,
    )

    data = _wav_bytes(_wind_noise_sweep(np, seconds=seconds, seed=seed), sample_rate=48000)
    diversity = inspect_harmonic_diversity_bytes(data)
    modulation = inspect_spectrotemporal_modulation_bytes(data)

    # The point of the detector: the basic validity check does NOT catch this.
    assert diversity.wav.is_probably_noise_or_invalid is False
    assert is_probably_noise_texture(diversity, modulation) is True


@pytest.mark.unit
def test_noise_texture_detector_spares_percussion_free_music():
    np = pytest.importorskip("numpy")

    from abstractmusic.audio_analysis import (
        inspect_harmonic_diversity_bytes,
        inspect_spectrotemporal_modulation_bytes,
        is_probably_noise_texture,
    )

    data = _wav_bytes(_drumless_chord_progression(np), sample_rate=48000)
    diversity = inspect_harmonic_diversity_bytes(data)
    modulation = inspect_spectrotemporal_modulation_bytes(data)

    assert is_probably_noise_texture(diversity, modulation) is False


@pytest.mark.unit
def test_noise_texture_detector_defers_to_basic_validity_checks():
    np = pytest.importorskip("numpy")

    from abstractmusic.audio_analysis import (
        inspect_harmonic_diversity_bytes,
        inspect_spectrotemporal_modulation_bytes,
        is_probably_noise_texture,
    )

    silence = _wav_bytes(np.zeros(48000 * 4), sample_rate=48000)
    diversity = inspect_harmonic_diversity_bytes(silence)
    modulation = inspect_spectrotemporal_modulation_bytes(silence)

    # Silent/invalid audio is the basic checks' job; the texture detector stays quiet.
    assert diversity.wav.is_probably_silent is True
    assert is_probably_noise_texture(diversity, modulation) is False


@pytest.mark.unit
def test_quality_gate_set_accepts_music_and_rejects_wind(tmp_path):
    """The canonical gate set the registry validation notes refer to."""

    np = pytest.importorskip("numpy")

    from abstractmusic.audio_analysis import evaluate_music_quality_gates

    reference = tmp_path / "reference.wav"
    reference.write_bytes(_wav_bytes(_drumless_chord_progression(np, seconds=16.0), sample_rate=48000))
    wind = tmp_path / "wind.wav"
    wind.write_bytes(_wav_bytes(_wind_noise_sweep(np, seconds=16.0), sample_rate=48000))

    self_check = evaluate_music_quality_gates(reference, reference)
    assert set(self_check) == {
        "wav_valid",
        "no_noise_texture",
        "no_single_note_collapse",
        "no_low_pitch_variety",
        "passes_reference_floor",
        "no_repetition_artifact",
    }
    assert self_check["wav_valid"] is True
    assert self_check["no_noise_texture"] is True
    assert self_check["passes_reference_floor"] is True

    wind_check = evaluate_music_quality_gates(wind, reference)
    assert wind_check["no_noise_texture"] is False
    assert wind_check["wav_valid"] is True  # which is exactly why the texture gate exists


def _click_track(np, bpm_segments, rate=48000):
    """Kick+hat pattern whose tempo changes across (bpm, seconds) segments."""

    out = []
    for bpm, seconds in bpm_segments:
        n = int(seconds * rate)
        seg = np.zeros(n)
        period = int(rate * 60.0 / bpm)
        for start in range(0, n - 400, period):
            t = np.arange(400) / rate
            seg[start:start + 400] += 0.8 * np.exp(-t * 40.0) * np.sin(2 * np.pi * 180.0 * t)
            seg[start:start + 200] += 0.3 * np.exp(-np.arange(200) / 30.0) * np.sin(
                2 * np.pi * 6000.0 * np.arange(200) / rate
            )
        out.append(seg)
    joined = np.concatenate(out)
    bed_t = np.arange(len(joined)) / rate
    joined = joined + 0.1 * np.sin(2 * np.pi * 110.0 * bed_t)
    return 0.7 * joined / np.max(np.abs(joined))


@pytest.mark.unit
def test_tempo_trajectory_flags_sustained_double_time_opening():
    np = pytest.importorskip("numpy")

    from abstractmusic.audio_analysis import inspect_tempo_trajectory_bytes

    # 12s at exactly double tempo, then 36s settled: the listening-confirmed shape.
    data = _wav_bytes(_click_track(np, [(200.0, 12.0), (100.0, 36.0)]), sample_rate=48000)
    stats = inspect_tempo_trajectory_bytes(data)

    assert stats.opening_is_plateau is True
    assert stats.opening_ratio == pytest.approx(2.0, rel=0.1)
    assert stats.has_probably_double_time_opening is True


@pytest.mark.unit
def test_tempo_trajectory_spares_steady_tracks_and_musical_intro_builds():
    np = pytest.importorskip("numpy")

    from abstractmusic.audio_analysis import inspect_tempo_trajectory_bytes

    steady = _wav_bytes(_click_track(np, [(128.0, 40.0)]), sample_rate=48000)
    assert inspect_tempo_trajectory_bytes(steady).has_probably_double_time_opening is False

    # Gliding intro build (the confirmed-good file's shape): stepwise descent,
    # no flat opening plateau.
    build = _wav_bytes(
        _click_track(np, [(230.0, 5.0), (180.0, 5.0), (140.0, 5.0), (100.0, 30.0)]),
        sample_rate=48000,
    )
    stats = inspect_tempo_trajectory_bytes(build)
    assert stats.has_probably_double_time_opening is False


@pytest.mark.unit
def test_tempo_trajectory_integer_ratio_condition_is_load_bearing():
    """A B-shaped opening (fast plateau over a steady tempo at a NON-integer
    ratio, ~2.68) must not flag. This test exists to kill the mutant that
    removes the integer-ratio condition."""

    np = pytest.importorskip("numpy")

    from abstractmusic.audio_analysis import inspect_tempo_trajectory_bytes

    data = _wav_bytes(_click_track(np, [(201.0, 12.0), (75.0, 36.0)]), sample_rate=48000)
    stats = inspect_tempo_trajectory_bytes(data)

    assert stats.opening_is_plateau is True
    assert stats.opening_ratio > 1.6  # would flag if the integer condition vanished
    assert abs(stats.opening_ratio - round(stats.opening_ratio)) >= 0.08
    assert stats.has_probably_double_time_opening is False


@pytest.mark.unit
def test_tempo_trajectory_plateau_condition_is_load_bearing():
    """A gliding opening that lands NEAR an integer overall ratio must not flag:
    without a flat plateau it is an intro build, not a wrong-tempo opening.
    This test kills the mutant that removes the plateau condition."""

    np = pytest.importorskip("numpy")

    from abstractmusic.audio_analysis import inspect_tempo_trajectory_bytes

    # Opening glides 224 -> 176 (>8% window spread, no plateau) over steady 100:
    # the tracker reads ratio ~1.995 — within 0.005 of an integer — with a
    # consistent steady section, so ONLY the plateau condition prevents the flag.
    data = _wav_bytes(
        _click_track(np, [(224.0, 6.0), (176.0, 6.0), (100.0, 36.0)]), sample_rate=48000
    )
    stats = inspect_tempo_trajectory_bytes(data)

    assert stats.opening_is_plateau is False
    assert stats.steady_is_consistent is True
    assert stats.opening_ratio > 1.6
    assert abs(stats.opening_ratio - round(stats.opening_ratio)) < 0.08
    assert stats.has_probably_double_time_opening is False
