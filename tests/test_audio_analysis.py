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
