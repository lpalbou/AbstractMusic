"""
Small local prompt/lyrics planner for music generation.

This module is intentionally dependency-free. It is not a replacement for
ACE-Step's 5 Hz LM, but it gives short user queries the richer caption,
metadata, and optional lyric structure that ACE-Step generally responds to
better than a bare phrase.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


_AUTO_VALUES = {"auto", "generate", "generated", "lyrics", "__auto__"}
_INSTRUMENTAL_VALUES = {"[instrumental]", "[inst]", "instrumental", "inst"}


@dataclass(frozen=True)
class MusicPromptPlan:
    """Resolved text conditioning for a music generation request."""

    prompt: str
    lyrics: Optional[str]
    vocal_language: Optional[str] = None
    bpm: Optional[int] = None
    keyscale: Optional[str] = None
    timesignature: Optional[str] = None
    enhanced_prompt: bool = False
    structured_prompt: bool = False
    generated_lyrics: bool = False
    instrumental: bool = False


@dataclass(frozen=True)
class _StyleProfile:
    name: str
    caption: str
    motifs: tuple[str, ...]
    bpm: int
    keyscale: str
    timesignature: str = "4"


_STYLE_PROFILES = (
    _StyleProfile(
        name="heroic_fantasy",
        caption=(
            "A heroic fantasy orchestral cue with a dramatic minor-mode theme, "
            "low strings, bold French horns, trombones, timpani, taiko-like war drums, "
            "wide cinematic percussion, solemn choir pads, and a rising adventure arc. "
            "It begins with a restrained modal motif, builds through layered ostinato strings, "
            "opens into a powerful brass-led chorus, then resolves with a memorable noble theme."
        ),
        motifs=("stone", "fire", "steel", "storm", "crown", "dawn"),
        bpm=96,
        keyscale="D minor",
    ),
    _StyleProfile(
        name="ambient_lofi",
        caption=(
            "Warm ambient lo-fi study music with soft dusty drums, mellow electric piano chords, "
            "subtle vinyl texture, rounded bass, evolving pads, gentle melodic fragments, "
            "and relaxed section changes that avoid looping one repeated note."
        ),
        motifs=("window", "rain", "lamp", "paper", "night", "city"),
        bpm=78,
        keyscale="C major",
    ),
    _StyleProfile(
        name="synthwave",
        caption=(
            "Melodic synthwave with analog arpeggios, wide gated drums, pulsing bass, "
            "bright lead synth hooks, lush pads, and a clear verse-to-chorus lift."
        ),
        motifs=("neon", "highway", "signal", "midnight", "horizon", "stars"),
        bpm=112,
        keyscale="A minor",
    ),
    _StyleProfile(
        name="rock",
        caption=(
            "Full-band rock music with tight drums, driving bass, layered rhythm guitars, "
            "a clear lead motif, dynamic fills, and a strong chorus payoff."
        ),
        motifs=("road", "spark", "thunder", "heart", "fire", "night"),
        bpm=128,
        keyscale="E minor",
    ),
    _StyleProfile(
        name="jazz",
        caption=(
            "Expressive jazz arrangement with brushed drums, walking bass, warm piano voicings, "
            "tasteful horn phrases, swing feel, and improvised melodic call-and-response."
        ),
        motifs=("moon", "smoke", "corner", "blue", "glass", "morning"),
        bpm=92,
        keyscale="Bb major",
    ),
    _StyleProfile(
        name="piano",
        caption=(
            "Emotional piano-led composition with clear chord movement, lyrical right-hand melody, "
            "supportive low register, subtle strings, and a natural cinematic build."
        ),
        motifs=("light", "river", "memory", "home", "winter", "breath"),
        bpm=72,
        keyscale="G major",
    ),
    _StyleProfile(
        name="electronic",
        caption=(
            "Polished electronic music with a defined groove, evolving bass movement, crisp drums, "
            "layered synth motifs, filtered transitions, and a strong musical arc."
        ),
        motifs=("pulse", "signal", "glass", "gravity", "motion", "echo"),
        bpm=124,
        keyscale="F minor",
    ),
)

_DEFAULT_PROFILE = _StyleProfile(
    name="general",
    caption=(
        "A complete, coherent music arrangement with a clear intro, theme development, contrasting middle section, "
        "and satisfying outro. Use varied harmony, evolving instrumentation, natural dynamics, and melodic movement "
        "instead of a static repeated tone."
    ),
    motifs=("light", "heart", "sky", "night", "road", "dream"),
    bpm=100,
    keyscale="C major",
)


def is_auto_lyrics_value(value: object) -> bool:
    """Return whether a user lyrics value requests automatic lyric generation."""

    return isinstance(value, str) and value.strip().lower() in _AUTO_VALUES


def is_instrumental_lyrics(value: object) -> bool:
    """Return whether lyrics explicitly request instrumental output."""

    return isinstance(value, str) and value.strip().lower() in _INSTRUMENTAL_VALUES


def create_prompt_plan(
    prompt: str,
    *,
    lyrics: Optional[str] = None,
    vocal_language: Optional[str] = None,
    duration_s: Optional[float] = None,
    bpm: Optional[int] = None,
    keyscale: Optional[str] = None,
    timesignature: Optional[str] = None,
    instrumental: bool = False,
    enhance_prompt: bool = False,
    structure_prompt: bool = True,
    auto_lyrics: bool = False,
) -> MusicPromptPlan:
    """Create richer text conditioning from a short user music query."""

    raw_prompt = _clean_space(prompt) or "music"
    profile = _select_profile(raw_prompt)
    wants_auto_lyrics = bool(auto_lyrics) or is_auto_lyrics_value(lyrics)
    wants_instrumental = bool(instrumental) or is_instrumental_lyrics(lyrics) or _looks_instrumental(raw_prompt)

    effective_lyrics: Optional[str]
    generated_lyrics = False
    if wants_instrumental:
        effective_lyrics = "[Instrumental]"
    elif wants_auto_lyrics:
        effective_lyrics = generate_lyrics(raw_prompt, profile=profile)
        generated_lyrics = True
    elif isinstance(lyrics, str) and lyrics.strip():
        effective_lyrics = lyrics.strip()
    else:
        effective_lyrics = None

    use_structure = bool(structure_prompt) and _needs_long_form_structure(duration_s)
    should_enhance = bool(enhance_prompt) or wants_auto_lyrics or use_structure
    effective_prompt = (
        enhance_caption(raw_prompt, profile=profile, duration_s=duration_s, include_structure=use_structure)
        if should_enhance
        else raw_prompt
    )

    return MusicPromptPlan(
        prompt=effective_prompt,
        lyrics=effective_lyrics,
        vocal_language=vocal_language,
        bpm=bpm if bpm is not None else (profile.bpm if should_enhance or wants_auto_lyrics else None),
        keyscale=keyscale if keyscale else (profile.keyscale if should_enhance or wants_auto_lyrics else None),
        timesignature=timesignature if timesignature else (profile.timesignature if should_enhance or wants_auto_lyrics else None),
        enhanced_prompt=should_enhance,
        structured_prompt=use_structure,
        generated_lyrics=generated_lyrics,
        instrumental=wants_instrumental,
    )


def enhance_caption(
    prompt: str,
    *,
    profile: Optional[_StyleProfile] = None,
    duration_s: Optional[float] = None,
    include_structure: bool = False,
) -> str:
    """Expand a short caption into a detailed music-generation caption."""

    clean = _clean_space(prompt) or "music"
    prof = profile or _select_profile(clean)
    duration_text = ""
    if duration_s is not None and float(duration_s) > 0:
        duration_text = f" Target duration is about {int(round(float(duration_s)))} seconds."
    structure_text = f" {_long_form_structure_text(prof, duration_s)}" if include_structure else ""
    return (
        f"{clean}. {prof.caption}{duration_text}{structure_text} "
        "Keep the arrangement musical and varied with changing chord voicings, counter-melodies, fills, "
        "and evolving texture across sections."
    )


def generate_lyrics(prompt: str, *, profile: Optional[_StyleProfile] = None) -> str:
    """Generate simple structured lyrics from a prompt without external services."""

    clean = _clean_space(prompt) or "music"
    prof = profile or _select_profile(clean)
    words = _theme_words(clean, prof)
    a, b, c, d, e, f = words[:6]
    title = _title_from_prompt(clean)
    return "\n".join(
        [
            f"[Intro - {title}]",
            f"{a.capitalize()} wakes beneath the {b}",
            f"Drums are calling through the {c}",
            "",
            "[Verse 1]",
            f"We walk the line where the {d} begins",
            f"With open hands and weathered skin",
            f"The {e} is rising, the road is wide",
            f"We carry the {f} through the turning tide",
            "",
            "[Pre-Chorus]",
            f"Hold the rhythm, hold the flame",
            "Every shadow learns our name",
            "",
            "[Chorus]",
            f"Raise the {a}, let the {b} sing",
            f"Through the {c}, hear the echoes ring",
            f"We are the spark, we are the sign",
            f"Moving as one across the line",
            "",
            "[Bridge]",
            f"If the {d} falls and the night is long",
            f"We answer back with a louder song",
            "",
            "[Final Chorus]",
            f"Raise the {a}, let the {b} sing",
            f"Through the {c}, hear the echoes ring",
            f"We are the spark, we are the sign",
            "Fading into the morning light",
        ]
    )


def _select_profile(prompt: str) -> _StyleProfile:
    text = prompt.lower()
    if any(k in text for k in ("fantasy", "heroic", "epic", "barbarian", "conan", "orchestral", "cinematic", "battle")):
        return _STYLE_PROFILES[0]
    if any(k in text for k in ("lo-fi", "lofi", "ambient", "study", "chill", "downtempo")):
        return _STYLE_PROFILES[1]
    if any(k in text for k in ("synthwave", "retrowave", "cyber", "neon")):
        return _STYLE_PROFILES[2]
    if any(k in text for k in ("rock", "guitar", "punk", "metal")):
        return _STYLE_PROFILES[3]
    if any(k in text for k in ("jazz", "swing", "bebop", "sax")):
        return _STYLE_PROFILES[4]
    if any(k in text for k in ("piano", "keys", "solo piano")):
        return _STYLE_PROFILES[5]
    if any(k in text for k in ("electronic", "edm", "techno", "house", "trance", "dance")):
        return _STYLE_PROFILES[6]
    return _DEFAULT_PROFILE


def _looks_instrumental(prompt: str) -> bool:
    text = prompt.lower()
    return any(k in text for k in ("instrumental", "no vocals", "without vocals", "no lyrics"))


def _needs_long_form_structure(duration_s: Optional[float]) -> bool:
    if duration_s is None:
        return False
    try:
        return float(duration_s) >= 45.0
    except (TypeError, ValueError):
        return False


def _long_form_structure_text(profile: _StyleProfile, duration_s: Optional[float]) -> str:
    try:
        duration = max(45.0, float(duration_s or 0.0))
    except (TypeError, ValueError):
        duration = 60.0
    sections = _section_phrases(profile)
    edges = (0.0, 0.12, 0.28, 0.45, 0.63, 0.80, 0.93, 1.0)
    parts = []
    for idx, phrase in enumerate(sections):
        start = _format_time(duration * edges[idx])
        end = _format_time(duration * edges[idx + 1])
        parts.append(f"{start}-{end} {phrase}")
    return (
        "Long-form structure: "
        + "; ".join(parts)
        + ". Add a clear change at each section boundary and avoid holding one motif for more than 8 bars."
    )


def _section_phrases(profile: _StyleProfile) -> tuple[str, ...]:
    by_profile = {
        "heroic_fantasy": (
            "sparse modal motif in low strings with distant war drums",
            "ostinato strings enter with horn responses",
            "brass main theme expands over timpani",
            "contrasting bridge with choir pads and lighter percussion",
            "full brass and wide drums reach the climax",
            "main theme returns with counter-melody",
            "final cadence and cinematic release",
        ),
        "ambient_lofi": (
            "warm keys and room tone introduce the harmony",
            "dusty drums and rounded bass establish the groove",
            "new melodic fragment answers the first theme",
            "filtered breakdown with pads and soft texture",
            "full groove returns with extra chord color",
            "reduced drums and bass variation",
            "gentle outro with fading keys",
        ),
        "synthwave": (
            "pulsing bass fades in under wide pads",
            "analog arpeggio and gated drums establish motion",
            "bright lead hook states the main theme",
            "breakdown with filtered pads and tom fills",
            "full chorus lift with lead synth counterline",
            "bass variation and reprise of the hook",
            "wide pad outro with fading arpeggio",
        ),
        "rock": (
            "drums and muted guitar introduce the riff",
            "driving bass locks with rhythm guitars",
            "lead motif opens into the main section",
            "contrasting bridge with lighter dynamics",
            "full-band chorus payoff with fills",
            "riff returns with added lead guitar",
            "tight final hit and decay",
        ),
        "jazz": (
            "piano voicings and brushed cymbals set the room",
            "walking bass enters with a relaxed swing feel",
            "horn phrase states the main theme",
            "improvised call-and-response bridge",
            "ensemble lift with stronger ride pattern",
            "theme reprise with tasteful fills",
            "soft tag ending",
        ),
        "piano": (
            "simple left-hand pulse and quiet melody",
            "fuller chords support the main theme",
            "strings enter under a higher piano line",
            "contrasting middle with softer dynamics",
            "cinematic build and widened harmony",
            "theme reprise in a higher register",
            "resolved final chord",
        ),
        "electronic": (
            "filtered groove and sub pulse fade in",
            "crisp drums and bass movement establish the rhythm",
            "lead synth motif defines the hook",
            "breakdown with texture and rising filter",
            "full drop with layered synth counterline",
            "groove variation and hook reprise",
            "stripped outro with echo tail",
        ),
    }
    return by_profile.get(
        profile.name,
        (
            "intro states a simple motif and tonal center",
            "main theme enters with a stronger groove",
            "second layer adds counter-melody and chord movement",
            "contrasting middle section changes texture",
            "full arrangement reaches the peak",
            "main theme returns with variation",
            "outro resolves the harmony",
        ),
    )


def _format_time(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    minutes, secs = divmod(total, 60)
    return f"{minutes}:{secs:02d}"


def _theme_words(prompt: str, profile: _StyleProfile) -> tuple[str, ...]:
    words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", prompt)]
    stop = {
        "the",
        "and",
        "with",
        "without",
        "music",
        "song",
        "style",
        "same",
        "like",
        "for",
        "that",
        "this",
        "from",
        "into",
    }
    picked = []
    for word in words:
        cleaned = word.strip("-'")
        if cleaned and cleaned not in stop and cleaned not in picked:
            picked.append(cleaned)
    for word in profile.motifs:
        if word not in picked:
            picked.append(word)
    return tuple((picked + list(_DEFAULT_PROFILE.motifs))[:6])


def _title_from_prompt(prompt: str) -> str:
    words = [w.capitalize() for w in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", prompt)[:3]]
    return " ".join(words) if words else "Theme"


def _clean_space(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()
