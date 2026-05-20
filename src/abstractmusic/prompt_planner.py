"""
Small local prompt/lyrics planner for music generation.

This module is intentionally dependency-free. It is not a replacement for
ACE-Step's 5 Hz LM, but it gives short user queries the richer caption,
metadata, and optional lyric structure that ACE-Step generally responds to
better than a bare phrase.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple


_AUTO_VALUES = {"auto", "generate", "generated", "lyrics", "__auto__"}
_INSTRUMENTAL_VALUES = {"[instrumental]", "[inst]", "instrumental", "inst"}
_PLAN_FIELD_NAMES = {
    "prompt",
    "lyrics",
    "vocal_language",
    "bpm",
    "keyscale",
    "timesignature",
    "enhanced_prompt",
    "structured_prompt",
    "generated_lyrics",
    "instrumental",
    "planner_backend",
    "planner_id",
    "planner_model",
    "generated_fields",
    "confidence",
    "warnings",
}


def _dedupe_tuple(values: Sequence[Any]) -> Tuple[str, ...]:
    out = []
    for value in values:
        text = str(value).strip()
        if text and text not in out:
            out.append(text)
    return tuple(out)


@dataclass(frozen=True)
class MusicPlanningRequest:
    """Input contract for text/music planning providers."""

    prompt: str
    lyrics: Optional[str] = None
    vocal_language: Optional[str] = None
    duration_s: Optional[float] = None
    bpm: Optional[int] = None
    keyscale: Optional[str] = None
    timesignature: Optional[str] = None
    instrumental: bool = False
    enhance_prompt: bool = False
    structure_prompt: bool = True
    auto_lyrics: bool = False
    backend: Optional[str] = None
    model_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "lyrics": self.lyrics,
            "vocal_language": self.vocal_language,
            "duration_s": self.duration_s,
            "bpm": self.bpm,
            "keyscale": self.keyscale,
            "timesignature": self.timesignature,
            "instrumental": self.instrumental,
            "enhance_prompt": self.enhance_prompt,
            "structure_prompt": self.structure_prompt,
            "auto_lyrics": self.auto_lyrics,
            "backend": self.backend,
            "model_id": self.model_id,
        }


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
    planner_backend: str = "deterministic-fallback"
    planner_model: Optional[str] = None
    generated_fields: Tuple[str, ...] = ()
    confidence: Optional[float] = None
    warnings: Tuple[str, ...] = ()
    raw: Dict[str, Any] = field(default_factory=dict)


class MusicPlanningProvider(Protocol):
    """Protocol for objects that can create music prompt plans."""

    def create_plan(self, request: MusicPlanningRequest) -> MusicPromptPlan: ...


@dataclass(frozen=True)
class CompiledMusicPromptPlan:
    """Deterministic rendering of a plan for one backend capability surface."""

    prompt: str
    lyrics: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


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
    _StyleProfile(
        name="arcade_shooter",
        caption=(
            "Original retro arcade space-shooter game music with urgent 16-bit synth bass, "
            "rapid arpeggiated ostinatos, punchy electronic drums, syncopated snare fills, "
            "dark minor-mode chord stabs, alarm-like lead synth hooks, and constant forward momentum. "
            "The cue feels like a side-scrolling spaceship battle: fast, tense, rhythmic, and instrumental."
        ),
        motifs=("orbit", "laser", "engine", "sector", "warning", "vector"),
        bpm=150,
        keyscale="C minor",
    ),
    _StyleProfile(
        name="arcade_platformer",
        caption=(
            "Original colorful retro platform-game music with bouncy chiptune leads, square-wave bass, "
            "bright syncopated drums, playful call-and-response motifs, quick melodic turnarounds, "
            "and a loopable upbeat arrangement. The cue is energetic, whimsical, rhythmic, and instrumental."
        ),
        motifs=("coin", "jump", "pipe", "cloud", "star", "castle"),
        bpm=132,
        keyscale="C major",
    ),
    _StyleProfile(
        name="arcade_action",
        caption=(
            "Original retro video-game action music with a tight drum loop, driving bass line, "
            "clear synth hook, short melodic phrases, quick fills, and loopable section changes. "
            "The cue is active, rhythmic, instrumental, and built for gameplay rather than a ballad."
        ),
        motifs=("level", "score", "spark", "drive", "quest", "signal"),
        bpm=128,
        keyscale="A minor",
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


class DeterministicMusicPlanner:
    """Dependency-free fallback planner.

    This is intentionally conservative. It resolves explicit flags and adds
    low-confidence heuristic structure; it is not a substitute for a real text
    planner supplied by AbstractCore or another host.
    """

    planner_backend = "deterministic-fallback"

    def create_plan(self, request: MusicPlanningRequest) -> MusicPromptPlan:
        return _create_deterministic_prompt_plan(request)


class PassthroughMusicPlanner:
    """Planner mode that preserves user text and explicit metadata only."""

    planner_backend = "off"

    def create_plan(self, request: MusicPlanningRequest) -> MusicPromptPlan:
        raw_prompt = _clean_space(request.prompt) or "music"
        lyrics = request.lyrics.strip() if isinstance(request.lyrics, str) and request.lyrics.strip() else None
        instrumental = bool(request.instrumental) or is_instrumental_lyrics(lyrics) or _looks_instrumental(raw_prompt)
        warnings = []
        if instrumental:
            lyrics = "[Instrumental]"
        elif is_auto_lyrics_value(lyrics) or bool(request.auto_lyrics):
            lyrics = None
            warnings.append("auto_lyrics_ignored_with_planner_off")
        return MusicPromptPlan(
            prompt=raw_prompt,
            lyrics=lyrics,
            vocal_language=request.vocal_language,
            bpm=request.bpm,
            keyscale=request.keyscale,
            timesignature=request.timesignature,
            instrumental=instrumental,
            planner_backend=self.planner_backend,
            generated_fields=(),
            confidence=None,
            warnings=tuple(warnings),
        )


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

    request = MusicPlanningRequest(
        prompt=str(prompt or ""),
        lyrics=lyrics,
        vocal_language=vocal_language,
        duration_s=duration_s,
        bpm=bpm,
        keyscale=keyscale,
        timesignature=timesignature,
        instrumental=instrumental,
        enhance_prompt=enhance_prompt,
        structure_prompt=structure_prompt,
        auto_lyrics=auto_lyrics,
    )
    return DeterministicMusicPlanner().create_plan(request)


def create_music_prompt_plan(
    request: MusicPlanningRequest,
    *,
    provider: Optional[Any] = None,
    mode: str = "deterministic",
) -> MusicPromptPlan:
    """Create a validated plan through an optional provider.

    `provider` is duck-typed so AbstractCore or tests can inject a planner
    without AbstractMusic importing AbstractCore. Supported provider shapes:
    - object with `create_plan(request)`
    - object with `plan_music_text(request_dict)`
    - callable accepting `request_dict`
    """

    mode_s = str(mode or "deterministic").strip().lower()
    if mode_s in {"off", "none", "raw"}:
        return validate_music_prompt_plan(request, PassthroughMusicPlanner().create_plan(request))

    fallback = DeterministicMusicPlanner()
    if provider is None:
        if mode_s == "required":
            raise TypeError("music text planner mode is required, but no planner provider was configured")
        return validate_music_prompt_plan(request, fallback.create_plan(request))
    if mode_s in {"deterministic", "fallback", "local"}:
        return validate_music_prompt_plan(request, fallback.create_plan(request))

    try:
        raw_plan = _call_planning_provider(provider, request)
        plan = _coerce_music_prompt_plan(raw_plan)
        return validate_music_prompt_plan(request, plan)
    except Exception as exc:
        if mode_s == "required":
            raise
        fallback_plan = fallback.create_plan(request)
        warnings = tuple(fallback_plan.warnings) + (f"text_planner_failed:{type(exc).__name__}",)
        return validate_music_prompt_plan(request, replace(fallback_plan, warnings=warnings))


def validate_music_prompt_plan(request: MusicPlanningRequest, plan: MusicPromptPlan) -> MusicPromptPlan:
    """Validate and normalize a planner result without changing the backend contract."""

    warnings = list(plan.warnings)
    prompt = _clean_space(plan.prompt) or _clean_space(request.prompt) or "music"
    lyrics = plan.lyrics.strip() if isinstance(plan.lyrics, str) and plan.lyrics.strip() else None

    user_lyrics = request.lyrics.strip() if isinstance(request.lyrics, str) and request.lyrics.strip() else None
    explicit_auto_lyrics = bool(request.auto_lyrics) or is_auto_lyrics_value(user_lyrics)
    request_instrumental = bool(request.instrumental) or is_instrumental_lyrics(user_lyrics) or _looks_instrumental(request.prompt)
    if request_instrumental:
        lyrics = "[Instrumental]"
    elif user_lyrics and not explicit_auto_lyrics and lyrics != user_lyrics:
        lyrics = user_lyrics
        warnings.append("planner_lyrics_overwrite_ignored")

    bpm = plan.bpm
    if request.bpm is not None:
        bpm = request.bpm
    if bpm is not None:
        try:
            bpm = int(bpm)
        except Exception as exc:
            raise ValueError("planned bpm must be an integer") from exc
        if bpm < 20 or bpm > 300:
            raise ValueError("planned bpm must be in the range 20..300")

    keyscale = request.keyscale if request.keyscale else plan.keyscale
    timesignature = request.timesignature if request.timesignature else plan.timesignature
    if timesignature is not None:
        ts = str(timesignature).strip()
        if not re.fullmatch(r"\d+(?:/\d+)?", ts):
            raise ValueError("planned timesignature must look like '4', '3', or '6/8'")
        timesignature = ts

    vocal_language = request.vocal_language if request.vocal_language else plan.vocal_language
    generated_fields = tuple(str(v) for v in (plan.generated_fields or ()) if str(v).strip())
    return replace(
        plan,
        prompt=prompt,
        lyrics=lyrics,
        vocal_language=vocal_language,
        bpm=bpm,
        keyscale=str(keyscale).strip() if isinstance(keyscale, str) and keyscale.strip() else None,
        timesignature=timesignature,
        instrumental=bool(request_instrumental or plan.instrumental),
        generated_fields=generated_fields,
        warnings=_dedupe_tuple(warnings),
    )


def compile_music_prompt_plan(
    plan: MusicPromptPlan,
    *,
    backend: str,
    native_lyrics_supported: bool,
) -> CompiledMusicPromptPlan:
    """Render a validated plan into prompt/lyrics plus planner metadata."""

    request_prompt = str(plan.prompt or "")
    request_lyrics = plan.lyrics
    warnings = list(plan.warnings)
    if not bool(native_lyrics_supported) and isinstance(request_lyrics, str) and request_lyrics.strip():
        request_prompt = f"{request_prompt}\n\nLyrics:\n{request_lyrics.strip()}"
        request_lyrics = None
        warnings.append("lyrics_folded_into_prompt")

    metadata: Dict[str, Any] = {
        "bpm": plan.bpm,
        "keyscale": plan.keyscale,
        "timesignature": plan.timesignature,
        "vocal_language": plan.vocal_language,
        "enhanced_prompt": plan.enhanced_prompt,
        "structured_prompt": plan.structured_prompt,
        "generated_lyrics": plan.generated_lyrics,
        "instrumental": plan.instrumental,
        "planner_backend": plan.planner_backend,
        "planner_model": plan.planner_model,
        "planner_generated_fields": tuple(plan.generated_fields),
        "planner_confidence": plan.confidence,
        "planner_warnings": _dedupe_tuple(warnings),
        "planner_target_backend": str(backend or ""),
    }
    if plan.raw:
        metadata["planner_raw"] = dict(plan.raw)
    return CompiledMusicPromptPlan(prompt=request_prompt, lyrics=request_lyrics, metadata=metadata)


def _create_deterministic_prompt_plan(request: MusicPlanningRequest) -> MusicPromptPlan:
    raw_prompt = _clean_space(request.prompt) or "music"
    profile = _select_profile(raw_prompt)
    wants_auto_lyrics = bool(request.auto_lyrics) or is_auto_lyrics_value(request.lyrics)
    profile_instrumental = (
        _profile_defaults_instrumental(profile)
        and not wants_auto_lyrics
        and not (isinstance(request.lyrics, str) and request.lyrics.strip())
    )
    wants_instrumental = (
        bool(request.instrumental)
        or is_instrumental_lyrics(request.lyrics)
        or _looks_instrumental(raw_prompt)
        or profile_instrumental
    )
    generated_fields = []
    warnings = []

    effective_lyrics: Optional[str]
    generated_lyrics = False
    if wants_instrumental:
        effective_lyrics = "[Instrumental]"
        generated_fields.append("lyrics")
    elif wants_auto_lyrics:
        effective_lyrics = generate_lyrics(raw_prompt, profile=profile)
        generated_lyrics = True
        generated_fields.append("lyrics")
        warnings.append("template_lyrics")
    elif isinstance(request.lyrics, str) and request.lyrics.strip():
        effective_lyrics = request.lyrics.strip()
    else:
        effective_lyrics = None

    use_structure = bool(request.structure_prompt) and _needs_long_form_structure(request.duration_s)
    should_enhance = bool(request.enhance_prompt) or wants_auto_lyrics or use_structure or _profile_prefers_enhancement(profile)
    effective_prompt = (
        enhance_caption(raw_prompt, profile=profile, duration_s=request.duration_s, include_structure=use_structure)
        if should_enhance
        else raw_prompt
    )
    if effective_prompt != raw_prompt:
        generated_fields.append("prompt")
    inferred_bpm = request.bpm if request.bpm is not None else (profile.bpm if should_enhance or wants_auto_lyrics else None)
    inferred_keyscale = request.keyscale if request.keyscale else (profile.keyscale if should_enhance or wants_auto_lyrics else None)
    inferred_timesignature = (
        request.timesignature if request.timesignature else (profile.timesignature if should_enhance or wants_auto_lyrics else None)
    )
    if request.bpm is None and inferred_bpm is not None:
        generated_fields.append("bpm")
    if not request.keyscale and inferred_keyscale:
        generated_fields.append("keyscale")
    if not request.timesignature and inferred_timesignature:
        generated_fields.append("timesignature")

    return MusicPromptPlan(
        prompt=effective_prompt,
        lyrics=effective_lyrics,
        vocal_language=request.vocal_language,
        bpm=inferred_bpm,
        keyscale=inferred_keyscale,
        timesignature=inferred_timesignature,
        enhanced_prompt=should_enhance,
        structured_prompt=use_structure,
        generated_lyrics=generated_lyrics,
        instrumental=wants_instrumental,
        planner_backend=DeterministicMusicPlanner.planner_backend,
        generated_fields=_dedupe_tuple(generated_fields),
        confidence=0.35 if generated_fields else None,
        warnings=_dedupe_tuple(warnings),
    )


def _call_planning_provider(provider: Any, request: MusicPlanningRequest) -> Any:
    create_plan = getattr(provider, "create_plan", None)
    if callable(create_plan):
        return create_plan(request)
    plan_music_text = getattr(provider, "plan_music_text", None)
    if callable(plan_music_text):
        return plan_music_text(request.to_dict())
    if callable(provider):
        return provider(request.to_dict())
    raise TypeError("music text planner must be callable or expose create_plan/plan_music_text")


def _coerce_music_prompt_plan(value: Any) -> MusicPromptPlan:
    if isinstance(value, MusicPromptPlan):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("music text planner must return MusicPromptPlan or mapping")
    raw = dict(value)

    def _optional_str(name: str) -> Optional[str]:
        item = raw.get(name)
        if item is None:
            return None
        text = str(item).strip()
        return text or None

    def _tuple_str(name: str) -> Tuple[str, ...]:
        item = raw.get(name)
        if item is None:
            return ()
        if isinstance(item, str):
            return (item,) if item.strip() else ()
        if isinstance(item, Sequence):
            return tuple(str(v).strip() for v in item if str(v).strip())
        return (str(item).strip(),)

    confidence = raw.get("confidence")
    try:
        confidence_f = float(confidence) if confidence is not None else None
    except Exception:
        confidence_f = None

    return MusicPromptPlan(
        prompt=str(raw.get("prompt") or ""),
        lyrics=_optional_str("lyrics"),
        vocal_language=_optional_str("vocal_language"),
        bpm=int(raw["bpm"]) if raw.get("bpm") is not None else None,
        keyscale=_optional_str("keyscale"),
        timesignature=_optional_str("timesignature"),
        enhanced_prompt=bool(raw.get("enhanced_prompt", False)),
        structured_prompt=bool(raw.get("structured_prompt", False)),
        generated_lyrics=bool(raw.get("generated_lyrics", False)),
        instrumental=bool(raw.get("instrumental", False)),
        planner_backend=str(raw.get("planner_backend") or raw.get("planner_id") or "injected"),
        planner_model=_optional_str("planner_model"),
        generated_fields=_tuple_str("generated_fields"),
        confidence=confidence_f,
        warnings=_tuple_str("warnings"),
        raw={k: v for k, v in raw.items() if k not in _PLAN_FIELD_NAMES},
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
    continuity_text = f" {_continuity_constraint(prof, duration_s)}"
    return (
        f"{clean}. {prof.caption}{duration_text}{structure_text}{continuity_text} "
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
    if any(
        k in text
        for k in (
            "r-type",
            "rtype",
            "r type",
            "space shooter",
            "shooter game",
            "shoot'em up",
            "shoot em up",
            "shmup",
            "bullet hell",
            "spaceship",
            "space battle",
            "side-scrolling shooter",
            "scrolling shooter",
        )
    ):
        return _profile_by_name("arcade_shooter")
    if any(k in text for k in ("mario", "mario bros", "mario bross", "platformer", "platform game", "jump and run")):
        return _profile_by_name("arcade_platformer")
    if any(k in text for k in ("video game", "game music", "arcade", "chiptune", "8-bit", "16-bit", "8 bit", "16 bit")):
        return _profile_by_name("arcade_action")
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


def _profile_defaults_instrumental(profile: _StyleProfile) -> bool:
    return profile.name in {"arcade_shooter", "arcade_platformer", "arcade_action"}


def _profile_by_name(name: str) -> _StyleProfile:
    for profile in _STYLE_PROFILES:
        if profile.name == name:
            return profile
    return _DEFAULT_PROFILE


def _profile_prefers_enhancement(profile: _StyleProfile) -> bool:
    return profile.name in {"arcade_shooter", "arcade_platformer", "arcade_action"}


def _continuity_constraint(profile: _StyleProfile, duration_s: Optional[float]) -> str:
    try:
        long_form = duration_s is not None and float(duration_s) >= 45.0
    except (TypeError, ValueError):
        long_form = False
    if profile.name in {"arcade_shooter", "arcade_platformer", "arcade_action"}:
        return (
            "Keep a continuous gameplay loop: drums, bass, or arpeggio remain audible through every transition; "
            "use quick fills and pickups between sections, with no empty gaps and no long fade-out before the final hit."
        )
    if long_form:
        return (
            "Keep audible musical continuity through section changes; use fills, pickups, or crossfades instead of empty gaps, "
            "and keep the ending musical until the final beat."
        )
    return "Use smooth transitions and keep the musical pulse audible."


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
        "arcade_shooter": (
            "tight electronic drum loop and pulsing minor synth bass launch immediately",
            "rapid arpeggio ostinato enters with short snare fills",
            "lead synth states a tense space-battle hook over driving percussion",
            "brief danger bridge shifts harmony while the bass pulse continues",
            "full-intensity shooter theme with extra counterline and crash hits",
            "hook returns with faster fills and brighter lead variations",
            "loopable final hit that stays energetic to the end",
        ),
        "arcade_platformer": (
            "bouncy square-wave bass and bright drum groove start the loop",
            "playful lead motif answers with short melodic jumps",
            "main platform theme opens with syncopated percussion",
            "brief bonus-room bridge changes harmony while the rhythm keeps moving",
            "full cheerful hook with octave lead and quick drum fills",
            "theme reprise adds a counter-melody",
            "loopable final tag with a clean upbeat hit",
        ),
        "arcade_action": (
            "drum loop and bass line establish gameplay momentum",
            "short synth motif enters with quick fills",
            "main hook opens into a stronger action groove",
            "compact bridge changes harmony without dropping the pulse",
            "full action theme peaks with layered lead and percussion",
            "hook reprise adds variation and fills",
            "loopable final tag that remains audible to the end",
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
