import argparse

import pytest


@pytest.mark.unit
def test_prompt_planner_enhances_heroic_fantasy_prompt():
    from abstractmusic.prompt_planner import create_prompt_plan

    plan = create_prompt_plan(
        "an heroic fantasy epic music in the same style as conan the barbarian",
        duration_s=30,
        enhance_prompt=True,
    )

    assert plan.enhanced_prompt is True
    assert "French horns" in plan.prompt
    assert "timpani" in plan.prompt
    assert "Target duration is about 30 seconds" in plan.prompt
    assert plan.bpm == 96
    assert plan.keyscale == "D minor"
    assert plan.timesignature == "4"
    assert plan.lyrics is None


@pytest.mark.unit
def test_prompt_planner_adds_long_form_structure_by_default():
    from abstractmusic.prompt_planner import create_prompt_plan

    plan = create_prompt_plan(
        "an heroic fantasy epic music in the same style as conan the barbarian",
        duration_s=120,
    )

    assert plan.enhanced_prompt is True
    assert plan.structured_prompt is True
    assert "Long-form structure:" in plan.prompt
    assert "0:00-" in plan.prompt
    assert "full brass and wide drums reach the climax" in plan.prompt
    assert "more than 8 bars" in plan.prompt
    assert plan.bpm == 96
    assert plan.keyscale == "D minor"


@pytest.mark.unit
def test_prompt_planner_can_disable_long_form_structure():
    from abstractmusic.prompt_planner import create_prompt_plan

    plan = create_prompt_plan("heroic fantasy music", duration_s=120, structure_prompt=False)

    assert plan.enhanced_prompt is False
    assert plan.structured_prompt is False
    assert "Long-form structure:" not in plan.prompt
    assert plan.bpm is None


@pytest.mark.unit
def test_prompt_planner_auto_lyrics_generates_structured_sections():
    from abstractmusic.prompt_planner import create_prompt_plan

    plan = create_prompt_plan("heroic fantasy battle song", lyrics="auto", vocal_language="en")

    assert plan.generated_lyrics is True
    assert plan.enhanced_prompt is True
    assert plan.vocal_language == "en"
    assert plan.lyrics is not None
    assert "[Verse 1]" in plan.lyrics
    assert "[Chorus]" in plan.lyrics
    assert "Raise the" in plan.lyrics


@pytest.mark.unit
def test_prompt_planner_instrumental_overrides_auto_lyrics():
    from abstractmusic.prompt_planner import create_prompt_plan

    plan = create_prompt_plan("instrumental cinematic fantasy", lyrics="auto", auto_lyrics=True)

    assert plan.instrumental is True
    assert plan.generated_lyrics is False
    assert plan.lyrics == "[Instrumental]"


@pytest.mark.unit
def test_cli_resolves_generation_text_for_acestep_and_generic_backend(capsys):
    from abstractmusic.cli import _resolve_generation_text

    ace_args = argparse.Namespace(
        backend="acestep-diffusers",
        duration=30.0,
        vocal_language="en",
        bpm=None,
        keyscale=None,
        timesignature=None,
        instrumental=False,
        enhance_prompt=True,
        structure_prompt=True,
        auto_lyrics=True,
        print_plan=True,
    )
    prompt, lyrics, meta = _resolve_generation_text(ace_args, "heroic fantasy song", None)

    assert "French horns" in prompt
    assert lyrics is not None and "[Chorus]" in lyrics
    assert meta["bpm"] == 96
    assert meta["structured_prompt"] is False
    assert meta["generated_lyrics"] is True
    assert "Effective music plan:" in capsys.readouterr().err

    generic_args = argparse.Namespace(**{**vars(ace_args), "backend": "musicgen", "print_plan": False})
    prompt, lyrics, meta = _resolve_generation_text(generic_args, "heroic fantasy song", "auto")

    assert lyrics is None
    assert "\n\nLyrics:\n" in prompt
    assert meta["generated_lyrics"] is True
