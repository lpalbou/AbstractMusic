import pytest


@pytest.mark.unit
def test_cli_help_does_not_error():
    from abstractmusic.cli import main

    # Should exit cleanly and not attempt model downloads/import heavy stacks.
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0


@pytest.mark.unit
def test_cli_allows_common_flags_after_subcommand():
    # Argparse subparsers normally reject top-level options placed after the subcommand.
    # We intentionally support the natural ordering used in docs:
    #   abstractmusic --backend acestep t2m "..." --duration 10 --out out.wav
    from abstractmusic.cli import build_parser

    args = build_parser().parse_args(
        ["--backend", "acestep", "t2m", "sci fi music", "--duration", "10", "--out", "out.wav"]
    )
    assert args.cmd == "t2m"
    assert float(args.duration) == 10.0


@pytest.mark.unit
def test_cli_accepts_acestep_diffusers_backend():
    from abstractmusic.cli import build_parser

    args = build_parser().parse_args(
        ["--backend", "acestep-diffusers", "t2m", "sci fi music", "--duration", "5", "--out", "out.wav"]
    )
    assert args.backend == "acestep-diffusers"
    assert args.cmd == "t2m"


@pytest.mark.unit
def test_cli_accepts_engine_alias_for_backend_flag():
    from abstractmusic.cli import build_parser

    args = build_parser().parse_args(["--engine", "xl", "repl"])

    assert args.backend == "acestep-diffusers"
    assert args.cmd == "repl"


@pytest.mark.unit
def test_cli_accepts_official_engine_alias():
    from abstractmusic.cli import build_parser

    args = build_parser().parse_args(["repl", "--engine", "official", "--lm-backend", "mlx"])

    assert args.backend == "acestep-official"
    assert args.lm_model_path == "acestep-5Hz-lm-1.7B"
    assert args.lm_backend == "mlx"
    assert args.shift == 3.0
    assert args.infer_method == "ode"
    assert args.lm_temperature == 0.85
    assert args.audio_cover_strength == 1.0

    args = build_parser().parse_args(["repl", "--engine", "official", "--bpm", "128", "--keyscale", "F# major", "--timesignature", "4"])
    assert args.bpm == 128
    assert args.keyscale == "F# major"
    assert args.timesignature == "4"


@pytest.mark.unit
def test_cli_accepts_musicgen_and_stable_audio_engines():
    from abstractmusic.cli import build_parser

    args = build_parser().parse_args(["--engine", "musicgen-small", "t2m", "lo-fi music", "--duration", "5"])
    assert args.backend == "musicgen"
    assert args.cmd == "t2m"

    args = build_parser().parse_args(["repl", "--engine", "stable-audio-open-small", "--duration", "5"])
    assert args.backend == "stable-audio"
    assert args.cmd == "repl"


@pytest.mark.unit
def test_cli_accepts_verbose_flag_after_subcommand():
    from abstractmusic.cli import build_parser

    args = build_parser().parse_args(["repl", "--engine", "official", "--verbose"])

    assert args.backend == "acestep-official"
    assert args.verbose is True


@pytest.mark.unit
def test_music_repl_switches_engine_and_parameters_without_loading_backend(capsys):
    from abstractmusic.cli import MusicREPL, build_parser

    args = build_parser().parse_args(["repl", "--duration", "5"])
    repl = MusicREPL(args)

    repl.onecmd("/engine xl")
    repl.onecmd("/duration 12")
    repl.onecmd("/bpm 128")
    repl.onecmd("/keyscale F# major")
    repl.onecmd("/timesignature 4")
    repl.onecmd("/steps 16")
    repl.onecmd("/seed 42")
    repl.onecmd("/lm acestep-5Hz-lm-0.6B")
    repl.onecmd("/lm-backend mlx")
    repl.onecmd("/verbose on")
    repl.onecmd("/lyrics [Instrumental]")
    repl.onecmd("/params")

    assert repl.args.backend == "acestep-diffusers"
    assert repl.args.duration == 12.0
    assert repl.args.bpm == 128
    assert repl.args.keyscale == "F# major"
    assert repl.args.timesignature == "4"
    assert repl.args.steps == 16
    assert repl.args.seed == 42
    assert repl.args.lm_model_path == "acestep-5Hz-lm-0.6B"
    assert repl.args.lm_backend == "mlx"
    assert repl.args.verbose is True
    assert repl.args.lyrics == "[Instrumental]"
    assert repl._manager is None
    out = capsys.readouterr().out
    assert "engine: acestep-diffusers" in out
    assert "duration: 12" in out
    assert "verbose: on" in out
    assert "lyrics: [Instrumental]" in out


@pytest.mark.unit
def test_music_repl_bare_prompt_dispatches_generate(monkeypatch):
    from abstractmusic.cli import MusicREPL, build_parser

    args = build_parser().parse_args(["repl"])
    repl = MusicREPL(args)
    calls = []
    monkeypatch.setattr(repl, "_generate", lambda prompt: calls.append(prompt))

    repl.onecmd("bright melodic synth loop")

    assert calls == ["bright melodic synth loop"]


@pytest.mark.unit
def test_music_repl_prompt_state_runs_current_prompt(monkeypatch):
    from abstractmusic.cli import MusicREPL, build_parser

    args = build_parser().parse_args(["repl"])
    repl = MusicREPL(args)
    calls = []
    monkeypatch.setattr(repl, "_generate", lambda prompt=None: calls.append(prompt or repl.current_prompt))

    repl.onecmd("/prompt bright melodic synth loop")
    repl.onecmd("/run")

    assert repl.current_prompt == "bright melodic synth loop"
    assert calls == ["bright melodic synth loop"]


@pytest.mark.unit
def test_music_repl_normalizes_hyphenated_slash_commands(monkeypatch):
    from abstractmusic.cli import MusicREPL, build_parser

    args = build_parser().parse_args(["repl"])
    repl = MusicREPL(args)
    calls = []
    monkeypatch.setattr(repl, "_generate", lambda prompt=None: calls.append(prompt))

    repl.onecmd("/guidance-scale 4")
    repl.onecmd("/shift 3")
    repl.onecmd("/infer-method sde")
    repl.onecmd("/sampler-mode heun")
    repl.onecmd("/lm-temperature 0.95")
    repl.onecmd("/lm-cfg-scale 2.5")
    repl.onecmd("/audio-cover-strength 0.35")
    repl.onecmd("/cover-noise-strength 0")
    repl.onecmd("/out-dir /tmp/abstractmusic-repl")

    assert repl.args.guidance_scale == 4.0
    assert repl.args.shift == 3.0
    assert repl.args.infer_method == "sde"
    assert repl.args.sampler_mode == "heun"
    assert repl.args.lm_temperature == 0.95
    assert repl.args.lm_cfg_scale == 2.5
    assert repl.args.audio_cover_strength == 0.35
    assert repl.args.cover_noise_strength == 0.0
    assert str(repl.out_dir) == "/tmp/abstractmusic-repl"
    assert calls == []


@pytest.mark.unit
def test_music_repl_request_parameters_do_not_reload_manager():
    from abstractmusic.cli import MusicREPL, build_parser

    args = build_parser().parse_args(["repl"])
    repl = MusicREPL(args)
    repl._manager = object()
    repl._manager_dirty = False

    repl.onecmd("/duration 30")
    repl.onecmd("/bpm 128")
    repl.onecmd("/steps 8")
    repl.onecmd("/seed 123")
    repl.onecmd("/guidance 1")
    repl.onecmd("/shift 3")
    repl.onecmd("/infer-method ode")
    repl.onecmd("/lm-temperature 0.85")
    repl.onecmd("/audio-cover-strength 0.35")

    assert repl._manager_dirty is False
    repl.onecmd("/engine xl")
    assert repl._manager_dirty is True
