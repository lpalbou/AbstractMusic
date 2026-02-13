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

