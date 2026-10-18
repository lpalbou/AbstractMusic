from __future__ import annotations

import io
import sys

from abstractmusic import cli


def test_help_on_a_cp1252_stream_never_raises(monkeypatch) -> None:
    """Windows pipes and redirects use the ANSI code page (often cp1252), which has no arrows."""
    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="cp1252", errors="strict"))
    try:
        cli.main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    sys.stdout.flush()
    assert b"usage" in raw.getvalue().lower()
