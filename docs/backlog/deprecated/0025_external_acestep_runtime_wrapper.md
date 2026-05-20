# Deprecated: External ACE-Step Runtime Wrapper

## Metadata
- Created: 2026-05-15
- Status: Deprecated
- Completed: 2026-05-15
- Deprecated: 2026-05-20
- Priority: P0

## Context

An earlier experiment wrapped a local ACE-Step runtime to compare AbstractMusic's output against a
known-good generation path. That experiment proved the public abstraction could move bytes through
a working model stack, but it did not satisfy the package goal: AbstractMusic must generate ACE-Step
music using its own backend code and model abstractions, not a source checkout or runtime package
outside AbstractMusic.

## Deprecation Report

The wrapper path is removed from runtime code, package extras, CLI aliases, plugin registration,
and current user-facing docs. The accepted reference WAVs remain comparison artifacts only; they
must not be treated as evidence that the standalone package backend works.

Replacement work is tracked by:

- `docs/backlog/planned/035_acestep_v15_backend_compatibility_hardening.md`
- `docs/backlog/planned/065_acestep_repetition_quality_gate.md`
- `docs/backlog/planned/030_acestep_diffusers_xl_provider.md`

## Validation Expectations

- CLI/backend alias audit must show no selectable out-of-package ACE-Step runtime backend.
- Package metadata must not expose an optional extra for the removed wrapper.
- AbstractCore plugin registration must only advertise package-owned or Diffusers-backed local
  providers.
- Real generation candidates for the default ACE-Step path must be produced by
  `src/abstractmusic/backends/acestep_v15.py` or another package-owned backend.
