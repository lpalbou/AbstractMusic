# Proposed: The documented Stable Audio CLI command fails before loading anything (task name mismatch)

## Metadata
- Created: 2026-11-10
- Status: Proposed
- Completed: N/A

## ADR status
- Governing ADRs: `docs/adr/0001_music_provider_abstraction_and_dependency_policy.md`
- ADR impact: None

## Context
Found during the 0.14.0 install-profile proof on Apple silicon (abstractmusic 0.1.17, also the repository
HEAD `49c049d`), with the checkpoint `stabilityai/stable-audio-open-small` cached:

```
$ abstractmusic --backend stable-audio t2m "a short calm piano loop" --out music_cli_stableaudio.wav --duration 8 --device auto --verbose
ERROR: Backend does not support task 'text_to_music'.
(exit 1, 0.09 s)
```

This is the command README.md and docs/getting-started.md give for Stable Audio. Called directly
(`MusicManager(backend=StableAudioBackend(...))` through the backend's own task), the same checkpoint
generates on MPS.

## Current code reality
`MusicManager.generate_audio` (used by the CLI and by `t2m`) always asks for the task `text_to_music`;
`StableAudioBackend` declares `text_to_audio` only, and no other `MusicManager` method reaches it. The CLI
accepts `--backend stable-audio` and then refuses it at the task check.

## Problem
A documented command fails at once; the user cannot reach Stable Audio from the CLI at all.

## What we want to do
Make `--backend stable-audio t2m` generate: either map `text_to_music` onto `text_to_audio` for backends that
declare only the latter (with the mapping reported in the result metadata), or have the Stable Audio backend
declare both tasks. Add a CLI-level test that runs every backend named in the docs through `t2m` against a
stub and fails on an unsupported-task error.

## Why
The CLI and the docs are the first contact; a backend that is listed must run from them.

## Requirements
- `abstractmusic --backend stable-audio t2m ...` produces a file.
- The README/getting-started command is executed by a test (stubbed backend) so the mismatch cannot return.

## Suggested implementation
In `music_manager.py`, resolve the requested task against the backend's declared tasks with one explicit
alias table (`text_to_music` -> `text_to_audio` for audio-only backends); record `task_resolved` in metadata.

## Scope
`music_manager.py`, the Stable Audio backend's task declaration, one CLI test, docs unchanged.

## Non-goals
Changing the Stable Audio runtime path (see 0092 for its dtype and duration reporting).

## Dependencies and related tasks
- 0092 (CLI shows no device/dtype; Stable Audio float32 and duration drift)
- 0087 (truthful provider availability)

## Expected outcomes
The documented command works; a test keeps it working.
