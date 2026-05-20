# Completed: Dependency Profiles And Optional Providers

## Metadata
- Created: 2026-05-15
- Status: Completed
- Completed: 2026-05-21
- Priority: P2

## Context

The user requires minimal dependencies, even if AbstractMusic has to reimplement glue internally.
The sibling projects favor lightweight base installs with explicit heavy local extras. Music now
matches that pattern: the base package is remote-capable and local runtimes are explicit extras.

## Current code reality

- `pyproject.toml` base dependencies are empty.
- `acemusic` is the default lightweight remote backend and uses only the standard library client.
- `remote` is a no-op extra for framework profile symmetry.
- `all-apple` and `all-gpu` now install the supported local backend dependency families for their
  platform profile.
- AbstractVoice and AbstractVision keep heavy local stacks behind extras and lazy imports.
- TinyMozart is a small unconditional MIDI piano model but has no declared license in HF metadata.
- Omni2Sound is CC BY-NC 4.0, very large, and requires a heavy pinned CUDA/script stack plus
  extra CLIP/T5 dependencies.

## Problem Addressed

The current base install pulls local model runtimes even when a host only wants contracts,
AbstractCore plugin discovery, docs, or a future remote provider. Optional research providers
could make this much worse if added naively.

## What was done

Split dependency profiles, added a lightweight remote default backend, and defined optional
provider boundaries without removing existing local generation routes.

## Why

Minimal dependencies make the package installable in server, CI, and orchestration contexts. Heavy
music generation remains available, but only when explicitly requested.

## Requirements Status

- Move heavy local stacks out of base when compatible with release strategy: done.
- Suggested extras:
  - `acestep`: ACE-Step custom/local backend dependencies.
  - `acestep-diffusers`: Diffusers ACE-Step provider dependencies.
  - `diffusers`: generic Diffusers audio pipeline dependencies.
  - `local`: all local music providers considered supported.
  - `midi`: TinyMozart/MIDI-related provider dependencies only if license is resolved.
  - `omni2sound`: only if kept as a non-commercial external provider, never default.
  - `apple`, `gpu`, `all-apple`, `all-gpu`: platform profile aliases.
  - `remote`: no-op base/remote profile alias.
  - `dev` and `test`: contributor/test tooling.
- Keep `abstractmusic` base focused on contracts, manager, artifact helpers, CLI/plugin shell,
  remote-compatible clients, and docs: done.
- Preserve lazy imports and helpful optional dependency errors: retained.
- Add packaging tests proving `import abstractmusic` does not import Torch/Diffusers in a base-only
  environment: done.
- Explicitly document provider license and commercial-use status: done in the registry and docs.

## Scope

- Package extras and dependency declarations.
- Optional dependency error messages.
- Docs and acknowledgments updates.
- Tests for import/dependency boundaries.
- Provider license policy for TinyMozart and Omni2Sound.

## Non-goals

- Do not implement TinyMozart or Omni2Sound inference in this task unless split into separate
  provider tasks.
- Do not make non-commercial providers available through default model selection.
- Do not add system-level dependencies like FluidSynth or CUDA-pinned stacks to base.

## Dependencies and related tasks

- `docs/adr/0001_music_provider_abstraction_and_dependency_policy.md`
- `docs/backlog/deprecated/0081_music_install_profile_boundary.md`
- `docs/backlog/planned/020_music_abstraction_and_capability_registry.md`

## Expected outcomes

- The default install is small and framework-friendly.
- Local generation remains explicit and documented.
- Optional providers have clear dependency and license boundaries.

## Validation

- Base-only install/import test in a clean environment.
- `python -m pytest -q -m "not integration"`
- Packaging build and `twine check`.
- Manual check that provider imports fail with actionable messages when extras are missing.

## Progress checklist

- [x] Promote/supersede the proposed install-profile item.
- [x] Split extras in `pyproject.toml`.
- [x] Update optional dependency messages.
- [x] Add packaging/import tests.
- [x] Update docs and acknowledgments.

## Guidance for the implementing agent

Do not optimize for one convenient local developer environment. Optimize for clear install intent:
base contracts are light, local model engines are explicit.

## Progress notes

2026-05-15: `pyproject.toml` now keeps base dependencies empty and moves local runtime stacks
behind explicit extras (`acestep`, `acestep-diffusers`, `diffusers`, `local`, `apple`, `gpu`,
`all-apple`, `all-gpu`, plus placeholder optional research extras). Remaining work is to verify a
clean base-only install and tighten provider-specific error messages/docs.

2026-05-21: Added the stdlib-only `acemusic` remote backend, changed the CLI and AbstractCore
plugin default to remote generation, added the `remote` no-op extra, expanded `all-apple` and
`all-gpu` with supported local backend dependency families, and updated docs/tests to treat local
generation as opt-in. `stable-audio-tools` remains intentionally documented as a separate
`--no-deps` install because the upstream package dependency graph is not suitable for a framework
aggregate extra.
