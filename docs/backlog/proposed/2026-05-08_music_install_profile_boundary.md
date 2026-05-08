# Proposed: Music Install Profile Boundary

## Metadata
- Created: 2026-05-08
- Status: Proposed
- Completed: N/A

## Context

AbstractMusic is a capability package that can sit behind AbstractCore and Gateway. It currently
focuses on local in-process music generation, especially ACE-Step, which is consistent with its
recent implementation history but not with the emerging remote-light default profile used for
Vision and Voice.

## Current Code Reality

- Base dependencies currently include local model-generation stacks such as NumPy, Diffusers,
  Torch, Transformers, Accelerate, Safetensors, Hugging Face Hub, and Einops.
- ACE-Step integration is local-first and model-heavy.
- Music can be discovered as an AbstractCore capability plugin.
- Music currently has no production HTTP server boundary and no OpenAI-compatible remote music
  backend contract equivalent to Vision/Voice.
- `abstractmusic[apple]`, `abstractmusic[gpu]`, `abstractmusic[all-apple]`, and
  `abstractmusic[all-gpu]` are currently no-op compatibility aliases because the base package
  already includes the local ACE-Step runtime stack.

## Problem

If Music becomes part of the default Gateway/Core capability story, its current base install
violates the remote-light default rule:

- installing Music pulls Torch/Diffusers local model stacks;
- default Gateway Docker would become much heavier if it included Music;
- there is no remote OpenAI/OpenAI-compatible music backend contract equivalent to Vision/Voice
  yet.

Because Music is currently local/in-process, there is no Music-level auth/CORS cascade to design.
If a future Music server or remote provider client is added, keep inbound server auth/origins
separate from outbound provider credentials and model configuration.

## Proposed Direction

Keep Music out of default Gateway remote-light profiles until it has a lightweight remote-capable
base.

Future package split:

- `abstractmusic`: light capability contracts, CLI glue, plugin entry point, and remote-compatible
  backend abstractions only.
- `abstractmusic[acestep]`: ACE-Step local backend and dependencies.
- `abstractmusic[local]`: all local music backends.
- `abstractmusic[server]` or `abstractmusic[remote]`: remote API music generation backends if a
  stable provider contract exists.
- `abstractmusic[apple]` / `abstractmusic[gpu]`: native profile aliases; currently no-op because
  the base package is local-first.
- `abstractmusic[all-apple]` / `abstractmusic[all-gpu]`: currently no-op compatibility aliases;
  after a local-extra split, they should install the platform-relevant Music-owned local stack.

Gateway should expose music readiness only when Music is installed and configured. Do not include
Music in `abstractgateway[server]` or lightweight Docker until the base is remote-light. It is
acceptable to include Music in native full Gateway profiles because those profiles intentionally
install local media engines on the host.

## Pending Changes Guidance

Do not keep generated cache/artifact changes as part of this strategy.

The current unified profile pass may keep the no-op compatibility aliases so Core/Gateway/root
aggregate profiles can cascade consistently. Do not add Music to lightweight server/Docker profiles
until a remote-light base or remote backend exists.

## Promotion Criteria

Promote when Gateway wants `/v1/audio/music` or workflow music generation in a default server
profile, or when a remote music backend contract is selected.

## Validation Ideas

- Packaging tests proving `abstractmusic` base excludes Torch/Diffusers after the split.
- Local ACE-Step tests gated behind `abstractmusic[acestep]`.
- Gateway capability discovery test for missing/unconfigured Music.
- Remote music backend contract tests once a provider exists.
