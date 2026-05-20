# Superseded: Music Install Profile Boundary

## Metadata
- Created: 2026-05-08
- Status: Superseded
- Completed: 2026-05-21
- Superseded by: `../completed/050_dependency_profiles_and_optional_providers.md`

## Context

AbstractMusic is a capability package that can sit behind AbstractCore and Gateway. This proposal
captured the need to move from a local-first base toward the remote-light default profile used for
Vision and Voice.

## Current Code Reality

- Base dependencies are now empty.
- `acemusic` is the default lightweight remote backend and uses only the Python standard library
  client code.
- Local ACE-Step integration is explicit behind optional extras.
- Music can be discovered as an AbstractCore capability plugin.
- `abstractmusic[remote]` is a no-op alias for framework profile symmetry.
- `abstractmusic[apple]`, `abstractmusic[gpu]`, `abstractmusic[all-apple]`, and
  `abstractmusic[all-gpu]` are concrete local runtime profiles.

## Original Problem

If Music became part of the default Gateway/Core capability story, its former base install violated
the remote-light default rule:

- installing Music pulls Torch/Diffusers local model stacks;
- default Gateway Docker would become much heavier if it included Music;
- there was no remote OpenAI/OpenAI-compatible music backend contract equivalent to Vision/Voice.

Because Music was local/in-process, there was no Music-level auth/CORS cascade to design. Any
future Music server should keep inbound server auth/origins separate from outbound provider
credentials and model configuration.

## Implemented Direction

Music can be included in default remote-light profiles once a host supplies remote credentials.
Local model generation remains explicit.

Future package split:

- `abstractmusic`: light capability contracts, CLI glue, plugin entry point, model metadata, and
  stdlib-only remote backend code.
- `abstractmusic[acestep]`: ACE-Step local backend and dependencies.
- `abstractmusic[local]`: all local music backends.
- `abstractmusic[remote]`: no-op alias because the base install already contains remote clients.
- `abstractmusic[apple]` / `abstractmusic[gpu]`: native local runtime profiles.
- `abstractmusic[all-apple]` / `abstractmusic[all-gpu]`: platform-relevant local runtime stacks.

Gateway should expose music readiness only when Music is installed and configured. Base Music is now
remote-light; native full Gateway profiles can still install local media engines on the host.

## Closure Notes

Completed through `completed/050_dependency_profiles_and_optional_providers.md`. Generated
cache/artifact changes remain out of scope.

## Validation

- Packaging tests prove `abstractmusic` base excludes Torch/Diffusers.
- Local ACE-Step tests remain gated behind optional extras and explicit backend selection.
- Remote ACE Music backend tests cover stdlib request shaping and response decoding without storing
  API keys.
