# Proposed: Local engine priority after two remote backends

## Metadata
- Created: 2026-05-21
- Status: Proposed
- Completed: N/A

## ADR status
- Governing ADRs: `docs/adr/0001_music_provider_abstraction_and_dependency_policy.md`
- ADR impact: None

## Context

AbstractMusic now has two lightweight remote endpoints in the base package: ACE Music and
ElevenLabs Music. That is enough remote provider coverage for the default profile. The strategic
gap is local/open-weight generation quality, runtime stability, and prompt adherence.

## Current code reality

- `acemusic` is the default stdlib-only remote backend.
- `elevenlabs` is a second stdlib-only remote backend scoped to official Music endpoints and
  structured composition plans.
- `acestep` / `acestep-diffusers` is the current recommended local path.
- `acestep-v15` remains quality-limited after repeated-loop failures.
- `musicgen` and `stable-audio` are implemented/available as optional comparison or short-clip
  paths, but licensing and quality limit their role.
- Planned local candidates already include HeartMuLa and YuE, both heavier than the remote clients.

## Problem or opportunity

Additional remote APIs would increase maintenance surface without solving the main AbstractMusic
problem: high-quality, controllable, locally served music generation that fits the same manager,
planner, quality-gate, and AbstractCore plugin contracts.

## Proposed direction

Treat remote provider expansion as capped for now. Prioritize local work in this order:

1. harden the `acestep-diffusers` route and quality gates for long-form structure;
2. run a Stable Audio 3 / newer open-weight Stability spike only after confirming license,
   checkpoint layout, and Apple/GPU runtime feasibility;
3. evaluate HeartMuLa as the next serious lyrics/tags local model if its codec/runtime can be
   integrated without upstream source-tree dependency;
4. keep YuE as a multi-stage research track until Stage-2 and codec output can generate real audio.

2026-05-21 update: the concrete next-engine recommendation was promoted to
`docs/backlog/completed/0083_stable_audio_3_local_provider_spike.md`. Stable Audio 3.0 Small should
be evaluated before adding more remote providers or spending more time on the older Stable Audio
Open Small short-clip path.

## Why it might matter

- focuses effort where AbstractMusic can become self-sufficient;
- keeps the base package lightweight and remote-capable without becoming a provider zoo;
- uses ElevenLabs composition-plan support to validate our structured planning abstraction before
  applying the same abstraction to local models.

## Promotion criteria

- a concrete local candidate has current official weights, permissive or acceptable licensing,
  and a feasible package-owned runtime path;
- the implementation can pass harmonic-diversity and repetition quality gates with real generated
  audio;
- dependency additions stay behind `apple`, `gpu`, or explicit provider extras.

## Validation ideas

- real 30s and 120s generation smokes on Apple and one GPU path;
- quality-gate comparison against the accepted ACE-Step reference and bad repeated-tone examples;
- import-light base install check;
- AbstractCore plugin discovery without model runtime imports.

## Non-goals

- do not add more default remote endpoints unless they unlock a new abstraction that local engines
  can also use;
- do not use external source checkouts as runtime implementations;
- do not mark a local candidate recommended before real audio validation.

## Guidance for future agents

Before implementing another provider, ask whether it advances locally served music generation or
only adds another API wrapper. Prefer work that improves the shared planning, sectioning,
quality-gate, residency, and runtime-selection contracts across local engines.
