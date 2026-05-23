# Proposed: Truthful Music Provider Runtime Availability

## Metadata
- Created: 2026-05-22
- Status: Proposed
- Completed: N/A
- Priority: P1

## Context

AbstractFlow receives music provider options through AbstractGateway:

- `GET /api/gateway/audio/music/providers`
- `GET /api/gateway/audio/music/models`

The live response reports `source = "abstractmusic.local"` and advertises providers including
`ace-music`, `ace-step`, `elevenlabs`, `heartmula`, `m-a-p`, `meta`, and `stability-ai`.

For the current local stack, user-visible runnable providers are expected to be limited to ACE Music,
ACE-Step, and Stable Audio/Stability AI. Flow does not hardcode this provider list; it displays what
Gateway/Runtime relay from AbstractMusic.

## Current code reality

- `src/abstractmusic/integrations/abstractcore_plugin.py` builds provider availability from the
  packaged music model registry without loading model runtimes.
- Tests currently assert generic discovery includes future/optional providers such as ElevenLabs,
  HeartMuLa, and ACE-Step.
- Planned provider work already exists for HeartMuLa and YuE, which indicates those should not be
  surfaced as fully runnable local providers before validation.

## Problem

The catalog conflates "known model family" with "usable in this deployment". This causes thin clients
such as AbstractFlow to show providers that are not actually runnable for the connected Gateway.

Observed problematic outcomes:

- HeartMuLa and YuE-style providers can appear as `status = "available"` even though they are planned
  or research-only.
- MusicGen/Meta can appear installed/available despite not being part of the current validated usable
  set.
- Clients cannot reliably distinguish runnable, configured, installed, planned, gated, experimental,
  and remote-unconfigured providers from the current provider-level summary.

## What we want to do

Make AbstractMusic provider discovery truthful enough for Gateway clients:

- provider-level `status` should reflect runtime usability, not just registry presence;
- provider records should expose an explicit `usable` or equivalent boolean;
- planned/research/experimental providers should not be returned as available runnable providers;
- models may still be discoverable as catalog entries, but their usability state must be explicit.

## Requirements

- Do not remove catalog knowledge for future providers.
- Do not make AbstractFlow hardcode music provider allowlists.
- Keep remote providers separate from local installed providers.
- Preserve enough metadata for advanced users to understand why a model/provider is hidden or disabled.

## Suggested implementation

- Derive provider availability from the best model-level status for that provider.
- Treat statuses such as `planned-research`, unimplemented, missing dependency, remote-unconfigured,
  and failed validation as not runnable.
- Keep `available_providers` limited to runnable providers.
- Add `known_providers` or `provider_details` entries for non-runnable providers with reasons.
- Add tests for the current intended usable set: ACE Music, ACE-Step, and Stable Audio/Stability AI.

## Scope

Includes AbstractMusic discovery/catalog truth only. Gateway and Runtime should be able to relay the
improved metadata without Flow-specific allowlists.

## Non-goals

- Do not modify AbstractFlow to hardcode provider names.
- Do not implement HeartMuLa, YuE, MusicGen, or ElevenLabs runtime fixes in this task.
- Do not change generation routing semantics except where they depend on provider availability truth.

## Validation

- Unit tests for `available_providers(task="text_to_music")` and model catalog status.
- Gateway smoke check through `/api/gateway/audio/music/providers`.
- Flow smoke check that the provider dropdown no longer shows non-runnable providers as selectable.

## Guidance for the implementing agent

Favor truthful, structured metadata over hiding details. Thin clients need one reliable field for
"can the connected deployment run this now?" and can show advanced non-runnable catalog entries later.
