# ADR 0001: Music Provider Abstraction And Dependency Policy

## Status

Accepted

## Context

AbstractMusic must expose a unified way to generate music while supporting multiple local and
remote-capable model families. Music generation models are heavy, hardware-sensitive, and unevenly
licensed. A default install that eagerly pulls every local runtime would conflict with the
AbstractCore and AbstractVision direction.

## Decision

AbstractMusic will keep a stable public request/result abstraction above provider-specific
backends. Provider-specific features such as lyrics, vocal language, duration parameter names,
guidance behavior, reference audio, or video input must be expressed through explicit capability
metadata and first-class request fields where they are broadly useful.

Heavy local inference stacks must be lazy-imported and should live behind optional extras once the
packaging split is implemented. Default providers should prefer permissive, commercially usable
licenses. Non-commercial or license-unclear providers may exist only as explicit optional providers
with clear warnings and tests that prove they cannot be selected silently as defaults.

## Consequences

- The manager stays thin and model-agnostic.
- Backends must expose enough capability metadata for callers and AbstractCore to avoid accidental
  unsupported calls.
- ACE-Step can be the primary local provider, but the public API cannot become ACE-Step-shaped.
- TinyMozart and Omni2Sound are not default providers under this policy.

## Enforcement

- Backlog items that add providers must update capability metadata and docs.
- Provider tests must cover unsupported parameter handling instead of silently ignoring inputs.
- Packaging tests should verify dependency profiles after the local-extra split.

