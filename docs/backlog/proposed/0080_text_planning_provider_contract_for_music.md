# Proposed: Text Planning Provider Contract For Music

## Metadata
- Created: 2026-05-20
- Status: Proposed
- Completed: N/A

## ADR status
- Governing ADRs: None
- ADR impact: Needs new ADR if promoted, because this would define a cross-surface text-generation contract and dependency boundary.

## Context
AbstractMusic now has a dependency-free local prompt planner that can expand short prompts and
produce simple structured lyrics before generation. That was useful as a smoke-test and fallback
surface, but it is not a serious advanced music planning strategy. Stronger music generation often
needs text generation before audio generation: prompt expansion, lyric drafting, language/style
normalization, section planning, metadata inference, and model-specific prompt formatting.

AbstractMusic is also an AbstractCore plugin, but it must not hard-depend on AbstractCore. The
package needs a way to optionally use text-generation capability when available while keeping the
base package import-light and usable as a standalone local music library.

## Current code reality
- `src/abstractmusic/prompt_planner.py` contains dependency-free heuristics for
  `create_prompt_plan(...)`, `enhance_caption(...)`, and `generate_lyrics(...)`. The lyric function
  is template-based and should be treated as a fallback, not a quality target.
- `src/abstractmusic/cli.py` exposes `--enhance-prompt`, `--auto-lyrics`, `--instrumental`, and
  `--print-plan`, then passes the resolved prompt/lyrics/BPM/key/time hints through the existing
  `MusicManager` request path.
- `src/abstractmusic/music_manager.py` and `src/abstractmusic/types.py` have first-class lyrics and
  backend-specific extra fields, but no text-planner provider contract.
- `src/abstractmusic/integrations/abstractcore_plugin.py` registers music backends as an
  AbstractCore capability plugin without importing AbstractCore directly. It reads owner/config
  values and supports backend injection, but has no text-planning injection point.
- `../abstractvoice/abstractvoice/examples/llm_provider.py` demonstrates a tiny
  OpenAI-compatible local LLM client for Ollama/LM Studio. AbstractVoice docs describe it as an
  example/demo surface; production agent/server orchestration should remain with AbstractCore.
- `../abstractvision/src/abstractvision/backends/base_backend.py` exposes explicit provider catalog
  hooks in the backend contract, and AbstractVision has stable-diffusion.cpp component fields for
  model-family LLM components. It does not establish a generic text-planning contract that
  AbstractMusic can simply reuse.
- ACE-Step's own 5 Hz LM is model-specific. It can generate music metadata and audio-code tokens,
  but it is not a general-purpose lyric/caption writing service and should not be confused with an
  AbstractMusic-wide text planner.

## Problem or opportunity
The current hardcoded lyric generation is a brittle placeholder. Keeping it as the default
advanced path would create low-quality outputs and bad abstractions. At the same time, simply
adding an AbstractCore dependency would violate the plugin boundary and make the base package less
portable.

The opportunity is to define a small, package-owned text-planning contract that can be backed by:

- a deterministic fallback planner for offline/no-LLM operation;
- a lightweight OpenAI-compatible local provider such as Ollama or LM Studio;
- an injected callable or adapter supplied by AbstractCore when AbstractMusic is loaded as a
  plugin;
- model-specific planners such as ACE-Step 5 Hz LM, but only behind explicit backend semantics.

## Proposed direction
Investigate a `MusicTextPlanner` or `MusicPlanningProvider` contract owned by AbstractMusic. It
should accept a structured request containing prompt, optional lyrics, instrumental flag, duration,
language, model/backend hints, and desired outputs. It should return a structured
`MusicPromptPlan`-like object with:

- effective caption;
- optional lyrics;
- instrumental intent;
- vocal language;
- BPM, key/scale, and time signature hints;
- provenance such as `planner_backend`, `generated_fields`, confidence, warnings, and raw text
  when useful.

Evaluate three integration modes before implementation:

1. Deterministic fallback: keep the current dependency-free planner, but make it visibly
   low-confidence and avoid presenting template lyrics as "good" generated lyrics.
2. Lightweight local LLM adapter: add an optional OpenAI-compatible text planner using only
   stdlib HTTP or a small optional extra, with presets for Ollama and LM Studio. This should be
   explicit and disabled by default.
3. AbstractCore injection: let the plugin accept an owner/config callable or planner object, such
   as `music_text_planner`, without importing AbstractCore. AbstractCore can later provide a real
   adapter that uses its configured text model.

The likely long-term shape is fallback plus injection first, then optional OpenAI-compatible local
provider if the contract proves stable.

## Why it might matter
Advanced music generation is not only audio synthesis. Better captions, lyrics, and metadata can
materially improve model quality, especially for ACE-Step and future lyrics/tags-conditioned
providers. A clean planner contract also prevents every backend from inventing its own prompt
rewriter and gives AbstractCore a safe place to plug in text-generation intelligence without
making AbstractMusic depend on it.

## Promotion criteria
Promote to planned when at least one of these is true:

- listening tests show that rich LLM-generated captions/lyrics reliably improve ACE-Step or
  another backend over deterministic fallback prompts;
- AbstractCore exposes or agrees on a stable injection surface suitable for plugin-owned media
  planners;
- a local OpenAI-compatible planner can be implemented without adding dependencies to the base
  package and without surprise network calls;
- a new provider requires text planning for acceptable output quality.

## Validation ideas
- Compare accepted prompts with and without planner output using existing harmonic-diversity and
  spectrotemporal gates plus human listening checks.
- Unit-test planner request/response serialization without importing heavy ML or AbstractCore.
- Add fake provider tests for deterministic fallback, injected callable, provider timeout, invalid
  JSON/text output, and opt-in remote/local HTTP behavior.
- Verify CLI, REPL, library, and AbstractCore plugin surfaces report planner provenance and never
  silently ignore lyrics or planner errors.
- Confirm `pip install abstractmusic` remains dependency-free and import-light.

## Non-goals
- Do not add a hard dependency on AbstractCore.
- Do not make remote HTTP calls by default.
- Do not require Ollama, LM Studio, OpenAI, or any provider-specific SDK for the base package.
- Do not treat ACE-Step's 5 Hz LM as a generic text LLM.
- Do not reintroduce any external ACE-Step source/runtime path.
- Do not claim the current template lyric fallback is high-quality lyric generation.

## Guidance for future agents
Start by preserving the current fallback behavior but rename/report it honestly as deterministic
fallback planning. Design the contract before wiring any provider. Prefer dependency injection and
stdlib HTTP over new dependencies. If the contract changes public behavior across CLI, REPL,
library mode, and AbstractCore plugin mode, create an ADR before implementation.
