# Architecture

AbstractMusic is organized around a small public contract:

- `MusicManager`: user-facing facade that builds requests and delegates to a backend.
- `MusicBackend`: provider interface for local or remote-compatible generation engines.
- `AudioGenerationRequest`: normalized request object for text-to-music/audio.
- `GeneratedAsset`: binary media result with MIME type and metadata.
- Model capability registry: packaged metadata describing known providers and model constraints.

## Provider Boundary

Provider-specific logic belongs in backends. The public API should not become ACE-Step-specific,
HeartMuLa-specific, or Diffusers-specific. Common concepts such as prompt, lyrics, duration,
format, seed, sample rate, and vocal language can be first-class request fields. Specialized
options should be explicit and documented.

## Text Planning Boundary

Text planning is a separate layer from backend generation. `MusicPlanningRequest` captures the
raw prompt, lyrics, duration, metadata hints, and model/backend context. A planner returns a
`MusicPromptPlan`; `compile_music_prompt_plan(...)` renders that plan into the deterministic
prompt/lyrics/metadata contract required by the selected backend.

The built-in planner is dependency-free and low-confidence by design. AbstractMusic must not import
AbstractCore or any LLM runtime for planning. Hosts can inject a planner through `MusicManager` or
through AbstractCore plugin config (`music_text_planner`, `music_text_planner_instance`, or
`music_text_planner_factory`). This keeps the intelligence layer swappable while preserving a stable
backend request contract.

When hosted by AbstractCore, the plugin may also use a narrow host text-generation service if the
host supplies one structurally through context/config. AbstractMusic only calls `generate_text(...)`
or `generate_structured(...)`, records planner provenance, and never receives raw AbstractCore
provider/facade objects.

The plugin surface is bidirectional: AbstractCore can also discover AbstractMusic providers,
models, and operations through lightweight `available_providers(...)`, `list_models(...)`,
`list_operations(...)`, and `capability_catalog(...)` methods. These discovery methods read the
packaged model metadata and must not load generation backends.

## Dependency Boundary

Heavy runtime stacks must be imported lazily. The base package stays focused on contracts, manager
code, artifact helpers, docs, CLI/plugin shells, provider metadata, and stdlib-only remote clients.
The base default is `acemusic`, which calls a configured hosted ACE Music-compatible API and does
not install local ML libraries. Local model engines live behind explicit extras such as `acestep`,
`acestep-v15`, `acestep-diffusers`, `diffusers`, `apple`, `gpu`, `all-apple`, and `all-gpu`.
The `acestep` extra installs the supported local ACE-Step Diffusers provider; `acestep-v15` is the
explicit quality-limited vendored v1.5 backend.

## Precision Policy

When official 8-bit model artifacts exist, prefer them for local providers. If no official 8-bit
artifact exists, prefer the smallest official 16-bit path before larger precision formats. Do not
invent unofficial quantized checkpoints as defaults.

On Apple hardware, prefer PyTorch MPS for standalone providers when supported. For local ACE-Step,
the `acestep` backend has an explicit CPU float32 fallback when MPS returns non-finite audio.
