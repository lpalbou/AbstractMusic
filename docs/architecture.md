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

## Dependency Boundary

Heavy runtime stacks must be imported lazily. The base package stays focused on contracts, manager
code, artifact helpers, docs, CLI/plugin shells, and provider metadata. Local model engines live
behind explicit extras such as `acestep-official`, `acestep`, `acestep-diffusers`, `diffusers`,
`apple`, and `gpu`.

## Precision Policy

When official 8-bit model artifacts exist, prefer them for local providers. If no official 8-bit
artifact exists, prefer the smallest official 16-bit path before larger precision formats. Do not
invent unofficial quantized checkpoints as defaults.

On Apple hardware, prefer official MLX when a provider has a supported MLX path. The
`acestep-official` backend follows this rule for the ACE-Step 5Hz LM. Otherwise use PyTorch MPS.
CPU fallback must be explicit in warnings or metadata.
