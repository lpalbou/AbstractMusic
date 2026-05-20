# AbstractMusic

`abstractmusic` is a model-agnostic **text-to-music / text-to-audio** library designed to plug into **AbstractCore** as an optional capability plugin. The base install is lightweight and remote-capable; local model runtimes live behind explicit extras.

## Install

```bash
pip install abstractmusic
```

The base package is import-light: contracts, manager, CLI shell, plugin wiring, docs, model
metadata, and a stdlib-only ACE Music remote backend. It does not install Torch, Diffusers,
Transformers, or NumPy.

Use the base install with a remote API key:

```bash
export ACEMUSIC_API_KEY=...
abstractmusic t2m "ambient lo-fi study music" --out out.wav --duration 30
```

Install a local runtime profile when you want in-process model generation:

```bash
pip install "abstractmusic[remote]"  # no-op alias; base install already contains remote clients
pip install "abstractmusic[acestep]"  # local ACE-Step Diffusers path
pip install "abstractmusic[acestep-v15]"  # explicit quality-limited ACE-Step v1.5 path
pip install "abstractmusic[acestep-diffusers]"
pip install "abstractmusic[apple]"
pip install "abstractmusic[gpu]"
pip install "abstractmusic[all-apple]"  # all supported Apple/MPS local runtime deps
pip install "abstractmusic[all-gpu]"  # all supported CUDA/ROCm-style local runtime deps
```

The `acestep` profile installs the package-owned ACE-Step route. On Apple MPS, AbstractMusic
prefers MPS bfloat16 when the local PyTorch stack supports it, then MPS float32, and only falls
back to CPU float32 if MPS still returns invalid audio.

## Quickstart (remote generation)

```python
from abstractmusic import MusicManager
from abstractmusic.backends import AceMusicBackend, AceMusicBackendConfig

backend = AceMusicBackend(config=AceMusicBackendConfig(api_key="..."))

mm = MusicManager(backend=backend)
wav_bytes = mm.t2m("uplifting synthwave with punchy drums", duration_s=30.0)
open("out.wav", "wb").write(wav_bytes)
```

## Quickstart (local generation)

```python
from abstractmusic import MusicManager
from abstractmusic.backends import AceStepDiffusersBackend, AceStepDiffusersBackendConfig

backend = AceStepDiffusersBackend(config=AceStepDiffusersBackendConfig())

mm = MusicManager(backend=backend)
wav_bytes = mm.t2m("uplifting synthwave with punchy drums", duration_s=10.0)
open("out.wav", "wb").write(wav_bytes)
```

The explicit quality-limited ACE-Step v1.5 backend can also be selected through the same public abstraction:

```python
from abstractmusic import MusicManager
from abstractmusic.backends import AceStepV15Backend, AceStepV15BackendConfig

backend = AceStepV15Backend(config=AceStepV15BackendConfig())
mm = MusicManager(backend=backend)
wav_bytes = mm.t2m("upbeat synthwave instrumental", duration_s=10.0)
open("out.wav", "wb").write(wav_bytes)
```

## Quickstart (AbstractCore integration)

```python
from abstractcore import create_llm

llm = create_llm(
    # Any provider/model works here. The LLM does *not* synthesize audio.
    "ollama",
    model="qwen3:4b-instruct",
    music_backend="acemusic",
    music_acemusic_api_key="...",
)

wav_bytes = llm.music.t2m("ambient lo-fi study music", format="wav", duration_s=10.0)
open("out.wav", "wb").write(wav_bytes)
```

## Notes

- The base default backend is `acemusic`, a remote ACE Music API adapter. It requires
  `ACEMUSIC_API_KEY`. Use `ACEMUSIC_BASE_URL` only when targeting a compatible custom endpoint.
- Audio output baseline is **WAV** (no external codecs required). The remote ACE Music backend can
  also request MP3 or FLAC.
- Local model weights are resolved through the default Hugging Face cache on first use (same workflow as Diffusers-based vision).
- Local `model_id` selectors must be Hugging Face repo ids. Local checkpoint directories and custom cache-dir overrides are intentionally not supported.
- The local ACE-Step path is `acestep` / `acestep-diffusers`, which uses package-owned orchestration around Diffusers AceStepPipeline and Hugging Face checkpoint files rather than an external ACE-Step source tree.
- `acestep-v15` remains explicit and quality-limited after repeated-loop validation failures.
- `musicgen` and `stable-audio` are optional small-model comparison backends; both are non-commercial and not default providers.
- For Stable Audio Open Small, install `stable-audio-tools` with `--no-deps` after `abstractmusic[stable-audio]`; AbstractMusic avoids the upstream package's UI/training dependency chain and owns the minimal inference loop.
- The standalone `acestep-v15` backend vendors the checkpoint’s custom Transformers model code into `abstractmusic` so we do **not** use `trust_remote_code` there.
- Known model/provider metadata is packaged in `src/abstractmusic/assets/music_model_capabilities.json`.
  See `docs/models.md` for the reviewed model list and precision policy.
- Full documentation starts at `docs/README.md`, including setup, API, architecture, models,
  troubleshooting, and release process notes.

## CLI / REPL

After installation, `abstractmusic` provides a small CLI:

```bash
# One-shot remote generation (default backend)
abstractmusic t2m "ambient lo-fi study music" --out out.wav --duration 30
abstractmusic --backend acemusic t2m "heroic fantasy epic music" --out out.wav --duration 30
abstractmusic --backend acemusic t2m "upbeat pop song" --lyrics auto --format mp3 --out out.mp3 --duration 30

# One-shot local generation
abstractmusic --backend acestep t2m "ambient lo-fi study music" --out out.wav --duration 10
abstractmusic --backend acestep-v15 t2m "ambient lo-fi study music" --out out.wav --duration 10
abstractmusic --backend acestep-diffusers t2m "ambient lo-fi study music" --out out.wav --duration 10
abstractmusic --backend musicgen t2m "ambient lo-fi study music" --out out.wav --duration 10
abstractmusic --backend stable-audio t2m "short ambient synth loop" --out out.wav --duration 10

# Richer local conditioning for ACE-Step
abstractmusic --backend acestep t2m "heroic fantasy epic music" --enhance-prompt --auto-lyrics --print-plan --out out.wav --duration 30
abstractmusic --backend acestep t2m "heroic fantasy epic instrumental music" --duration 120 --instrumental --print-plan --out out.wav

# Interactive REPL
abstractmusic repl
abstractmusic --engine xl repl
abstractmusic --engine musicgen repl
```

The REPL accepts bare prompts, a reusable `/prompt` + `/run` flow, and slash commands for engine/parameter changes:

```text
/engine xl
/duration 12
/steps 8
/seed 123
/verbose off
/lyrics [Instrumental]
/enhance-prompt on
/structure-prompt on
/auto-lyrics on
/prompt bright melodic synth pop loop with steady drums
/run
bright melodic synth pop loop with steady drums
```

Set duration either at startup (`abstractmusic repl --duration 30`) or inside the REPL
(`/duration 30`). ACE-Step v1.5 expects 10-600 seconds. Add `--verbose` or use `/verbose on` only
when you want backend logs and progress bars.
For generations of 45 seconds or more, `--structure-prompt` is enabled by default and adds a compact
intro/build/bridge/climax/outro section map to the caption. Use `--no-structure-prompt` or
`/structure-prompt off` to pass long prompts through unchanged.

## Text Planning Boundary

AbstractMusic separates text planning from audio synthesis. The built-in planner is dependency-free:
it can enrich short captions, infer simple BPM/key/time hints, preserve explicit lyrics, and produce
template lyrics when `--auto-lyrics` is requested. It is intentionally a fallback, not a full language
model.

Host applications can inject a smarter planner without making AbstractMusic depend on that host:
`MusicManager(..., text_planner=planner, text_planner_mode="auto")` accepts an object with
`create_plan(request)`, an object with `plan_music_text(request_dict)`, or a callable that accepts
`request_dict`. In AbstractCore plugin mode the same hook is exposed through owner config keys
`music_text_planner`, `music_text_planner_instance`, or `music_text_planner_factory`. The compiled
plan is then applied deterministically per backend, and planner provenance is stored in output
metadata.

When AbstractMusic is hosted by AbstractCore, it can also consume a narrow host text-generation
service structurally if one is supplied by the host context or config. The service must expose only
`generate_text(...)` and/or `generate_structured(...)`; AbstractMusic does not import AbstractCore,
does not receive raw provider objects, and keeps the deterministic fallback for standalone use.

The AbstractCore plugin also exposes lightweight music discovery methods (`available_providers`,
`list_models`, `list_provider_models`, `list_operations`, and `capability_catalog`) from packaged
metadata. These methods are import-light and must not instantiate model runtimes.

## Licensing note

- The default base backend calls the configured ACE Music remote API. Check the remote provider's
  terms for generated-output rights and provider-side model licensing.
- The local ACE-Step example uses **ACE-Step Diffusers XL Turbo** (`ACE-Step/acestep-v15-xl-turbo-diffusers`), tagged `license:mit` on Hugging Face, through the package-owned adapter.
- The vendored standalone ACE-Step model code files carry **Apache-2.0** headers (both permissive).
- `facebook/musicgen-small` is exposed through `--backend musicgen`; its model weights are **CC BY-NC 4.0**, so it is a non-commercial validation backend.
- `stabilityai/stable-audio-open-small` is exposed through `--backend stable-audio`; it is gated on Hugging Face and uses the **Stability AI Community License**.
- If you switch to `--backend diffusers`, **model licenses vary** by checkpoint. Choose a model compatible with your intended usage.

## CI/CD

GitHub Actions validates tests, package builds, and documentation builds. Releases run from
`v*.*.*` tags or manual dispatch through `.github/workflows/release.yml`.

Manual dispatch defaults to `publish=false`, which is a rehearsal path: it validates version,
changelog, package build, and docs without creating tags or publishing. To publish manually, set
`publish=true` and `publish_confirmation=publish-abstractmusic-<version>`.

Publishing uses PyPI trusted publishing with the `pypi` environment. Documentation deployment uses
GitHub Pages with the `github-pages` environment. Repository setup must configure PyPI trusted
publisher metadata for `release.yml` and GitHub Pages source as GitHub Actions.

### macOS / Apple Silicon note (MLX/MPS)

On Apple systems, the local `acestep` / `acestep-diffusers` path tries PyTorch MPS first. ACE-Step
Diffusers fp16 can overflow during transformer denoising on MPS, so the automatic dtype prefers MPS
bfloat16 when supported and MPS float32 otherwise. CPU float32 is only the final fallback when MPS
still returns non-finite audio.

Some Diffusers audio pipelines can fail on the `mps` device due to PyTorch backend limitations (typically during vocoder inference).
`abstractmusic` will **retry on CPU** with a clear warning (`#FALLBACK`) when it detects the known MPS channel-limit error.
To force CPU directly, use `--device cpu`.

For the explicit `acestep-v15` PyTorch/MPS path, `abstractmusic` defaults to **fp16** (bf16 disabled) to keep memory usage reasonable on typical unified-memory Macs.
If you run into numerical issues, you can override with `--dtype float32` (at the cost of significantly higher memory use).
The standalone path caps MPS memory to ~16 GiB by setting `PYTORCH_MPS_HIGH_WATERMARK_RATIO` (configurable via `--mps-max-memory-gb` or `--mps-high-watermark-ratio`).
In addition, standalone ACE-Step text-encoder conditioning is executed on **CPU float32** on MPS builds as a compatibility fallback (`#FALLBACK`) to avoid known mixed-dtype MPSGraph kernel aborts; conditioning tensors are cast back to the model dtype/device before diffusion.
The standalone ACE-Step backend keeps turbo controls at `infer_method=ode`, `steps=8`, `shift=3.0`, but uses seeded random source latents for direct text-to-music to avoid silence-conditioned tone collapse.
The experimental 5Hz LM audio-code planner is off by default because using coarse code hints as full cover conditioning can imprint repetitive artifacts.
If a standalone run returns non-finite latents, `abstractmusic` retries once with the alternate infer method using an incremented seed (`#FALLBACK`) instead of writing a silent/invalid WAV.

For instrumental standalone ACE-Step runs, pass lyrics as `[Instrumental]`.
Standalone decoded waveforms are DC-centered before normalization to avoid one-sided/noisy artifacts from amplifying tiny decoder bias.

Upstream references:
- PyTorch MPS env var `PYTORCH_ENABLE_MPS_FALLBACK=1` (fallback to CPU when an op is unsupported): `https://docs.pytorch.org/docs/stable/mps_environment_variables.html`
- Example upstream issue tracking the specific MPS channel-limit error: `https://github.com/pytorch/pytorch/issues/144445`
