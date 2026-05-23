# AbstractMusic

`abstractmusic` is a model-agnostic **text-to-music / text-to-audio** library designed to plug into **AbstractCore** as an optional capability plugin. The base install is lightweight and remote-capable; local model runtimes live behind explicit extras.

## Install

```bash
pip install abstractmusic
```

The base package is import-light: contracts, manager, CLI shell, plugin wiring, docs, model
metadata, and stdlib-only remote music backends for ACE Music and ElevenLabs Music. It does not
install Torch, Diffusers, Transformers, or NumPy.

Use the base install with a remote API key:

```bash
export ACEMUSIC_API_KEY=...
abstractmusic t2m "ambient lo-fi study music" --out out.wav --duration 30
```

Or select ElevenLabs Music explicitly:

```bash
export ELEVENLABS_API_KEY=...
abstractmusic --backend elevenlabs t2m "cinematic instrumental synth cue" --format mp3 --out out.mp3 --duration 30
```

Install a local runtime profile when you want in-process model generation:

```bash
pip install "abstractmusic[remote]"  # no-op alias; base install already contains remote clients
pip install "abstractmusic[acestep]"  # local ACE-Step path
pip install "abstractmusic[stable-audio-3]"  # internal Stable Audio 3 runtime, gated HF weights
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

ElevenLabs Music is also available as a music-only remote backend:

```python
from abstractmusic import MusicManager
from abstractmusic.backends import ElevenLabsMusicBackend, ElevenLabsMusicBackendConfig

backend = ElevenLabsMusicBackend(config=ElevenLabsMusicBackendConfig(api_key="..."))
mm = MusicManager(backend=backend)
mp3_bytes = mm.t2m("cinematic instrumental synth cue", duration_s=30.0, format="mp3")
open("out.mp3", "wb").write(mp3_bytes)
```

## Quickstart (local generation)

```python
from abstractmusic import MusicManager
from abstractmusic.backends import AceStepBackend, AceStepBackendConfig

backend = AceStepBackend(config=AceStepBackendConfig())

mm = MusicManager(backend=backend)
wav_bytes = mm.t2m("uplifting synthwave with punchy drums", duration_s=10.0)
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

You can also pass provider-neutral style tags (instruments, vocalist traits, mix notes) through the
same contract:

```python
wav_bytes = llm.music.t2m(
    "a clean jazz arrangement with a clear chorus lift",
    duration_s=25.0,
    lyrics="auto",
    positive_styles=["saxophone", "brushed drums", "female vocalist"],
    negative_styles=["no autotune", "no chipmunk voice"],
)
open("jazz.wav", "wb").write(wav_bytes)
```

## Notes

- The base default backend is `acemusic`, a remote ACE Music API adapter. It requires
  `ACEMUSIC_API_KEY`. Use `ACEMUSIC_BASE_URL` only when targeting a compatible custom endpoint.
- The second remote backend is `elevenlabs`, which calls only ElevenLabs Music endpoints and
  requires `ELEVENLABS_API_KEY`. Use AbstractVoice, not AbstractMusic, for ElevenLabs voice/TTS.
- Audio output baseline is **WAV** (no external codecs required). The remote ACE Music backend can
  also request MP3 or FLAC.
- Local model weights are resolved through the default Hugging Face cache on first use (same workflow as Diffusers-based vision).
- Local `model_id` selectors must be Hugging Face repo ids. Local checkpoint directories and custom cache-dir overrides are intentionally not supported.
- The local ACE-Step path is `acestep`, which uses package-owned orchestration around Diffusers AceStepPipeline and Hugging Face checkpoint files rather than an external ACE-Step source tree.
- `musicgen`, `stable-audio`, and `stable-audio-3` are optional local comparison/generation
  backends; they are not default providers.
- Stable Audio Open Small uses AbstractMusic-vendored `stable-audio-tools==0.0.19` model code and an internal minimal inference loop; you do not need to install the upstream `stable-audio-tools` package.
- `stable-audio-3` uses an AbstractMusic-owned internal inference subset with Hugging Face
  weights/configs. It does not import the upstream `stable_audio_3` package or require a local
  Stable Audio checkout. Model terms must be accepted on Hugging Face. The current implementation
  has passed focused 30-second and 120-second Small Music validation runs; broader prompt/seed and
  GPU validation are still required before it is marked recommended.
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
abstractmusic --backend elevenlabs t2m "cinematic instrumental synth cue" --format mp3 --out out.mp3 --duration 30
abstractmusic --backend elevenlabs t2m "upbeat pop song" --lyrics auto --composition-mode plan --format mp3 --out out.mp3 --duration 30

# One-shot local generation
abstractmusic --backend acestep t2m "ambient lo-fi study music" --out out.wav --duration 10
abstractmusic --backend musicgen t2m "ambient lo-fi study music" --out out.wav --duration 10
abstractmusic --backend stable-audio t2m "short ambient synth loop" --out out.wav --duration 10
abstractmusic --backend stable-audio-3 t2m "rhythmic space shooter game music" --out out.wav --duration 30 --steps 16

# Sound effects (Stable Audio 3 Small SFX)
abstractmusic t2m \
  --backend stable-audio-3 \
  --model-id stabilityai/stable-audio-3-small-sfx \
  --duration 3 \
  --out sfx.wav \
  "A short sci-fi laser zap, crisp transient, no reverb tail"

# Sound effects (Stable Audio Open Small)
abstractmusic t2m \
  --backend stable-audio \
  --duration 11 \
  --out slam.wav \
  "A heavy wooden door slam, close-mic, natural room tail"

# Richer local conditioning for ACE-Step
abstractmusic --backend acestep t2m "heroic fantasy epic music" --enhance-prompt --auto-lyrics --print-plan --out out.wav --duration 30
abstractmusic --backend acestep t2m "heroic fantasy epic instrumental music" --duration 120 --instrumental --print-plan --out out.wav

# Vocals + lyrics (ACE Music remote)
abstractmusic t2m \
  --backend acemusic \
  --duration 25 \
  --format mp3 \
  --lyrics auto \
  --style "soulful pop, female vocalist, warm bass, tight drums, big chorus, modern radio mix" \
  --out soulful-pop.mp3 \
  "A soulful pop song with a big uplifting chorus and a catchy hook"

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
(`/duration 30`). Add `--verbose` or use `/verbose on` only when you want backend logs and
progress bars.
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
metadata. Planners may also return `composition_plan`; compatible backends such as `elevenlabs`
translate it into native structured music plans, while other backends continue using the compiled
prompt and lyrics.

When AbstractMusic is hosted by AbstractCore, it can also consume a narrow host text-generation
service structurally if one is supplied by the host context or config. The service must expose only
`generate_text(...)` and/or `generate_structured(...)`; AbstractMusic does not import AbstractCore,
does not receive raw provider objects, and keeps the deterministic fallback for standalone use.

The AbstractCore plugin also exposes lightweight music discovery methods (`available_providers`,
`list_models`, `list_provider_models`, `list_operations`, and `capability_catalog`) from packaged
metadata. Discovery uses backend-oriented provider ids such as `acemusic`, `elevenlabs`,
`acestep`, `stable-audio`, and `stable-audio-3`, and only reports providers/models whose runtime
is usable in the current environment. These methods are import-light and must not instantiate model
runtimes.

## Licensing note

- The default base backend calls the configured ACE Music remote API. Check the remote provider's
  terms for generated-output rights and provider-side model licensing.
- The `elevenlabs` backend calls ElevenLabs Music only. ElevenLabs voice/TTS belongs in
  AbstractVoice. ElevenLabs Music API access may require a paid Music-enabled account tier.
- The local ACE-Step example uses **ACE-Step Diffusers XL Turbo** (`ACE-Step/acestep-v15-xl-turbo-diffusers`), tagged `license:mit` on Hugging Face, through the package-owned `acestep` adapter.
- `facebook/musicgen-small` is exposed through `--backend musicgen`; its model weights are **CC BY-NC 4.0**, so it is a non-commercial validation backend.
- `stabilityai/stable-audio-open-small` is exposed through `--backend stable-audio`; it is gated on Hugging Face and uses the **Stability AI Community License**.
- `stabilityai/stable-audio-3-small-music` is exposed through `--backend stable-audio-3`;
  it is gated on Hugging Face, uses the **Stability AI Community License** plus text-encoder
  terms, and runs through AbstractMusic-owned internal runtime code.
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

On Apple systems, the local `acestep` path tries PyTorch MPS first. ACE-Step
Diffusers fp16 can overflow during transformer denoising on MPS, so the automatic dtype prefers MPS
bfloat16 when supported and MPS float32 otherwise. CPU float32 is only the final fallback when MPS
still returns non-finite audio.

Some Diffusers audio pipelines can fail on the `mps` device due to PyTorch backend limitations (typically during vocoder inference).
`abstractmusic` will **retry on CPU** with a clear warning (`#FALLBACK`) when it detects the known MPS channel-limit error.
To force CPU directly, use `--device cpu`.

If you run into numerical issues, you can override with `--dtype float32` (at the cost of significantly higher memory use).
The `acestep` backend now follows checkpoint-specific defaults: turbo checkpoints use 8 steps with
guidance effectively off, while base and SFT checkpoints should usually run around 50 steps with
guidance enabled. If you need instrumental output, pass lyrics as `[Instrumental]`.

Upstream references:
- PyTorch MPS env var `PYTORCH_ENABLE_MPS_FALLBACK=1` (fallback to CPU when an op is unsupported): `https://docs.pytorch.org/docs/stable/mps_environment_variables.html`
- Example upstream issue tracking the specific MPS channel-limit error: `https://github.com/pytorch/pytorch/issues/144445`
