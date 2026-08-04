# Troubleshooting

Use this page for user-visible setup, runtime, and release failures. For model support and license
boundaries, see [models.md](models.md). For the public API, see [api.md](api.md).

## Default remote generation fails with a missing API key

The base install uses the ACE Music remote backend by default. Set `ACEMUSIC_API_KEY`, or pass the
key through host configuration:

```bash
export ACEMUSIC_API_KEY=...
abstractmusic t2m "ambient lo-fi study music" --out out.wav --duration 30
```

If you use a compatible proxy or self-hosted endpoint, set `ACEMUSIC_BASE_URL` or pass
`--acemusic-base-url`.

## ACE Music returns the wrong duration

ACE-Step OpenRouter-compatible servers support a *tagged mode* where the caption prompt is wrapped
in `<prompt>...</prompt>`. Without tags, some deployments can ignore `audio_config.duration` and
return audio that is longer or shorter than requested.

AbstractMusic uses tagged prompts by default for the ACE Music backend (unless `sample_mode` is
enabled). It also enforces duration **only for WAV outputs** by trimming or padding the returned
WAV to the requested duration.

If you need exact durations, use WAV:

```bash
abstractmusic t2m "ambient lo-fi study music" --duration 25 --format wav --out out.wav
```

If you request `--format mp3` or `--format flac`, duration is provider-controlled and may not
match exactly; prefer a local backend when strict duration matters.

## ElevenLabs Music returns `limited_access` or HTTP 402

The `elevenlabs` backend uses `ELEVENLABS_API_KEY` and only calls ElevenLabs Music endpoints.
If live generation returns HTTP 402 with `limited_access`, the key authenticated but the account
tier does not have Music API access. Upgrade to a Music-enabled paid plan or use another backend:

```bash
export ELEVENLABS_API_KEY=...
abstractmusic --backend elevenlabs t2m "cinematic instrumental synth cue" --format mp3 --out out.mp3 --duration 30
```

## Local generation fails with missing packages

The base package is import-light and does not install heavy model runtimes. Install the backend
extra you plan to use:

```bash
pip install "abstractmusic[acestep]"
```

For local development:

```bash
python -m pip install -e ".[acestep,test]"
```

Verify the CLI sees the backend:

```bash
abstractmusic --backend acestep t2m "short ambient synth loop" --out out.wav --duration 10
```

## AbstractCore discovery lists no music providers

`available_providers(...)` reports only providers that can run right now. A provider is left out
when its runtime extra is not installed, when its weights are not in the Hugging Face cache, or when
its remote API does not answer. Ask the same capability object for `provider_details(...)`, which
reports every provider along with the reason it is unusable:

```python
for item in capability.provider_details(task="text_to_music"):
    print(item["provider_id"], item["usable"], item["status"], item["metadata"]["reason"])
```

Match the reported `status` to the fix:

| `status` | Meaning | Fix |
| --- | --- | --- |
| `not-installed` | The optional runtime extra is missing. | `pip install "abstractmusic[acestep]"` |
| `no-local-weights` | The runtime is installed, but no model is downloaded. | Download a model id from `metadata.models` |
| `no-loadable-weights` | The only cached checkpoints use a repository layout the backend cannot load. | Download the Diffusers-layout variant named in `metadata.reason` |
| `incompatible-model` | The configured `music_model_id` is known to be unloadable by any pipeline. | Configure a Diffusers-layout model id |
| `not-configured` | No API key for a remote provider. | Set `ACEMUSIC_API_KEY` or `ELEVENLABS_API_KEY` |
| `unauthorized` | The API rejected the key. | Check the key and its account entitlements |
| `unreachable` | The endpoint did not answer within 5 seconds. | Check connectivity, base URL, and any proxy |
| `unavailable` | The provider returned a server error. | Retry later; check the provider's status page |

`metadata.cached_models` lists which of the provider's models are already downloaded, and
`metadata.models` lists everything it supports, so you can see what is available to fetch.

If a model looks downloaded but still reports `no-local-weights`, the cached copy is incomplete —
a sharded checkpoint missing some of its shards is reported as not ready, because loading it would
start a large download. Check a specific model id directly:

```bash
python -c "from abstractmusic.availability import hf_cache_root, is_model_cached; print(hf_cache_root(), is_model_cached('ACE-Step/acestep-v15-xl-turbo-diffusers'))"
```

If that prints `False` for a model you expect, re-run the download to completion. See
[Architecture](architecture.md#discovery-boundary) for how availability is determined.

## ACE-Step downloads or cache access fails

The local `acestep` backend uses Hugging Face model weights. It does not accept local ACE-Step
source-tree paths or checkpoint-directory overrides. Confirm the model is accessible from the
standard Hugging Face cache:

```bash
python - <<'PY'
from huggingface_hub import model_info
print(model_info("ACE-Step/acestep-v15-xl-turbo-diffusers").id)
PY
```

If the machine is offline, populate the Hugging Face cache before running generation.

## Apple MPS returns non-finite audio or falls back to CPU

On Apple hardware, the local ACE-Step backend tries PyTorch MPS first. The automatic
dtype policy avoids fp16 denoising on MPS because it can overflow during transformer inference on
some local stacks. AbstractMusic prefers MPS bfloat16 when supported, then MPS float32, and only
falls back to CPU float32 if MPS still returns invalid audio.

To force CPU directly:

```bash
abstractmusic --backend acestep --device cpu t2m "ambient music" --out out.wav --duration 10
```

## Output sounds like wind or noise instead of music

The guided ACE-Step XL checkpoints (`acestep-v15-xl-sft-diffusers`,
`acestep-v15-xl-base-diffusers`) can collapse into wind/whoosh-like sweeps when conditioned on
long template captions. Short raw prompts work reliably. Check and fix:

- Do not pass `--enhance-prompt` with XL sft/base checkpoints; template caption expansion is the
  known trigger, and the planner warns (`caption_expansion_on_caption_sensitive_model`) when you
  override it. (With the turbo checkpoint, `--enhance-prompt` is safe and can help.)
- Long-form structure maps and `--auto-lyrics` avoid the known trigger on these checkpoints: the
  registry marks them caption-sensitive, so the deterministic planner renders those captions
  compactly (your prompt plus the section map, no template prose) and records
  `compact_caption_for_caption_sensitive_model`. This removes the caption shape that failed at
  30 seconds; it is not a separate quality guarantee. An injected LLM text planner owns its own
  caption content — a long planner caption on a sensitive checkpoint is recorded as
  `long_planner_caption_on_caption_sensitive_model` in provenance rather than rewritten.
- Confirm what actually reached the model with `--print-plan`.
- Generation metadata reports the screen verdict: `audio_stats.probably_noise_texture` is `true`
  when the output matches the wind/whoosh signature. Retry with a different seed or a shorter
  prompt.

## Long generations sound static or repetitive

For generations of 45 seconds or more, the CLI enables `--structure-prompt` by default. This adds a
compact section map to the caption. For more control, pass explicit structure and use a seed:

```bash
abstractmusic --backend acestep t2m \
  "rhythmic space shooter game music, fast drums, evolving synth bass, arcade lead theme" \
  --duration 120 \
  --seed 123 \
  --enhance-prompt \
  --print-plan \
  --out out.wav
```

ACE-Step is still a local model with limits. If a candidate contains clock-like repeated tones,
single-note collapse, or long near-silent pauses, treat it as a failed generation and try a new
seed or a more explicit prompt.

## Stable Audio Open Small cannot download weights

`stabilityai/stable-audio-open-small` is gated on Hugging Face. Accept the model terms with the
Hugging Face account used by the machine and expose a token as `HF_TOKEN` or
`HUGGINGFACE_HUB_TOKEN`.

```bash
pip install "abstractmusic[stable-audio]"
export HF_TOKEN=...
```

## Stable Audio 3 cannot download weights

`stabilityai/stable-audio-3-small-music` and `stabilityai/stable-audio-3-medium` are gated on
Hugging Face. Accept the model terms with the account used by the machine and expose a token as
`HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`.

```bash
pip install "abstractmusic[stable-audio-3]"
export HF_TOKEN=...
abstractmusic --backend stable-audio-3 t2m "rhythmic space shooter game music" --duration 30 --steps 16 --out out.wav
```

This backend uses AbstractMusic-owned internal runtime code. Do not install the upstream
`stable_audio_3` package or point AbstractMusic at a local Stable Audio checkout.

## Release workflow refuses to publish

The release workflow validates the package version, changelog entry, tag, and PyPI availability.
Manual publishing requires:

- `version` matching `src/abstractmusic/_version.py`
- `publish=true`
- `publish_confirmation=publish-abstractmusic-<version>`

If PyPI already has the target version, bump the package version and changelog instead of trying to
republish the same artifact.
