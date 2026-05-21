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

On Apple hardware, the local ACE-Step Diffusers backend tries PyTorch MPS first. The automatic
dtype policy avoids fp16 denoising on MPS because it can overflow during transformer inference on
some local stacks. AbstractMusic prefers MPS bfloat16 when supported, then MPS float32, and only
falls back to CPU float32 if MPS still returns invalid audio.

To force CPU directly:

```bash
abstractmusic --backend acestep --device cpu t2m "ambient music" --out out.wav --duration 10
```

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
Hugging Face account used by the machine. The runtime also expects `stable-audio-tools` to be
installed without its full dependency chain:

```bash
pip install "abstractmusic[stable-audio]"
pip install --no-deps stable-audio-tools==0.0.19
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
