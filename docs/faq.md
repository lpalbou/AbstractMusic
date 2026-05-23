# FAQ

## Does the fast test suite prove real music generation works?

No. The current unit tests mostly validate abstraction and tensor plumbing with fakes. Real model
smoke tests are planned and must be opt-in because checkpoints are large.

## Which model should be the default?

The base default is `acemusic`, a lightweight remote ACE Music API backend, because the default
package profile should not install local ML runtimes. For local generation, ACE-Step Diffusers XL
Turbo through `acestep` is the supported Apple/GPU path that passed the current 30-second
reference-floor smoke.

`elevenlabs` is the second supported remote backend, but it is explicit rather than the default.
Two remote endpoints are enough for the lightweight package profile; new provider work should now
focus on local/open-weight engines unless a remote endpoint unlocks a missing abstraction.

## Does AbstractMusic use Suno, OpenAI, or Anthropic by default?

No. OpenAI exposes speech/audio APIs but not a text-to-music endpoint suited to this backend.
Anthropic does not expose music generation. Suno has a consumer music product, but the public API
surface found during review is third-party or unofficial, so it is not a default backend. A Suno
adapter can be added later if there is an official, stable, documented API contract.

## Does the ElevenLabs backend include voice?

No. AbstractMusic calls only ElevenLabs Music endpoints. Voice, text-to-speech, speech-to-speech,
and voice cloning belong in AbstractVoice.

## How do I specify instruments or vocalist traits?

Use plain text in the prompt (for example: “jazz with saxophone and brushed drums”) or set
provider-neutral style tags:

- CLI: `--style "saxophone, brushed drums, female vocalist"`
- REPL: `/style saxophone, brushed drums, female vocalist`

This nudges the model, but it is not a guaranteed identity selector (no voice cloning here).

## Can HeartMuLa generate music?

Yes. The HeartMuLa model card describes it as a text-to-audio/music model, and the official
HeartMuLa repo documents lyrics and tag conditioned generation with HeartCodec. It is Apache-2.0
licensed, but it is a heavy optional provider candidate rather than a base dependency.

## Why not make Omni2Sound a default provider?

Omni2Sound is CC BY-NC 4.0, very large, CUDA/script-oriented, and focused on video/text-to-audio
and foley-style generation. It should not be selected silently for commercial-capable music
generation.

## Do we use 8-bit models?

When official 8-bit model artifacts exist, they should be preferred. If no official 8-bit artifact
exists, use the smallest official 16-bit path. The registry records whether a reviewed model has an
official 8-bit source.
