# FAQ

## Does the fast test suite prove real music generation works?

No. The current unit tests mostly validate abstraction and tensor plumbing with fakes. Real model
smoke tests are planned and must be opt-in because checkpoints are large.

## Which model should be the default?

The base default is `acemusic`, a lightweight remote ACE Music API backend, because the default
package profile should not install local ML runtimes. For local generation, ACE-Step Diffusers XL
Turbo through `acestep` / `acestep-diffusers` is the supported Apple/GPU path that passed the
current 30-second reference-floor smoke. The explicit `acestep-v15` backend uses vendored model
code but remains quality-limited after repeated-loop validation failures.

## Does AbstractMusic use Suno, OpenAI, or Anthropic by default?

No. OpenAI exposes speech/audio APIs but not a text-to-music endpoint suited to this backend.
Anthropic does not expose music generation. Suno has a consumer music product, but the public API
surface found during review is third-party or unofficial, so it is not a default backend. A Suno
adapter can be added later if there is an official, stable, documented API contract.

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
