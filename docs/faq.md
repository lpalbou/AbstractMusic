# FAQ

## Does the fast test suite prove real music generation works?

No. The current unit tests mostly validate abstraction and tensor plumbing with fakes. Real model
smoke tests are planned and must be opt-in because checkpoints are large.

## Which model should be the default?

ACE-Step v1.5 through `acestep-official` is the current practical default because it is
music-focused, MIT-licensed, and uses the upstream 5Hz LM planner. The ACE-Step XL Turbo Diffusers
checkpoint remains a preferred next provider candidate because it uses a cleaner Diffusers pipeline
layout.

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
