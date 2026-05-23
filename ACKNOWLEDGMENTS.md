# Acknowledgments

AbstractMusic builds on open-source libraries, openly released model work, and the broader
AbstractFramework ecosystem.

## Runtime libraries

- Hugging Face Diffusers: local audio/music pipeline loading and the ACE-Step Diffusers pipeline.
- PyTorch: tensor runtime for local inference.
- Hugging Face Transformers: text encoders, tokenizers, and custom model loading.
- Hugging Face Hub: model discovery and checkpoint download.
- Safetensors: checkpoint file format support.
- Einops: tensor operations used by local runtime code and tests.
- NumPy: audio array handling and tests.

## Model families reviewed or integrated

- ACE-Step v1.5 and ACE-Step XL checkpoints from the ACE-Step project. The reviewed Hugging Face
  model cards declare MIT licensing for the ACE-Step weights.
- Qwen text encoder artifacts bundled with ACE-Step Diffusers checkpoints. The ACE-Step Diffusers
  model card notes Apache-2.0 licensing for the text encoder components.
- HeartMuLa and HeartCodec from the HeartMuLa project. The reviewed HeartMuLa Hugging Face and
  GitHub pages declare Apache-2.0 licensing.
- YuE from the m-a-p project, reviewed as a possible multi-stage lyrics-to-music provider. The
  reviewed Stage-1 Hugging Face metadata declares Apache-2.0 licensing, but Stage-1 is not an
  end-to-end audio generator by itself.
- TinyMozart, reviewed as a possible small MIDI/piano provider. Its Hugging Face metadata did not
  declare a license during the 2026-05-15 review, so it is not a default provider candidate.
- Omni2Sound, reviewed as a multimodal text/video-to-audio provider. Its Hugging Face model card
  declares CC BY-NC 4.0, so it is not suitable as a default commercial-capable provider.

## Integrations

- AbstractCore: capability plugin discovery and runtime artifact-store integration.
- AbstractVision and AbstractVoice: architecture, docs, packaging, and backlog conventions used as
  references for this project.
