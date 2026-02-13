import types

import pytest


@pytest.mark.unit
def test_acestep_backend_generates_wav_bytes_with_fakes(monkeypatch, tmp_path):
    import torch

    from abstractmusic.backends.acestep_v15 import AceStepV15Backend, AceStepV15BackendConfig
    from abstractmusic.types import AudioGenerationRequest

    # ---------------------------------------------------------------------
    # Fake HF hub download: provide a tiny silence_latent.pt on disk.
    # Upstream tensor layout is [1, C, T]; backend transposes to [1, T, C].
    # ---------------------------------------------------------------------
    silence_path = tmp_path / "silence_latent.pt"
    torch.save(torch.zeros((1, 64, 1000), dtype=torch.float32), silence_path)

    def _fake_hf_hub_download(*, repo_id, revision=None, cache_dir=None, filename):
        _ = repo_id, revision, cache_dir, filename
        return str(silence_path)

    monkeypatch.setattr(
        "abstractmusic.backends.acestep_v15._lazy_import_hf_hub_download",
        lambda: _fake_hf_hub_download,
    )

    # ---------------------------------------------------------------------
    # Fake tokenizer + text encoder
    # ---------------------------------------------------------------------
    class _TokOut:
        def __init__(self, input_ids):
            self.input_ids = input_ids
            self.attention_mask = torch.ones_like(input_ids)

    class _FakeTokenizer:
        model_max_length = 4096

        def __call__(self, text, return_tensors="pt", padding=False, truncation=False):
            _ = text, return_tensors, padding, truncation
            return _TokOut(torch.tensor([[1, 2, 3, 4]], dtype=torch.long))

    class _EncOut:
        def __init__(self, last_hidden_state):
            self.last_hidden_state = last_hidden_state

    class _FakeTextEncoder:
        def __init__(self):
            self._emb = torch.nn.Embedding(100, 1024)

        def to(self, *_a, **_k):
            return self

        def eval(self):
            return self

        def forward(self, input_ids=None, attention_mask=None, lyric_attention_mask=None):
            _ = attention_mask, lyric_attention_mask
            # [B, L, 1024]
            hs = self._emb(input_ids)
            return _EncOut(hs)

        __call__ = forward

        def get_input_embeddings(self):
            return self._emb

    # ---------------------------------------------------------------------
    # Fake ACE-Step DiT model
    # ---------------------------------------------------------------------
    class _FakeModel:
        def to(self, *_a, **_k):
            return self

        def eval(self):
            return self

        def generate_audio(self, **kwargs):
            # Validate a few key shapes to ensure we passed the right tensors.
            src = kwargs["src_latents"]
            assert isinstance(src, torch.Tensor)
            assert src.ndim == 3 and src.shape[-1] == 64
            # Text2music path now initializes src latents from seeded random noise.
            assert float(src.std().item()) > 0.0
            # Default chunk mask is zeros to avoid injecting constant features.
            chunk = kwargs["chunk_masks"]
            assert isinstance(chunk, torch.Tensor)
            assert float(chunk.abs().max().item()) == 0.0
            # No lyrics were provided in this test request; backend should pass
            # a null lyric mask instead of synthetic lyric text.
            lyr_mask = kwargs["lyric_attention_mask"]
            assert isinstance(lyr_mask, torch.Tensor)
            assert int(lyr_mask.shape[-1]) == 1
            assert float(lyr_mask.abs().max().item()) == 0.0
            t = int(src.shape[1])
            return {"target_latents": torch.zeros((1, t, 64), dtype=torch.float32), "time_costs": {}}

    # ---------------------------------------------------------------------
    # Fake VAE (AutoencoderOobleck)
    # ---------------------------------------------------------------------
    class _VaeOut:
        def __init__(self, sample):
            self.sample = sample

    class _FakeVAE:
        dtype = torch.float32

        def to(self, *_a, **_k):
            return self

        def eval(self):
            return self

        def decode(self, latents):
            # latents: [B, 64, T] -> return stereo [B, 2, samples]
            b = int(latents.shape[0])
            samples = int(latents.shape[-1] * 10)
            return _VaeOut(sample=torch.zeros((b, 2, samples), dtype=torch.float32))

    # transformers import shim
    def _lazy_transformers():
        class _AutoModel:
            @staticmethod
            def from_pretrained(*_a, **_k):
                sub = str(_k.get("subfolder") or "")
                if "Qwen3-Embedding-0.6B" in sub:
                    return _FakeTextEncoder()
                return _FakeModel()

        class _AutoTokenizer:
            @staticmethod
            def from_pretrained(*_a, **_k):
                return _FakeTokenizer()

        return _AutoModel, _AutoTokenizer

    monkeypatch.setattr("abstractmusic.backends.acestep_v15._lazy_import_transformers", _lazy_transformers)

    # vendored ACE-Step turbo model shim (avoid importing/downloading real model code)
    class _FakeAceStepConditionGenerationModel:
        @staticmethod
        def from_pretrained(*_a, **_k):
            return _FakeModel()

    monkeypatch.setattr(
        "abstractmusic.backends.acestep_v15._lazy_import_acestep_v15_turbo_model",
        lambda: _FakeAceStepConditionGenerationModel,
    )

    # diffusers shim
    def _lazy_oobleck():
        class _AutoencoderOobleck:
            @staticmethod
            def from_pretrained(*_a, **_k):
                return _FakeVAE()

        return _AutoencoderOobleck

    monkeypatch.setattr("abstractmusic.backends.acestep_v15._lazy_import_diffusers_oobleck", _lazy_oobleck)

    # ---------------------------------------------------------------------
    # Run
    # ---------------------------------------------------------------------
    backend = AceStepV15Backend(config=AceStepV15BackendConfig(repo_id="ACE-Step/Ace-Step1.5", device="cpu"))
    asset = backend.generate_audio(AudioGenerationRequest(prompt="sci fi music", duration_s=1.0))

    assert asset.mime_type == "audio/wav"
    assert isinstance(asset.data, (bytes, bytearray))
    assert bytes(asset.data)[:4] == b"RIFF"
