import pytest


@pytest.mark.unit
def test_acestep_aligns_conditioning_to_model_dtype(monkeypatch, tmp_path):
    import torch

    from abstractmusic.backends.acestep_v15 import AceStepV15Backend, AceStepV15BackendConfig
    from abstractmusic.types import AudioGenerationRequest

    silence_path = tmp_path / "silence_latent.pt"
    torch.save(torch.zeros((1, 64, 1000), dtype=torch.float32), silence_path)

    def _fake_hf_hub_download(*, repo_id, revision=None, cache_dir=None, filename, **kwargs):
        _ = repo_id, revision, cache_dir, filename, kwargs
        return str(silence_path)

    monkeypatch.setattr(
        "abstractmusic.backends.acestep_v15._lazy_import_hf_hub_download",
        lambda: _fake_hf_hub_download,
    )

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
            # Keep embedding in float32 to force mismatch with model float16.
            self._emb = torch.nn.Embedding(100, 1024, dtype=torch.float32)

        def to(self, *_a, **_k):
            return self

        def eval(self):
            return self

        def forward(self, input_ids=None, attention_mask=None, lyric_attention_mask=None):
            _ = attention_mask, lyric_attention_mask
            return _EncOut(self._emb(input_ids))  # float32

        __call__ = forward

        def get_input_embeddings(self):
            return self._emb

    class _FakeModel:
        def __init__(self):
            self._p = torch.nn.Parameter(torch.zeros((1,), dtype=torch.float16))

        def parameters(self):
            return iter([self._p])

        def to(self, *_a, **_k):
            return self

        def eval(self):
            return self

        def generate_audio(self, **kwargs):
            assert kwargs["text_hidden_states"].dtype == torch.float16
            assert kwargs["lyric_hidden_states"].dtype == torch.float16
            assert kwargs["src_latents"].dtype == torch.float16
            t = int(kwargs["src_latents"].shape[1])
            return {"target_latents": torch.zeros((1, t, 64), dtype=torch.float16), "time_costs": {}}

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
            b = int(latents.shape[0])
            samples = int(latents.shape[-1] * 10)
            return _VaeOut(sample=torch.zeros((b, 2, samples), dtype=torch.float32))

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

    class _FakeAceStepConditionGenerationModel:
        @staticmethod
        def from_pretrained(*_a, **_k):
            return _FakeModel()

    monkeypatch.setattr(
        "abstractmusic.backends.acestep_v15._lazy_import_acestep_v15_turbo_model",
        lambda: _FakeAceStepConditionGenerationModel,
    )

    def _lazy_oobleck():
        class _AutoencoderOobleck:
            @staticmethod
            def from_pretrained(*_a, **_k):
                return _FakeVAE()

        return _AutoencoderOobleck

    monkeypatch.setattr("abstractmusic.backends.acestep_v15._lazy_import_diffusers_oobleck", _lazy_oobleck)

    backend = AceStepV15Backend(config=AceStepV15BackendConfig(repo_id="ACE-Step/Ace-Step1.5", device="cpu"))
    asset = backend.generate_audio(AudioGenerationRequest(prompt="sci fi music", duration_s=1.0))
    assert asset.mime_type == "audio/wav"
    assert bytes(asset.data)[:4] == b"RIFF"
