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

    def _fake_hf_hub_download(*, repo_id, revision=None, cache_dir=None, filename, **kwargs):
        _ = repo_id, revision, cache_dir, filename, kwargs
        return str(silence_path)

    monkeypatch.setattr(
        "abstractmusic.backends.acestep_v15._lazy_import_hf_hub_download",
        lambda: _fake_hf_hub_download,
    )

    # ---------------------------------------------------------------------
    # Fake tokenizer + text encoder
    # ---------------------------------------------------------------------
    seen = {"texts": []}

    class _TokOut:
        def __init__(self, input_ids):
            self.input_ids = input_ids
            self.attention_mask = torch.ones_like(input_ids)

    class _FakeTokenizer:
        model_max_length = 4096

        def __call__(self, text, return_tensors="pt", padding=False, truncation=False, max_length=None):
            _ = return_tensors, padding, truncation, max_length
            seen["texts"].append(text)
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
            # The standalone text2music path uses seeded random source latents
            # to avoid collapsing into static silence-conditioned tones.
            assert float(src.abs().max().item()) > 0.0
            # Upstream "auto" chunk mask mode uses 2.0 sentinels.
            chunk = kwargs["chunk_masks"]
            assert isinstance(chunk, torch.Tensor)
            assert float(chunk.min().item()) == 2.0
            assert float(chunk.max().item()) == 2.0
            # Upstream always formats the lyric branch, even when lyrics are empty.
            lyr_mask = kwargs["lyric_attention_mask"]
            assert isinstance(lyr_mask, torch.Tensor)
            assert int(lyr_mask.shape[-1]) > 0
            assert bool(lyr_mask.any().item()) is True
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
    backend = AceStepV15Backend(
        config=AceStepV15BackendConfig(
            repo_id="ACE-Step/Ace-Step1.5",
            device="cpu",
            quality_retry_enabled=False,
        )
    )
    asset = backend.generate_audio(AudioGenerationRequest(prompt="sci fi music", duration_s=1.0))

    assert asset.mime_type == "audio/wav"
    assert isinstance(asset.data, (bytes, bytearray))
    assert bytes(asset.data)[:4] == b"RIFF"
    assert any(text.startswith("# Languages\nunknown\n\n# Lyric\n") for text in seen["texts"])


@pytest.mark.unit
def test_acestep_backend_uses_internal_lm_planner_only_when_enabled(monkeypatch, tmp_path):
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

    seen = {"texts": []}

    class _TokOut:
        def __init__(self, input_ids):
            self.input_ids = input_ids
            self.attention_mask = torch.ones_like(input_ids)

    class _FakeTokenizer:
        model_max_length = 4096

        def __call__(self, text, return_tensors="pt", padding=False, truncation=False, max_length=None):
            _ = return_tensors, padding, truncation, max_length
            seen["texts"].append(text)
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
            return _EncOut(self._emb(input_ids))

        __call__ = forward

        def get_input_embeddings(self):
            return self._emb

    class _FakeModel:
        def __init__(self):
            class _FakeQuantizer:
                @staticmethod
                def get_output_from_indices(indices):
                    return indices.to(dtype=torch.float32).repeat(1, 1, 64)

            class _FakeTokenizerState:
                quantizer = _FakeQuantizer()

            class _FakeDetokenizer:
                def __call__(self, quantized):
                    return quantized

            self.tokenizer = _FakeTokenizerState()
            self.detokenizer = _FakeDetokenizer()

        def to(self, *_a, **_k):
            return self

        def eval(self):
            return self

        def generate_audio(self, **kwargs):
            src = kwargs["src_latents"]
            assert float(src.abs().max().item()) > 0.0
            chunk = kwargs["chunk_masks"]
            assert float(chunk.min().item()) == 2.0
            assert float(chunk.max().item()) == 2.0
            assert int(kwargs["is_covers"][0].item()) == 1
            lm_hints = kwargs["precomputed_lm_hints_25Hz"]
            assert isinstance(lm_hints, torch.Tensor)
            assert tuple(lm_hints.shape) == (1, 250, 64)
            assert torch.allclose(src, lm_hints)
            assert kwargs.get("audio_codes") is None
            caption_prompts = [text for text in seen["texts"] if text.startswith("# Instruction\n")]
            assert caption_prompts
            assert any("# Metas" in text for text in caption_prompts)
            assert any("Generate audio semantic tokens based on the given conditions:" in text for text in caption_prompts)
            assert any("Fill the audio semantic mask based on the given conditions:" in text for text in caption_prompts)
            assert kwargs["audio_cover_strength"] == pytest.approx(0.5)
            assert kwargs["non_cover_text_hidden_states"] is not None
            assert kwargs["non_cover_text_attention_mask"] is not None
            t = int(src.shape[1])
            return {"target_latents": torch.zeros((1, t, 64), dtype=torch.float32), "time_costs": {}}

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

    backend = AceStepV15Backend(
        config=AceStepV15BackendConfig(
            repo_id="ACE-Step/Ace-Step1.5",
            device="cpu",
            use_audio_code_planner=True,
        )
    )
    monkeypatch.setattr(
        backend,
        "_maybe_plan_audio_codes",
        lambda **_kwargs: {
            "caption": "planned caption",
            "vocal_language": "unknown",
            "bpm": 88,
            "keyscale": "C minor",
            "timesignature": "4",
            "cot_text": "<think>\ncaption: planned caption\n</think>",
            "audio_codes": torch.arange(50, dtype=torch.long).unsqueeze(0).unsqueeze(-1),
            "audio_code_count": 50,
            "lm_model": "fake-lm",
        },
    )

    asset = backend.generate_audio(AudioGenerationRequest(prompt="sci fi music", duration_s=10.0))

    assert asset.mime_type == "audio/wav"
    assert bytes(asset.data)[:4] == b"RIFF"
    assert asset.metadata["planner_mode"] == "lm_audio_codes"
    assert asset.metadata["planner_audio_code_count"] == 50


@pytest.mark.unit
def test_acestep_backend_segments_long_mps_requests_before_model_load(monkeypatch):
    import torch

    from abstractmusic.backends.acestep_v15 import AceStepV15Backend, AceStepV15BackendConfig
    from abstractmusic.types import AudioGenerationRequest, GeneratedAsset

    backend = AceStepV15Backend(config=AceStepV15BackendConfig(device="mps", use_audio_code_planner=True))

    planner = {
        "caption": "planned caption",
        "vocal_language": "unknown",
        "audio_codes": torch.arange(300, dtype=torch.long).unsqueeze(0).unsqueeze(-1),
        "audio_code_count": 300,
    }
    seen = {"segmented": None}

    monkeypatch.setattr(backend, "_maybe_plan_audio_codes", lambda **_kwargs: planner)
    monkeypatch.setattr(backend, "_release_lm_runtime", lambda: None)
    monkeypatch.setattr(
        backend,
        "_ensure_loaded",
        lambda: (_ for _ in ()).throw(AssertionError("_ensure_loaded should not run before segmented dispatch")),
    )

    def _fake_segmented(*, request, planner, extra):
        seen["segmented"] = {
            "duration_s": float(request.duration_s),
            "planner_audio_code_count": int(planner["audio_codes"].shape[1]),
            "extra": dict(extra),
        }
        return GeneratedAsset(
            data=b"RIFFfake",
            mime_type="audio/wav",
            metadata={"segmented_long_generation": True},
        )

    monkeypatch.setattr(backend, "_generate_segmented_long_audio", _fake_segmented)

    asset = backend.generate_audio(AudioGenerationRequest(prompt="ambient", duration_s=60.0))

    assert asset.mime_type == "audio/wav"
    assert asset.metadata["segmented_long_generation"] is True
    assert seen["segmented"] == {
        "duration_s": 60.0,
        "planner_audio_code_count": 300,
        "extra": {"_acestep_user_seed_provided": False},
    }


@pytest.mark.unit
def test_acestep_backend_stitches_segmented_long_audio(monkeypatch):
    import torch

    from abstractmusic.backends.acestep_v15 import (
        AceStepV15Backend,
        AceStepV15BackendConfig,
        _decode_wav_bytes,
        _encode_wav_bytes,
    )
    from abstractmusic.types import AudioGenerationRequest, GeneratedAsset

    backend = AceStepV15Backend(config=AceStepV15BackendConfig(device="mps"))

    planner = {
        "caption": "planned caption",
        "vocal_language": "unknown",
        "audio_codes": torch.arange(300, dtype=torch.long).unsqueeze(0).unsqueeze(-1),
        "audio_code_count": 300,
    }

    seen_segment_requests = []

    def _fake_recursive_generate(self, segment_request):
        segment_index = len(seen_segment_requests)
        seen_segment_requests.append(segment_request)

        segment_extra = dict(segment_request.extra or {})
        assert segment_extra["_acestep_segmented_run"] is True
        segment_planner = segment_extra["_acestep_internal_planner"]
        segment_codes = segment_planner["audio_codes"]
        assert tuple(segment_codes.shape) == (1, 50, 1)
        assert int(segment_codes[0, 0, 0].item()) == segment_index * 50
        assert int(segment_codes[0, -1, 0].item()) == (segment_index + 1) * 50 - 1

        sample_rate = 10
        samples = int(round(float(segment_request.duration_s) * sample_rate))
        wav = torch.full((2, samples), 0.1 * float(segment_index + 1), dtype=torch.float32)
        return GeneratedAsset(
            data=_encode_wav_bytes(wav, sample_rate=sample_rate),
            mime_type="audio/wav",
            metadata={"segment_index": segment_index},
        )

    monkeypatch.setattr(backend, "generate_audio", types.MethodType(_fake_recursive_generate, backend))

    asset = backend._generate_segmented_long_audio(
        request=AudioGenerationRequest(prompt="ambient", duration_s=60.0, seed=7),
        planner=planner,
        extra={},
    )

    audio, sample_rate = _decode_wav_bytes(asset.data)

    assert sample_rate == 10
    assert tuple(audio.shape) == (2, 600)
    assert len(seen_segment_requests) == 6
    assert [int(req.seed) for req in seen_segment_requests] == [7, 8, 9, 10, 11, 12]
    assert asset.metadata["segmented_long_generation"] is True
    assert asset.metadata["segment_count"] == 6
    assert asset.metadata["segment_duration_s"] == pytest.approx(10.0)
    assert asset.metadata["segment_crossfade_s"] == pytest.approx(0.0)
    assert asset.metadata["duration_s"] == pytest.approx(60.0)


@pytest.mark.unit
def test_acestep_backend_uses_local_files_only_for_model_loads(monkeypatch, tmp_path):
    import torch

    from abstractmusic.backends.acestep_v15 import AceStepV15Backend, AceStepV15BackendConfig

    silence_path = tmp_path / "silence_latent.pt"
    torch.save(torch.zeros((1, 64, 1000), dtype=torch.float32), silence_path)

    seen = {"dit": None, "text_tok": None, "text_model": None, "vae": None, "hf": None}

    class _FakeModel:
        def to(self, *_a, **_k):
            return self

        def eval(self):
            return self

    class _FakeTextEncoder(_FakeModel):
        def __init__(self):
            self._emb = torch.nn.Embedding(16, 8)

        def get_input_embeddings(self):
            return self._emb

        def forward(self, input_ids=None, attention_mask=None, lyric_attention_mask=None):
            _ = attention_mask, lyric_attention_mask
            return types.SimpleNamespace(last_hidden_state=self._emb(input_ids))

        __call__ = forward

    class _FakeTokenizer:
        model_max_length = 1024
        pad_token_id = 0
        eos_token_id = 0

        def __call__(self, *_a, **_k):
            return types.SimpleNamespace(
                input_ids=torch.tensor([[1, 2]], dtype=torch.long),
                attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
            )

    class _FakeVAE:
        dtype = torch.float32

        def to(self, *_a, **_k):
            return self

        def eval(self):
            return self

    def _fake_hf_hub_download(**kwargs):
        seen["hf"] = dict(kwargs)
        return str(silence_path)

    monkeypatch.setattr(
        "abstractmusic.backends.acestep_v15._lazy_import_hf_hub_download",
        lambda: _fake_hf_hub_download,
    )

    class _FakeAceStepConditionGenerationModel:
        @staticmethod
        def from_pretrained(*_a, **kwargs):
            seen["dit"] = dict(kwargs)
            return _FakeModel()

    monkeypatch.setattr(
        "abstractmusic.backends.acestep_v15._lazy_import_acestep_v15_turbo_model",
        lambda: _FakeAceStepConditionGenerationModel,
    )

    def _lazy_transformers():
        class _AutoModel:
            @staticmethod
            def from_pretrained(*_a, **kwargs):
                seen["text_model"] = dict(kwargs)
                return _FakeTextEncoder()

        class _AutoTokenizer:
            @staticmethod
            def from_pretrained(*_a, **kwargs):
                seen["text_tok"] = dict(kwargs)
                return _FakeTokenizer()

        return _AutoModel, _AutoTokenizer

    monkeypatch.setattr("abstractmusic.backends.acestep_v15._lazy_import_transformers", _lazy_transformers)

    def _lazy_oobleck():
        class _AutoencoderOobleck:
            @staticmethod
            def from_pretrained(*_a, **kwargs):
                seen["vae"] = dict(kwargs)
                return _FakeVAE()

        return _AutoencoderOobleck

    monkeypatch.setattr("abstractmusic.backends.acestep_v15._lazy_import_diffusers_oobleck", _lazy_oobleck)

    backend = AceStepV15Backend(config=AceStepV15BackendConfig(repo_id="ACE-Step/Ace-Step1.5", device="cpu"))
    backend._ensure_loaded()

    assert seen["dit"]["local_files_only"] is True
    assert seen["text_tok"]["local_files_only"] is True
    assert seen["text_model"]["local_files_only"] is True
    assert seen["vae"]["local_files_only"] is True
    assert seen["hf"]["local_files_only"] is True


@pytest.mark.unit
def test_acestep_backend_cfg_audio_code_generation(monkeypatch):
    import torch

    from abstractmusic.backends.acestep_v15 import AceStepV15Backend, AceStepV15BackendConfig

    class _FakeBatch:
        def __init__(self, input_ids):
            self.input_ids = input_ids
            self.attention_mask = torch.ones_like(input_ids)

    class _FakeTokenizer:
        eos_token_id = 2
        pad_token_id = 2
        padding_side = "right"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            _ = tokenize, add_generation_prompt
            return "\n".join(str(item["content"]) for item in messages)

        def __call__(self, text, return_tensors="pt", padding=False, truncation=False):
            _ = return_tensors, padding, truncation
            if isinstance(text, list):
                return _FakeBatch(torch.tensor([[11, 12], [21, 22]], dtype=torch.long))
            return _FakeBatch(torch.tensor([[11, 12]], dtype=torch.long))

        def decode(self, token_ids, skip_special_tokens=False):
            _ = skip_special_tokens
            if hasattr(token_ids, "tolist"):
                token_ids = token_ids.tolist()
            mapping = {
                100: "<|audio_code_10|>",
                101: "<|audio_code_20|>",
                2: "</s>",
            }
            return "".join(mapping.get(int(tok), "") for tok in token_ids)

        def encode(self, text, add_special_tokens=False):
            _ = add_special_tokens
            if text == "</think>":
                return [9]
            return [1]

        def get_added_vocab(self):
            return {
                "<|audio_code_10|>": 100,
                "<|audio_code_20|>": 101,
            }

    class _FakeLM:
        def __init__(self):
            self.call_count = 0

        def to(self, *_a, **_k):
            return self

        def eval(self):
            return self

        def __call__(self, input_ids=None, attention_mask=None, past_key_values=None, use_cache=True):
            _ = input_ids, attention_mask, past_key_values, use_cache
            self.call_count += 1
            logits = torch.full((2, 1, 256), -1000.0, dtype=torch.float32)
            if self.call_count == 1:
                logits[:, :, 100] = 5.0
                logits[0, :, 100] = 6.0
            elif self.call_count == 2:
                logits[:, :, 101] = 5.0
                logits[0, :, 101] = 6.0
            else:
                logits[:, :, 2] = 5.0
            return types.SimpleNamespace(logits=logits, past_key_values=None)

    backend = AceStepV15Backend(
        config=AceStepV15BackendConfig(
            device="cpu",
            lm_temperature=0.0,
            lm_cfg_scale=2.0,
            lm_top_k=0,
            lm_top_p=1.0,
        )
    )
    backend._lm_tokenizer = _FakeTokenizer()
    backend._lm_model = _FakeLM()
    backend._lm_device = "cpu"
    backend._lm_dtype = torch.float32
    backend._lm_model_label = "fake"
    backend._lm_audio_token_ids = (100, 101)
    backend._lm_think_end_token_id = 9
    monkeypatch.setattr(backend, "_ensure_lm_loaded", lambda: None)

    codes = backend._generate_audio_code_ids(
        caption="ambient lo-fi study music",
        lyrics="",
        cot_text="<think>\nbpm: 80\ncaption: ambient lo-fi study music\nduration: 10\n</think>",
        target_code_count=2,
    )

    assert tuple(codes.shape) == (1, 2, 1)
    assert codes.squeeze(-1).tolist() == [[10, 20]]


@pytest.mark.unit
def test_acestep_backend_prefers_primary_lm_on_cpu():
    from abstractmusic.backends.acestep_v15 import AceStepV15Backend, AceStepV15BackendConfig

    backend = AceStepV15Backend(config=AceStepV15BackendConfig())
    candidates = backend._lm_candidate_sources("cpu")

    assert candidates[0][2] == "acestep-5Hz-lm-1.7B"


@pytest.mark.unit
def test_acestep_backend_retries_seedless_bad_planner_outputs(monkeypatch, tmp_path):
    from dataclasses import dataclass

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

        def __call__(self, text, return_tensors="pt", padding=False, truncation=False, max_length=None):
            _ = text, return_tensors, padding, truncation, max_length
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
            return _EncOut(self._emb(input_ids))

        __call__ = forward

        def get_input_embeddings(self):
            return self._emb

    seen = {"seeds": []}

    class _FakeModel:
        def __init__(self):
            class _FakeQuantizer:
                @staticmethod
                def get_output_from_indices(indices):
                    return indices.to(dtype=torch.float32).repeat(1, 1, 64)

            class _FakeTokenizerState:
                quantizer = _FakeQuantizer()

            class _FakeDetokenizer:
                def __call__(self, quantized):
                    return quantized

            self.tokenizer = _FakeTokenizerState()
            self.detokenizer = _FakeDetokenizer()

        def to(self, *_a, **_k):
            return self

        def eval(self):
            return self

        def generate_audio(self, **kwargs):
            seen["seeds"].append(int(kwargs["seed"]))
            src = kwargs["src_latents"]
            t = int(src.shape[1])
            return {"target_latents": torch.zeros((1, t, 64), dtype=torch.float32), "time_costs": {}}

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
    monkeypatch.setattr("abstractmusic.backends.acestep_v15.random.randint", lambda *_a, **_k: 7)

    quality_states = iter([False, True])

    @dataclass(frozen=True)
    class _FakeStats:
        ok: bool

        @property
        def is_probably_music_like(self):
            return self.ok

        @property
        def is_probably_repetitive_noise(self):
            return False

    monkeypatch.setattr(
        "abstractmusic.backends.acestep_v15.inspect_music_signal_bytes",
        lambda _data: _FakeStats(ok=next(quality_states)),
    )

    backend = AceStepV15Backend(
        config=AceStepV15BackendConfig(
            repo_id="ACE-Step/Ace-Step1.5",
            device="cpu",
            quality_retry_enabled=True,
            quality_retry_max_attempts=2,
            quality_retry_fallback_seeds=(123, 124),
        )
    )
    monkeypatch.setattr(
        backend,
        "_maybe_plan_audio_codes",
        lambda **_kwargs: {
            "caption": "planned caption",
            "vocal_language": "unknown",
            "bpm": 88,
            "keyscale": "C minor",
            "timesignature": "4",
            "cot_text": "<think>\ncaption: planned caption\n</think>",
            "audio_codes": torch.arange(50, dtype=torch.long).unsqueeze(0).unsqueeze(-1),
            "audio_code_count": 50,
            "lm_model": "fake-lm",
        },
    )

    asset = backend.generate_audio(AudioGenerationRequest(prompt="ambient", duration_s=10.0))

    assert asset.mime_type == "audio/wav"
    assert seen["seeds"] == [7, 123]
    assert asset.metadata["quality_retry_count"] == 1
    assert asset.metadata["quality_gate_passed"] is True
