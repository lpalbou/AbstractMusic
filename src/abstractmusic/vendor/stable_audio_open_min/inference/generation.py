"""Minimal Stable Audio Open inference loop (rectified-flow only).

This module exists only to satisfy imports from the vendored model code. The
public StableAudio backend in AbstractMusic owns the supported generation path.
"""

from __future__ import annotations

from typing import Any, Optional

from .sampling import DistributionShift


def _time_shift(dist_shift: Any, t: Any, seq_len: int) -> Any:
    if dist_shift is None:
        return t
    time_shift = getattr(dist_shift, "time_shift", None)
    if callable(time_shift):
        return time_shift(t, seq_len)
    return t


def _sample_rf_pingpong(torch: Any, model_fn: Any, noise: Any, *, steps: int, dist_shift: Any, **extra_args: Any) -> Any:
    t = torch.linspace(1, 0, int(steps) + 1, device=noise.device, dtype=noise.dtype)
    t = _time_shift(dist_shift, t, int(noise.shape[-1]))
    ts = noise.new_ones([noise.shape[0]])
    x = noise
    for i in range(len(t) - 1):
        denoised = x - t[i] * model_fn(x, t[i] * ts, **extra_args)
        t_next = t[i + 1]
        x = (1 - t_next) * denoised + t_next * torch.randn_like(x)
    return x


def _sample_rf_euler(torch: Any, model_fn: Any, noise: Any, *, steps: int, dist_shift: Any, **extra_args: Any) -> Any:
    t = torch.linspace(1, 0, int(steps) + 1, device=noise.device, dtype=noise.dtype)
    t = _time_shift(dist_shift, t, int(noise.shape[-1]))
    x = noise
    for t_curr, t_prev in zip(t[:-1], t[1:]):
        t_curr_tensor = t_curr * torch.ones((x.shape[0],), dtype=x.dtype, device=x.device)
        dt = t_prev - t_curr
        v = model_fn(x, t_curr_tensor, **extra_args)
        x = x + dt * v
    return x


def generate_diffusion_cond(
    model: Any,
    *,
    steps: int = 8,
    cfg_scale: float = 1.0,
    conditioning: Any = None,
    conditioning_tensors: Optional[dict] = None,
    batch_size: int = 1,
    sample_size: int = 44100 * 11,
    seed: int = -1,
    device: str = "cpu",
    sampler_type: str = "pingpong",
    **sampler_kwargs: Any,
) -> Any:
    """Generate samples for RF Stable Audio models.

    This supports only the rectified-flow objective used by Stable Audio Open
    checkpoints. v-objective sampling is intentionally unsupported here.
    """

    import numpy as np  # type: ignore

    torch = __import__("torch")

    if conditioning_tensors is None:
        conditioning_tensors = model.conditioner(conditioning, device)
    conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)

    pretransform = getattr(model, "pretransform", None)
    latent_sample_size = int(sample_size)
    if pretransform is not None:
        latent_sample_size = latent_sample_size // int(getattr(pretransform, "downsampling_ratio", 1) or 1)

    seed_value = int(seed)
    if seed_value < 0:
        seed_value = int(np.random.randint(0, 2**31 - 1))
    torch.manual_seed(seed_value)
    if str(device) == "cuda":
        try:
            torch.cuda.manual_seed_all(seed_value)
        except Exception:
            pass

    noise = torch.randn([int(batch_size), int(getattr(model, "io_channels")), latent_sample_size], device=device)

    model_dtype = next(model.model.parameters()).dtype
    noise = noise.type(model_dtype)
    conditioning_inputs = {
        key: value.type(model_dtype) if value is not None and hasattr(value, "type") else value
        for key, value in conditioning_inputs.items()
    }

    diff_objective = str(getattr(model, "diffusion_objective", ""))
    if diff_objective not in {"rectified_flow", "rf_denoiser"}:
        raise ValueError(
            "Vendored Stable Audio Open minimal runtime supports rectified-flow checkpoints only. "
            f"Model diffusion_objective={diff_objective!r}."
        )

    dist_shift = getattr(model, "dist_shift", None)
    sampler_kwargs = {
        **conditioning_inputs,
        "dist_shift": dist_shift,
        "cfg_scale": float(cfg_scale),
        "batch_cfg": True,
        "rescale_cfg": True,
        **dict(sampler_kwargs or {}),
    }

    sampler = str(sampler_type or "pingpong").strip().lower()
    if sampler == "pingpong":
        sampled = _sample_rf_pingpong(torch, model.model, noise, steps=int(steps), **sampler_kwargs)
    elif sampler == "euler":
        sampled = _sample_rf_euler(torch, model.model, noise, steps=int(steps), **sampler_kwargs)
    else:
        raise ValueError("generate_diffusion_cond supports sampler_type='pingpong' or 'euler' in minimal mode.")

    if pretransform is not None:
        sampled = sampled.to(next(pretransform.parameters()).dtype)
        sampled = pretransform.decode(sampled)

    return sampled


__all__ = ["DistributionShift", "generate_diffusion_cond"]

