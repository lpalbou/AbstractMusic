"""Small native-Haar DCW helper for ACE-Step turbo sampling.

This implements the one-level temporal correction used by ACE-Step's sampler
without depending on the external ACE-Step package or pytorch_wavelets.
Latents are shaped [batch, time, channels] at 25 Hz.
"""

from __future__ import annotations

import math

import torch


def _haar_pair(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    original_t = int(x.shape[1])
    if original_t % 2:
        x = torch.cat([x, torch.zeros_like(x[:, :1, :])], dim=1)
    scale = 1.0 / math.sqrt(2.0)
    even = x[:, 0::2, :]
    odd = x[:, 1::2, :]
    low = (even + odd) * scale
    high = (even - odd) * scale
    return low, high, original_t


def _haar_inverse(low: torch.Tensor, high: torch.Tensor, original_t: int) -> torch.Tensor:
    scale = 1.0 / math.sqrt(2.0)
    even = (low + high) * scale
    odd = (low - high) * scale
    out = torch.empty(
        (low.shape[0], low.shape[1] * 2, low.shape[2]),
        device=low.device,
        dtype=low.dtype,
    )
    out[:, 0::2, :] = even
    out[:, 1::2, :] = odd
    return out[:, :original_t, :]


def dcw_pix(x: torch.Tensor, y: torch.Tensor, scaler: float) -> torch.Tensor:
    if scaler == 0.0:
        return x
    return x + float(scaler) * (x - y)


def dcw_low(x: torch.Tensor, y: torch.Tensor, scaler: float) -> torch.Tensor:
    if scaler == 0.0:
        return x
    xl, xh, out_t = _haar_pair(x)
    yl, _yh, _ = _haar_pair(y)
    xl = xl + float(scaler) * (xl - yl)
    return _haar_inverse(xl, xh, out_t).to(dtype=x.dtype)


def dcw_high(x: torch.Tensor, y: torch.Tensor, scaler: float) -> torch.Tensor:
    if scaler == 0.0:
        return x
    xl, xh, out_t = _haar_pair(x)
    _yl, yh, _ = _haar_pair(y)
    xh = xh + float(scaler) * (xh - yh)
    return _haar_inverse(xl, xh, out_t).to(dtype=x.dtype)


def dcw_double(x: torch.Tensor, y: torch.Tensor, low_scaler: float, high_scaler: float) -> torch.Tensor:
    if low_scaler == 0.0 and high_scaler == 0.0:
        return x
    xl, xh, out_t = _haar_pair(x)
    yl, yh, _ = _haar_pair(y)
    if low_scaler != 0.0:
        xl = xl + float(low_scaler) * (xl - yl)
    if high_scaler != 0.0:
        xh = xh + float(high_scaler) * (xh - yh)
    return _haar_inverse(xl, xh, out_t).to(dtype=x.dtype)


class DCWCorrector:
    """Applies per-step differential correction in the temporal Haar domain."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        mode: str = "double",
        scaler: float = 0.05,
        high_scaler: float = 0.02,
    ) -> None:
        if mode not in {"low", "high", "double", "pix"}:
            raise ValueError("dcw_mode must be one of: low, high, double, pix")
        self.enabled = bool(enabled)
        self.mode = str(mode)
        self.scaler = float(scaler)
        self.high_scaler = float(high_scaler)

    @property
    def is_active(self) -> bool:
        if not self.enabled:
            return False
        if self.mode == "double":
            return self.scaler != 0.0 or self.high_scaler != 0.0
        return self.scaler != 0.0

    def apply(self, x_next: torch.Tensor, denoised: torch.Tensor, t_curr: float) -> torch.Tensor:
        if not self.is_active:
            return x_next
        t = float(t_curr)
        low_s = t * self.scaler
        high_s = (1.0 - t) * self.scaler
        double_high_s = (1.0 - t) * self.high_scaler
        if self.mode == "low":
            return dcw_low(x_next, denoised, low_s)
        if self.mode == "high":
            return dcw_high(x_next, denoised, high_s)
        if self.mode == "double":
            return dcw_double(x_next, denoised, low_s, double_high_s)
        return dcw_pix(x_next, denoised, self.scaler)
