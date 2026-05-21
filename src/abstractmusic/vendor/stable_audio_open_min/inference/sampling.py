"""Minimal Stable Audio Open sampling utilities.

The upstream `stable-audio-tools` package wires multiple samplers and depends
on `k-diffusion` for the v-objective path. AbstractMusic only needs a compact
schedule shift helper for the rectified-flow checkpoints it supports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DistributionShift:
    base_shift: float = 0.5
    max_shift: float = 1.15
    max_length: int = 4096
    min_length: int = 256
    use_sine: bool = False

    def time_shift(self, t: Any, seq_len: int) -> Any:
        import math

        sigma = 1.0
        denom = max(float(self.max_length - self.min_length), 1.0)
        mu = -(self.base_shift + (self.max_shift - self.base_shift) * (float(seq_len) - self.min_length) / denom)
        t_out = 1 - math.exp(mu) / (math.exp(mu) + (1 / (1 - t) - 1) ** sigma)
        if self.use_sine:
            t_out = (t_out * (math.pi / 2)).sin() if hasattr(t_out, "sin") else t_out
        return t_out


def sample(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError(
        "Stable Audio Open v-objective sampling is not available in AbstractMusic's vendored minimal runtime. "
        "Only rectified-flow Stable Audio Open checkpoints are supported."
    )


__all__ = ["DistributionShift", "sample"]
