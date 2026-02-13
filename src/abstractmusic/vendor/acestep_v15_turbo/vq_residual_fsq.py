"""
AbstractMusic vendored quantization utilities (MIT) for ACE-Step v1.5.

This module vendors (with minimal modifications) the MIT-licensed implementations of:
- `FSQ` (Finite Scalar Quantization)
- `ResidualFSQ`

from `vector-quantize-pytorch` (https://github.com/lucidrains/vector-quantizer-pytorch).

Why we vendor:
- Avoid adding `vector-quantize-pytorch` (and its transitive deps like `einx`) as a hard runtime dependency.
- Keep the ACE-Step backend self-contained under permissive licensing.

Minimal modifications vs upstream:
- Remove the dependency on `einx` by replacing `get_at(...)` with a small `torch.gather` helper.
"""

# MIT License
#
# Copyright (c) 2020 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import random
from contextlib import nullcontext
from functools import partial, wraps
from math import ceil
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange, reduce, repeat, unpack
from torch import Tensor, atanh, clamp, int32, tanh, tensor
from torch.amp import autocast
from torch.nn import Module


# ---------------------------------------------------------------------------
# Helper utilities (vendored from vector-quantize-pytorch)
# ---------------------------------------------------------------------------


def _exists(v):
    return v is not None


def _default(*args):
    for arg in args:
        if _exists(arg):
            return arg
    return None


def _identity(t):
    return t


def _maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not _exists(x):
            return x
        return fn(x, *args, **kwargs)

    return inner


def _pack_one(t, pattern):
    return pack([t], pattern)


def _unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def _round_ste(z):
    """Round with straight-through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


def _floor_ste(z):
    """Floor with straight-through gradients."""
    zhat = z.floor()
    return z + (zhat - z).detach()


# ---------------------------------------------------------------------------
# FSQ — Finite Scalar Quantization (MIT, vendored)
# ---------------------------------------------------------------------------


class FSQ(Module):
    def __init__(
        self,
        levels: list[int] | tuple[int, ...],
        dim: int | None = None,
        num_codebooks: int = 1,
        keep_num_codebooks_dim: bool | None = None,
        scale: float | None = None,
        allowed_dtypes: tuple[torch.dtype, ...] = (torch.float32, torch.float64),
        channel_first: bool = False,
        projection_has_bias: bool = True,
        return_indices: bool = True,
        force_quantization_f32: bool = True,
        preserve_symmetry: bool = False,
        noise_dropout: float = 0.0,
        bound_hard_clamp: bool = False,  # for residual fsq, if input is pre-softclamped to the right range
    ):
        super().__init__()

        assert not (
            any([l == 2 for l in levels]) and not preserve_symmetry
        ), "turn on `preserve_symmetry` for using any levels == 2, or use a greater level"

        if isinstance(levels, tuple):
            levels = list(levels)

        _levels = tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent=False)

        self.scale = scale
        self.preserve_symmetry = preserve_symmetry
        self.noise_dropout = noise_dropout

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = _default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = _default(dim, len(_levels) * num_codebooks)
        self.channel_first = channel_first

        has_projections = self.dim != effective_codebook_dim
        self.project_in = (
            nn.Linear(self.dim, effective_codebook_dim, bias=projection_has_bias)
            if has_projections
            else nn.Identity()
        )
        self.project_out = (
            nn.Linear(effective_codebook_dim, self.dim, bias=projection_has_bias)
            if has_projections
            else nn.Identity()
        )

        self.has_projections = has_projections
        self.return_indices = return_indices

        if return_indices:
            # NOTE: avoid .item() on tensors that could be on meta; compute in Python.
            codebook_size = 1
            for l in levels:
                codebook_size *= int(l)
            self.codebook_size = int(codebook_size)

            implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size))
            self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

        self.allowed_dtypes = allowed_dtypes
        self.force_quantization_f32 = force_quantization_f32
        self.bound_hard_clamp = bound_hard_clamp

    def bound(self, z, eps: float = 1e-3, hard_clamp: bool = False):
        """Bound `z`, an array of shape (..., d)."""
        maybe_tanh = tanh if not hard_clamp else partial(clamp, min=-1.0, max=1.0)
        maybe_atanh = atanh if not hard_clamp else _identity

        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = maybe_atanh(offset / half_l)
        bounded_z = maybe_tanh(z + shift) * half_l - offset
        half_width = self._levels // 2
        return _round_ste(bounded_z) / half_width

    # symmetry-preserving and noise-approximated quantization, section 3.2 in https://arxiv.org/abs/2411.19842
    def symmetry_preserving_bound(self, z, hard_clamp: bool = False):
        """QL(x) = 2 / (L - 1) * [(L - 1) * (tanh(x) + 1) / 2 + 0.5] - 1"""
        maybe_tanh = tanh if not hard_clamp else partial(clamp, min=-1.0, max=1.0)

        levels_minus_1 = self._levels - 1
        scale = 2.0 / levels_minus_1
        bracket = (levels_minus_1 * (maybe_tanh(z) + 1) / 2.0) + 0.5
        bracket = _floor_ste(bracket)
        return scale * bracket - 1.0

    def quantize(self, z):
        """Quantizes z, returns quantized zhat, same shape as z."""
        shape, device, noise_dropout, preserve_symmetry = (
            z.shape[0],
            z.device,
            self.noise_dropout,
            self.preserve_symmetry,
        )
        bound_fn = self.symmetry_preserving_bound if preserve_symmetry else self.bound

        bounded_z = bound_fn(z, hard_clamp=self.bound_hard_clamp)

        # determine where to add a random offset elementwise if using noise dropout
        if not self.training or noise_dropout == 0.0:
            return bounded_z

        offset_mask = torch.bernoulli(torch.full_like(bounded_z, noise_dropout)).bool()
        offset = torch.rand_like(bounded_z) - 0.5
        bounded_z = torch.where(offset_mask, bounded_z + offset, bounded_z)

        return bounded_z

    def _scale_and_shift(self, zhat_normalized):
        if self.preserve_symmetry:
            return (zhat_normalized + 1.0) / (2.0 / (self._levels - 1))

        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat):
        if self.preserve_symmetry:
            return zhat * (2.0 / (self._levels - 1)) - 1.0

        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def indices_to_level_indices(self, indices):
        """Converts indices to indices at each level."""
        indices = rearrange(indices, "... -> ... 1")
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def codes_to_indices(self, zhat):
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).round().to(int32)

    def indices_to_codes(self, indices):
        """Inverse of `codes_to_indices`."""
        assert _exists(indices)

        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))
        codes = self._indices_to_codes(indices)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, "... c d -> ... (c d)")

        codes = self.project_out(codes)

        if is_img_or_video or self.channel_first:
            codes = rearrange(codes, "b ... d -> b d ...")

        return codes

    def forward(self, z):
        """
        Einstein notation:
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """

        is_img_or_video = z.ndim >= 4
        need_move_channel_last = is_img_or_video or self.channel_first

        if need_move_channel_last:
            z = rearrange(z, "b d ... -> b ... d")

        z, ps = _pack_one(z, "b * d")

        z = self.project_in(z)

        # match dtype constraints
        z_dtype = z.dtype
        quant_context = nullcontext()
        if self.force_quantization_f32 and z_dtype not in self.allowed_dtypes:
            quant_context = autocast("cuda", enabled=False)
            z = z.float()

        with quant_context:
            zhat = self.quantize(z)

        if self.scale is not None:
            zhat = zhat * self.scale

        if self.return_indices:
            indices = self.codes_to_indices(zhat)
        else:
            indices = None

        zhat = self.project_out(zhat)

        zhat = _unpack_one(zhat, ps, "b * d")

        if need_move_channel_last:
            zhat = rearrange(zhat, "b ... d -> b d ...")

        if not self.return_indices:
            return zhat

        indices = _unpack_one(indices, ps, "b *")
        if is_img_or_video:
            # back to image/video indices
            indices = rearrange(indices, "b ... -> b ...")

        return zhat, indices


# ---------------------------------------------------------------------------
# ResidualFSQ (MIT, vendored) — minimal modifications to remove `einx`
# ---------------------------------------------------------------------------


def _round_up_multiple(num, mult):
    return ceil(num / mult) * mult


def _gather_codebooks(codebooks: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather code vectors for each quantizer.

    Args:
        codebooks: [Q, C, D]
        indices: [B, N, Q] (long)

    Returns:
        gathered: [Q, B, N, D]
    """

    q, c, d = codebooks.shape
    b, n, q2 = indices.shape
    assert q2 == q

    idx = indices.permute(2, 0, 1)  # [Q, B, N]
    codebooks_exp = codebooks.unsqueeze(1).expand(q, b, c, d)  # [Q, B, C, D]
    idx_exp = idx.unsqueeze(-1).expand(q, b, n, d)  # [Q, B, N, D]
    return torch.gather(codebooks_exp, dim=2, index=idx_exp)  # [Q, B, N, D]


class ResidualFSQ(Module):
    """Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf"""

    def __init__(
        self,
        *,
        levels: list[int],
        num_quantizers,
        dim=None,
        is_channel_first: bool = False,
        quantize_dropout: bool = False,
        quantize_dropout_cutoff_index: int = 0,
        quantize_dropout_multiple_of: int = 1,
        soft_clamp_input_value: float | list[float] | Tensor | None = None,
        bound_hard_clamp: bool = True,
        **kwargs,
    ):
        super().__init__()

        codebook_dim = len(levels)
        dim = _default(dim, codebook_dim)

        requires_projection = codebook_dim != dim
        self.project_in = nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        self.has_projections = requires_projection

        self.is_channel_first = is_channel_first
        self.num_quantizers = num_quantizers

        self.levels = levels
        self.layers = nn.ModuleList([])

        levels_tensor = tensor(levels)
        assert (levels_tensor > 1).all()

        scales = []
        for ind in range(num_quantizers):
            scales.append(levels_tensor.float() ** -ind)

            fsq = FSQ(
                levels=levels,
                dim=codebook_dim,
                preserve_symmetry=True,
                bound_hard_clamp=bound_hard_clamp,
                **kwargs,
            )

            self.layers.append(fsq)

        assert all([not fsq.has_projections for fsq in self.layers])

        self.codebook_size = self.layers[0].codebook_size
        self.register_buffer("scales", torch.stack(scales), persistent=False)

        self.quantize_dropout = quantize_dropout and num_quantizers > 1
        assert quantize_dropout_cutoff_index >= 0

        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_multiple_of = quantize_dropout_multiple_of

        # soft clamping the input value
        if bound_hard_clamp:
            assert not _exists(soft_clamp_input_value)
            soft_clamp_input_value = 1 + (1 / (levels_tensor - 1))

        if isinstance(soft_clamp_input_value, (list, float)):
            soft_clamp_input_value = tensor(soft_clamp_input_value)

        self.register_buffer("soft_clamp_input_value", soft_clamp_input_value, persistent=False)

    @property
    def codebooks(self):
        codebooks = [layer.implicit_codebook for layer in self.layers]
        codebooks = torch.stack(codebooks, dim=0)
        return codebooks

    def get_codes_from_indices(self, indices):
        batch, quantize_dim = indices.shape[0], indices.shape[-1]

        # may also receive indices in the shape of 'b h w q' (accept_image_fmap)
        indices, ps = pack([indices], "b * q")

        # because of quantize dropout, one can pass in indices that are coarse
        # and the network should be able to reconstruct
        if quantize_dim < self.num_quantizers:
            assert (
                self.quantize_dropout > 0.0
            ), "quantize dropout must be greater than 0 if you wish to reconstruct from a signal with less fine quantizations"
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value=-1)

        # take care of quantizer dropout
        mask = indices == -1
        indices = indices.masked_fill(mask, 0)

        # gathered: [Q, B, N, D]
        all_codes = _gather_codebooks(self.codebooks, indices)

        # mask out any codes that were dropout-ed
        all_codes = all_codes.masked_fill(rearrange(mask, "b n q -> q b n 1"), 0.0)

        # scale the codes
        scales = rearrange(self.scales, "q d -> q 1 1 d")
        all_codes = all_codes * scales

        all_codes, = unpack(all_codes, ps, "q b * d")
        return all_codes

    def get_output_from_indices(self, indices):
        codes = self.get_codes_from_indices(indices)
        codes_summed = reduce(codes, "q ... -> ...", "sum")
        return self.project_out(codes_summed)

    def forward(self, x, return_all_codes: bool = False, rand_quantize_dropout_fixed_seed=None):
        num_quant, quant_dropout_multiple_of, device = (
            self.num_quantizers,
            self.quantize_dropout_multiple_of,
            x.device,
        )

        # handle channel first
        if self.is_channel_first:
            x = rearrange(x, "b d ... -> b ... d")
            x, ps = pack([x], "b * d")

        # maybe project in
        x = self.project_in(x)

        # maybe softclamp input before residual layers
        if _exists(self.soft_clamp_input_value):
            clamp_value = self.soft_clamp_input_value
            x = (x / clamp_value).tanh() * clamp_value

        quantized_out = 0.0
        residual = x
        all_indices = []

        should_quantize_dropout = self.training and self.quantize_dropout and torch.is_grad_enabled()

        # sample a layer index at which to dropout further residual quantization
        if should_quantize_dropout:
            if not _exists(rand_quantize_dropout_fixed_seed):
                rand_quantize_dropout_fixed_seed = torch.randint(0, 10_000, (), device=device).item()

            rand = random.Random(rand_quantize_dropout_fixed_seed)
            rand_quantize_dropout_index = rand.randrange(self.quantize_dropout_cutoff_index, num_quant)

            if quant_dropout_multiple_of != 1:
                rand_quantize_dropout_index = (
                    _round_up_multiple(rand_quantize_dropout_index + 1, quant_dropout_multiple_of) - 1
                )

            null_indices = torch.full(x.shape[:2], -1.0, device=device, dtype=torch.long)

        # go through the layers
        with autocast("cuda", enabled=False):
            for quantizer_index, (layer, scale) in enumerate(zip(self.layers, self.scales)):
                if should_quantize_dropout and quantizer_index > rand_quantize_dropout_index:
                    all_indices.append(null_indices)
                    continue

                quantized, indices = layer(residual / scale)
                quantized = quantized * scale

                residual = residual - quantized.detach()
                quantized_out = quantized_out + quantized
                all_indices.append(indices)

        # project out, if needed
        quantized_out = self.project_out(quantized_out)

        # stack all indices
        all_indices = torch.stack(all_indices, dim=-1)

        # channel first out
        if self.is_channel_first:
            quantized_out, = unpack(quantized_out, ps, "b * d")
            all_indices, = unpack(all_indices, ps, "b * d")

            quantized_out = rearrange(quantized_out, "b ... d -> b d ...")
            all_indices = rearrange(all_indices, "b ... d -> b d ...")

        ret = (quantized_out, all_indices)
        if not return_all_codes:
            return ret

        all_codes = self.get_codes_from_indices(all_indices)
        return (*ret, all_codes)

