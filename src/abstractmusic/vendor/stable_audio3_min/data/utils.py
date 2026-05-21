import math
from typing import Optional

import torch


class PadCrop:
    def __init__(self, n_samples, randomize=True):
        self.n_samples = int(n_samples)
        self.randomize = bool(randomize)

    def __call__(self, signal):
        n, s = signal.shape
        start = 0
        if self.randomize:
            start = int(torch.randint(0, max(0, s - self.n_samples) + 1, []).item())
        end = start + self.n_samples
        output = signal.new_zeros([n, self.n_samples])
        output[:, : min(s, self.n_samples)] = signal[:, start:end]
        return output


def create_padding_mask_from_lengths(valid_lengths: torch.Tensor, total_seq_len: int) -> torch.Tensor:
    device = valid_lengths.device
    positions = torch.arange(total_seq_len, device=device).unsqueeze(0)
    return positions < valid_lengths.unsqueeze(1)


def compute_effective_seq_len_from_conditioning(
    conditioning: list,
    sample_rate: int,
    downsampling_ratio: int = 1,
    device: str = "cuda",
) -> Optional[torch.Tensor]:
    if conditioning is None or not any("seconds_total" in c for c in conditioning):
        return None

    lengths = []
    for cond_dict in conditioning:
        if "seconds_total" not in cond_dict:
            return None
        audio_samples = int(float(cond_dict["seconds_total"]) * int(sample_rate))
        lengths.append(math.ceil(audio_samples / max(1, int(downsampling_ratio))))
    return torch.tensor(lengths, dtype=torch.float32, device=device)
