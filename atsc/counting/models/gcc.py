# Copyright 2024 Robert Bosch GmbH.
# SPDX-License-Identifier: GPL-3.0-only

"""Generalized cross-correlation with phase attention."""

import torch
from torch import nn


class GCCPhat(nn.Module):
    """Extract GCC-Phat from multi-channel STFT."""

    def __init__(self, max_coeff: int | None = None):
        """
        Initialize GCCPhat layer.

        Args:
            max_coeff: maximum number of coefficients, first max_coeff//2 and last max_coeff//2
        """
        super().__init__()
        self.max_coeff = max_coeff

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward step for GCCPhat layer.

        Args:
            x: STFT
                STFT: N,ch,T,F complex64 tensor
                    N: batch size
                    ch: number of channels
                    T: time frames
                    F = nfft/2+1

        Returns: N,comb,S,T float32 tensor
                N: number of signals in the batch
                comb: number of channels combinations
                S: max_coeff or nfft
                T: time frames
        """
        num_channels = x.shape[1]

        out_list = []
        for ch1 in range(num_channels - 1):
            for ch2 in range(ch1 + 1, num_channels):
                x1 = x[:, ch1]
                x2 = x[:, ch2]
                xcc = torch.angle(x1 * torch.conj(x2))
                xcc = torch.exp(1j * xcc)
                gcc_phat = torch.fft.irfft(xcc, dim=-2)
                if self.max_coeff is not None:
                    gcc_phat = torch.cat(
                        [
                            gcc_phat[:, -self.max_coeff // 2 :],
                            gcc_phat[:, : self.max_coeff // 2],
                        ],
                        dim=-2,
                    )
                out_list.append(gcc_phat)

        return torch.stack(out_list, dim=1)
