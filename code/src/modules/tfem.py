"""Temporal Feature Engineering Module (TFEM).

Implements the LSTM encoder side of TFEM. The non-lagged statistics and
event markers are computed offline (see ``AgriNet_sliding_window_7d.csv``
and ``src/datasets/agrinet.py``); this module simply embeds the
(window_days, n_features) sequence into a hidden vector consumed by the
MHSA fusion block of the main model.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TFEMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, T, F)  ->  (B, hidden_dim)"""
        out, (h_n, _) = self.lstm(seq)
        last = h_n[-1]
        return self.norm(last)
