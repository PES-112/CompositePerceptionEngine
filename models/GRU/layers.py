"""
layers.py
=========
Building blocks shared by both architectures. Keeping one copy of the encoder
and head means GRU/ and GRU_attention/ cannot drift apart: the attention layer
is the only thing the comparison varies.
"""

from __future__ import annotations

import torch
from torch import nn


class TrackEncoder(nn.Module):
    """One object's history [T, F] + its class -> one vector [hidden]."""

    def __init__(self, n_features: int, n_classes: int, hidden: int = 64,
                 class_dim: int = 8, layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.class_emb = nn.Embedding(n_classes, class_dim)
        self.inp = nn.Sequential(nn.Linear(n_features + class_dim, hidden), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)

    def forward(self, x: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
        """x [B, N, T, F], cls [B, N] -> [B, N, hidden]."""
        b, n, t, _ = x.shape
        c = self.class_emb(cls).unsqueeze(2).expand(b, n, t, -1)
        steps = self.inp(torch.cat([x, c], dim=-1)).reshape(b * n, t, -1)
        # The last step is always the current frame, where the object is present,
        # so the final hidden state is up to date without sequence packing.
        _, h = self.gru(steps)
        return h[-1].reshape(b, n, -1)


class ThreatHead(nn.Module):
    """
    [B, N, hidden] -> per object either one encounter logit [B, N] (n_outputs=1)
    or arrival-time logits [B, N, n_outputs]: one per slice of the horizon plus
    a final "not within the horizon" class.
    """

    def __init__(self, hidden: int = 64, dropout: float = 0.1, n_outputs: int = 1):
        super().__init__()
        self.n_outputs = n_outputs
        self.net = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, n_outputs))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out = self.net(h)
        return out.squeeze(-1) if self.n_outputs == 1 else out
