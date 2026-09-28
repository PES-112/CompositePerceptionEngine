"""
model.py — Architecture 1: plain per-object GRU
================================================
Each object is scored from its own history only. Objects in the same frame
never see each other, so this is the no-scene-context baseline that
GRU_attention/ is measured against.
"""

from __future__ import annotations

import torch
from torch import nn

from models.GRU.layers import ThreatHead, TrackEncoder


class TrackGRU(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden: int = 64,
                 dropout: float = 0.1, gru_layers: int = 1):
        super().__init__()
        self.encoder = TrackEncoder(n_features, n_classes, hidden, layers=gru_layers, dropout=dropout)
        self.head = ThreatHead(hidden, dropout)

    def forward(self, x: torch.Tensor, cls: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x [B, N, T, F], cls [B, N], mask [B, N] -> encounter logits [B, N].
        `mask` is accepted only so both architectures share one call signature;
        objects are scored independently, so padding cannot affect real objects.
        """
        return self.head(self.encoder(x, cls))
