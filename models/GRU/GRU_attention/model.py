"""
model.py — Architecture 2: per-object GRU + attention across objects
=====================================================================
Same encoder and head as GRU/, plus one pre-norm transformer block that lets
every object in a frame attend to every other object before it is scored.

  - Residual connection: if other objects don't help, the block can learn to
    pass the encoder output straight through, so this model is not forced to
    do worse than the plain GRU.
  - No positional encoding across objects: fact sheets list objects sorted by
    K, so any notion of "position in the list" would leak K0's ranking.
    Without it, attention is permutation-equivariant — object order can't matter.
  - Context dropout (training only): randomly hides other objects from each
    other, so one flaky detection can't become something every prediction
    leans on. Every object stays a training target; only its context shrinks.
"""

from __future__ import annotations

import torch
from torch import nn

from models.GRU.layers import ThreatHead, TrackEncoder


class TrackGRUAttention(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden: int = 64,
                 dropout: float = 0.1, gru_layers: int = 1,
                 heads: int = 4, context_dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.context_dropout = context_dropout
        self.encoder = TrackEncoder(n_features, n_classes, hidden, layers=gru_layers, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(2 * hidden, hidden), nn.Dropout(dropout))
        self.head = ThreatHead(hidden, dropout)

    def _blocked(self, mask: torch.Tensor) -> torch.Tensor:
        """
        [B * heads, N, N], True = may not attend. Padding is never attended to.
        An object can always see itself, so no row is ever fully blocked —
        a fully blocked row makes softmax return NaN.
        """
        b, n = mask.shape
        hidden_keys = ~mask
        if self.training and self.context_dropout > 0:
            hidden_keys = hidden_keys | (torch.rand(b, n, device=mask.device) < self.context_dropout)
        blocked = hidden_keys.unsqueeze(1).expand(b, n, n).clone()
        idx = torch.arange(n, device=mask.device)
        blocked[:, idx, idx] = False
        return blocked.repeat_interleave(self.heads, dim=0)

    def forward(self, x: torch.Tensor, cls: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x [B, N, T, F], cls [B, N], mask [B, N] -> encounter logits [B, N]."""
        h = self.encoder(x, cls)
        a = self.norm1(h)
        a, _ = self.attn(a, a, a, attn_mask=self._blocked(mask), need_weights=False)
        h = h + self.drop(a)
        h = h + self.ffn(self.norm2(h))
        return self.head(h)
