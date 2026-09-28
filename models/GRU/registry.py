"""Architecture name -> class, and where each one's runs are written."""

from __future__ import annotations

from pathlib import Path

from torch import nn

from models.GRU.GRU.model import TrackGRU
from models.GRU.GRU_attention.model import TrackGRUAttention

HERE = Path(__file__).resolve().parent

ARCHS: dict[str, type[nn.Module]] = {
    "gru": TrackGRU,
    "gru_attention": TrackGRUAttention,
}
RUN_ROOTS: dict[str, Path] = {
    "gru": HERE / "GRU" / "runs",
    "gru_attention": HERE / "GRU_attention" / "runs",
}


def build_model(arch: str, model_kwargs: dict) -> nn.Module:
    return ARCHS[arch](**model_kwargs)
