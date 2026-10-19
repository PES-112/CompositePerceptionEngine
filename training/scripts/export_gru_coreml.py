"""
Export a trained arrival GRU (models/GRU) to Core ML for the iPhone timing app
(ios/CPETiming) and, later, the on-device engine.

The exported model has fixed shapes, one frame at a time:
    inputs   x    [1, N, T, F] float32   history features (dataset/GRU/samples.FEATURES)
             cls  [1, N]       int32     class ids (CLASS_VOCAB order)
    output   p    [1, N, K+1]  float32   softmax over the K arrival slices + "never"
N = the checkpoint's max_objects (32), T = its history (8), F = 9. Unused
object slots are zero rows; their outputs are ignored. The softmax is inside the
graph so the app reads probabilities directly; P(arrive) = 1 - p[..., -1].

The conversion is checked against PyTorch on random inputs (max abs difference
in probabilities) unless --no-check. Run it on macOS or Linux: on Windows
coremltools converts the graph but can neither save an ML Program (no native
BlobWriter) nor run one.

    python training/scripts/export_gru_coreml.py                                  # corridor_h8_seed0, FP16
    python training/scripts/export_gru_coreml.py --run models/GRU/GRU/runs/corridor_hist4_seed0 --precision 32
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.GRU.samples import CLASS_VOCAB, FEATURES  # noqa: E402
from models.GRU.registry import build_model  # noqa: E402

DEFAULT_RUN = PROJECT_ROOT / "models/GRU/GRU/runs/corridor_h8_seed0"
DEFAULT_OUT = PROJECT_ROOT / "models/GRU/exports"


class Probabilities(nn.Module):
    """
    TrackGRU's forward with fixed shapes and the softmax folded in. Same layers
    and weights; the shapes are Python ints rather than x.shape, which keeps
    shape arithmetic out of the traced graph. (coremltools 9.0 also needs
    numpy < 2.4: newer numpy rejects its int() of 1-element arrays.)
    """

    def __init__(self, model: nn.Module, n: int, t: int):
        super().__init__()
        self.enc, self.head, self.n, self.t = model.encoder, model.head, n, t

    def forward(self, x: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
        n, t = self.n, self.t
        c = self.enc.class_emb(cls.long()).reshape(1, n, 1, -1).expand(1, n, t, self.enc.class_emb.embedding_dim)
        steps = self.enc.inp(torch.cat([x, c], dim=-1)).reshape(n, t, -1)
        _, h = self.enc.gru(steps)
        return torch.softmax(self.head(h[-1].reshape(1, n, -1)), dim=-1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, default=DEFAULT_RUN, help="A models/GRU run directory with model.pt.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--precision", choices=["16", "32"], default="16")
    p.add_argument("--no-check", action="store_true")
    args = p.parse_args()

    import coremltools as ct

    ckpt = torch.load(args.run / "model.pt", map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    if ckpt["arch"] != "gru":
        raise SystemExit("only the per-object GRU is exported (GRU_attention lost held-out; architecture.md §6.4)")
    if not cfg.get("time_bins"):
        raise SystemExit(f"{args.run} is a yes/no model; the engine needs --time-bins")
    model = build_model(ckpt["arch"], ckpt["model_kwargs"]).eval()
    model.load_state_dict(ckpt["state_dict"])
    n = int(cfg["dataset_params"].get("max_objects", 32))
    t = int(cfg.get("history_steps") or cfg["dataset_params"]["history"])
    f = len(FEATURES)
    wrapped = Probabilities(model, n, t).eval()
    with torch.no_grad():   # the rewrite must match the checkpoint's own forward exactly
        xs = torch.randn(1, n, t, f) * 0.5
        cs = torch.randint(0, len(CLASS_VOCAB), (1, n))
        ref = torch.softmax(model(xs, cs, torch.ones(1, n, dtype=torch.bool)), dim=-1)
        torch.testing.assert_close(wrapped(xs, cs.int()), ref)
    x = torch.zeros(1, n, t, f)
    cls = torch.zeros(1, n, dtype=torch.int32)
    traced = torch.jit.trace(wrapped, (x, cls))

    if platform.system() == "Windows":
        # The ML Program converter runs here, but its weight serializer (BlobWriter) is a native
        # library shipped for macOS/Linux only; the CI workflow (.github/workflows/ios-timing.yml) exports on macOS.
        print("note: coremltools can't save ML Programs on Windows; conversion is checked, then this stops.")
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="x", shape=(1, n, t, f), dtype=np.float32),
                ct.TensorType(name="cls", shape=(1, n), dtype=np.int32)],
        outputs=[ct.TensorType(name="p", dtype=np.float32)],
        compute_precision=ct.precision.FLOAT16 if args.precision == "16" else ct.precision.FLOAT32,
    )
    mlmodel.short_description = (f"CPE arrival GRU ({args.run.name}): P(arrival slice | history), "
                                 f"{cfg['time_bins']} slices over {cfg['horizon']} steps of 0.2 s + never")
    mlmodel.user_defined_metadata.update({
        "run": args.run.name, "features": ",".join(FEATURES), "classes": ",".join(CLASS_VOCAB),
        "time_bins": str(cfg["time_bins"]), "horizon_steps": str(cfg["horizon"]), "step_s": "0.2",
        "max_objects": str(n), "history": str(t),
        "velocity_units": "legacy (2x): checkpoints trained on pre-2026-10-14 CSVs; see GRUArrivalPredictor"})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"cpe_gru_{args.run.name}_fp{args.precision}"
    out = args.out_dir / f"{stem}.mlpackage"
    mlmodel.save(str(out))

    check: dict = {"skipped": True}
    if not args.no_check:
        if platform.system() == "Darwin":
            rng = np.random.default_rng(0)
            xs = rng.standard_normal((1, n, t, f)).astype(np.float32) * 0.5
            cs = rng.integers(0, len(CLASS_VOCAB), (1, n)).astype(np.int32)
            ref = wrapped(torch.from_numpy(xs), torch.from_numpy(cs)).detach().numpy()
            got = mlmodel.predict({"x": xs, "cls": cs})["p"]
            check = {"skipped": False, "max_abs_diff": float(np.abs(ref - got).max())}
            print(f"Core ML vs PyTorch: max |Δp| = {check['max_abs_diff']:.2e}")
        else:
            check = {"skipped": True, "reason": f"coremltools cannot run models on {platform.system()}"}
            print(f"Converted; numeric check skipped ({check['reason']}). Run it on a Mac.")

    (args.out_dir / f"{stem}.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "run": str(args.run.as_posix()),
        "package": out.name, "precision": args.precision, "shapes": {"x": [1, n, t, f], "cls": [1, n],
                                                                       "p": [1, n, cfg["time_bins"] + 1]},
        "coremltools": ct.__version__, "torch": torch.__version__, "check": check}, indent=2), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
