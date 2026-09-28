"""
train.py
========
Train one of the two GRU architectures on a dataset built by
dataset/GRU/build.py, then score it on that dataset's test sessions.

    python -m dataset.GRU.build --csv-dir data/processed/ablation_30pct
    python -m models.GRU.train --arch gru           --dataset dataset/GRU/ablation_30pct_h8_current_m32_s0 --seeds 0 1 2
    python -m models.GRU.train --arch gru_attention --dataset dataset/GRU/ablation_30pct_h8_current_m32_s0 --seeds 0 1 2

Both architectures read the same dataset files (same samples, same session
split) and run this same loop, so a gap between their metrics comes from the
attention layer. Each seed writes models/GRU/<GRU|GRU_attention>/runs/<run-name>_seed<N>/
containing model.pt, config.json, history.csv, metrics.json and report.md.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset.GRU.samples import N_CLASSES, N_FEATURES, collate, load_dataset
from models.GRU.evaluate import evaluate_samples, write_report
from models.GRU.registry import ARCHS, RUN_ROOTS, build_model

POS_WEIGHT_CAP = 50.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", choices=sorted(ARCHS), required=True)
    p.add_argument("--dataset", type=Path, required=True, help="A folder written by dataset/GRU/build.py.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--run-name", default=f"run_{time.strftime('%Y_%m_%d')}")
    p.add_argument("--out-root", type=Path, default=None, help="Defaults to models/GRU/<arch folder>/runs.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # model
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--gru-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--heads", type=int, default=4, help="gru_attention only.")
    p.add_argument("--context-dropout", type=float, default=0.1, help="gru_attention only.")
    # optimisation
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64, help="Frames per batch.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=6, help="Early-stop after this many epochs without val improvement.")
    return p.parse_args()


def run_epoch(model: nn.Module, loader: DataLoader, device: str,
              pos_weight: torch.Tensor, optimizer: torch.optim.Optimizer | None = None) -> float:
    """Mean per-object BCE over the epoch; trains when an optimizer is given."""
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            mask = batch["mask"].to(device)
            logits = model(batch["x"].to(device), batch["cls"].to(device), mask)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits[mask], batch["y"].to(device)[mask], pos_weight=pos_weight)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            n = int(mask.sum())
            total += loss.item() * n
            count += n
    return total / max(count, 1)


def train_seed(args: argparse.Namespace, seed: int, manifest: dict, samples: dict, run_dir: Path) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_kwargs = {"n_features": N_FEATURES, "n_classes": N_CLASSES, "hidden": args.hidden,
                    "dropout": args.dropout, "gru_layers": args.gru_layers}
    if args.arch == "gru_attention":
        model_kwargs.update(heads=args.heads, context_dropout=args.context_dropout)
    model = build_model(args.arch, model_kwargs).to(args.device)

    y = np.concatenate([s["y"] for s in samples["train"]])
    pos_weight = float(min((len(y) - y.sum()) / max(y.sum(), 1.0), POS_WEIGHT_CAP))
    pos_weight_t = torch.tensor(pos_weight, device=args.device)

    train_loader = DataLoader(samples["train"], batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(samples["val"], batch_size=args.batch_size, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val, best_epoch, best_state, stale, log = float("inf"), 0, None, 0, []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, args.device, pos_weight_t, optimizer)
        val_loss = run_epoch(model, val_loader, args.device, pos_weight_t)
        log.append((epoch, train_loss, val_loss))
        print(f"  seed {seed} epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}", flush=True)
        if val_loss < best_val - 1e-4:
            best_val, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError(f"seed {seed}: validation loss was never finite - check the dataset for NaNs")
    model.load_state_dict(best_state)

    config = {
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "seeds"},
        "seed": seed,
        "dataset_name": manifest["name"],
        "dataset_fingerprint": manifest["source"]["fingerprint"],
        "dataset_params": manifest["params"],
        "features": manifest["features"],
        "split": manifest["split"],
        "samples": {k: len(v) for k, v in samples.items()},
        "pos_weight": pos_weight,
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "train_seconds": round(time.time() - started, 1),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": args.arch, "model_kwargs": model_kwargs, "state_dict": best_state, "config": config},
               run_dir / "model.pt")
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "history.csv").write_text(
        "epoch,train_loss,val_loss\n" + "".join(f"{e},{t:.6f},{v:.6f}\n" for e, t, v in log), encoding="utf-8")

    metrics = evaluate_samples(model, samples["test"], args.device)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_report(metrics, run_dir / "report.md", f"{args.arch} - {run_dir.name}")
    m = metrics["methods"]
    print(f"  seed {seed} done: best epoch {best_epoch}, "
          f"test encounter_top1 model {m['model']['all']['encounter_top1'][0]:.3f} "
          f"vs K0 {m['K0']['all']['encounter_top1'][0]:.3f} -> {run_dir}")


def main() -> None:
    args = parse_args()
    out_root = args.out_root or RUN_ROOTS[args.arch]
    run_dirs = {seed: out_root / f"{args.run_name}_seed{seed}" for seed in args.seeds}
    clash = [d for d in run_dirs.values() if (d / "model.pt").exists()]
    if clash and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite {clash[0]} (pass --overwrite or a new --run-name)")

    manifest, samples = load_dataset(args.dataset)
    print(f"{args.arch} on {manifest['name']}: "
          + ", ".join(f"{part} {manifest['stats'][part]['sessions']} sessions / {len(v)} frames"
                      for part, v in samples.items()))

    for seed in args.seeds:
        train_seed(args, seed, manifest, samples, run_dirs[seed])


if __name__ == "__main__":
    main()
