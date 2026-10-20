"""
rebuild_local.py
================
The Colab rebuild (notebooks/colab_rebuild.ipynb) as one resumable script, for a local or college
machine (Windows or Linux, ideally with an NVIDIA GPU). Same steps, same commands, same outputs.

    python tools/rebuild_local.py                # run or resume everything
    python tools/rebuild_local.py --status       # what's done, what's left

Stage-1 CSVs go to data/processed/sanpo_real_462/ (walks already there are skipped, so CSVs copied in
from an earlier run are reused). Each finished Part-B step writes a marker to data/rebuild_local/.done/,
so re-running after a crash or reboot carries on where it stopped. At the end, every output is zipped
into cpe_rebuild_<run-tag>_results.zip in the repo root, laid out like the Colab zip.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "data/processed/sanpo_real_462"
DONE = ROOT / "data/rebuild_local/.done"
PY = f'"{sys.executable}"'
DS = "dataset/GRU/sanpo_real_462_tracked_d3_h8_current_m32"
CORRIDOR = f"{DS}_official_corridor_w0.75_d3_h20"
LADDER = f"evaluation/benchmarks/baseline_ladder/{Path(CORRIDOR).name}"
RUNS = ("--run gru=corridor_h8:gru --run gru_attention=corridor_h8:gru_attention "
        "--run gru_hist4=corridor_hist4:gru --run gru_hist1=corridor_hist1:gru")
FIELDS = ("accel_std", "static_accel_std", "depth_base", "depth_lin", "depth_quad", "bearing_std_deg",
          "init_speed_std", "robust_nu", "gate_nis")
OUTPUT_PATHS = [
    f"{DS}_f5_s0", f"{DS}_official",
    *[f"{DS}_official_corridor_w{w}_d3_h{h}" for w in ("0.75", "1.5") for h in (13, 20, 25)],
    "src/threat_engine/fusion.py",
    "evaluation/benchmarks/kalman_noise_fit",
    "models/GRU/GRU/runs", "models/GRU/GRU_attention/runs",
    "evaluation/benchmarks/baseline_ladder",
    "evaluation/benchmarks/threat_engine_replay",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-tag", default="headfx", help="Names the results zip and the replay folder.")
    p.add_argument("--seeds", default="0 1 2")
    p.add_argument("--download-workers", type=int, default=16)
    p.add_argument("--allow-cpu", action="store_true", help="Run Stage 1 without a GPU (several times slower).")
    p.add_argument("--status", action="store_true")
    return p.parse_args()


def run(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=ROOT)


def step(name: str, *commands: str, after=None) -> None:
    if (DONE / name).exists():
        print(f"[skip] {name} (done earlier)")
        return
    t = time.time()
    for c in commands:
        run(c)
    if after:
        after()
    DONE.mkdir(parents=True, exist_ok=True)
    (DONE / name).write_text(time.strftime("%Y-%m-%d %H:%M"))
    print(f"[done] {name} in {(time.time() - t) / 60:.1f} min", flush=True)


def apply_fit() -> None:
    """Write the Kalman noise (fitted or declared, whichever wins on validation) into fusion.py."""
    out = json.loads((ROOT / "evaluation/benchmarks/kalman_noise_fit/fit_pose_ahead.json").read_text())
    val = {name: out["scores"][name]["val"]["ahead_ll"] for name in ("declared", "fitted")}
    chosen = max(val, key=val.get)
    print("validation 1 s-ahead log-likelihood:", val, "-> using", chosen)
    path = ROOT / "src/threat_engine/fusion.py"
    src = path.read_text(encoding="utf-8")
    for k in FIELDS:
        src, n = re.subn(rf"^(    {k}: float = )[0-9.eE+-]+$",
                         lambda m: m.group(1) + repr(float(f"{out[chosen][k]:.3g}")), src, flags=re.M)
        assert n == 1, k
    path.write_text(src, encoding="utf-8")


STEPS = ["b1_datasets", "b2_kalman", "b3_gru", "b4_ladder", "b4_replay", "b4_kalman_before_after"]


def status() -> list[str]:
    n = len(list(CSV_DIR.glob("*.csv")))
    missing = [s for s in STEPS if not (DONE / s).exists()]
    print(f"Stage 1: {n}/462 walks have a CSV in {CSV_DIR.relative_to(ROOT)}")
    print("Part B done:", [s for s in STEPS if s not in missing], "| left:", missing)
    return missing


def main() -> None:
    args = parse_args()
    if args.status:
        status()
        return

    import torch
    print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.cuda.is_available())
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("No CUDA GPU visible to torch. Install the CUDA build of torch "
                         "(https://pytorch.org/get-started/locally/), or pass --allow-cpu to run slowly.")

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    nested = [d for d in CSV_DIR.iterdir() if d.is_dir() and any(d.glob("*.csv"))]
    if nested and not any(CSV_DIR.glob("*.csv")):
        raise SystemExit(f"The CSVs are in a subfolder ({nested[0].name}): move them up into {CSV_DIR}")
    print(f"{len(list(CSV_DIR.glob('*.csv')))} walks already have a CSV (skipped by Stage 1)")

    # Part A: Stage 1. Re-running retries only walks without a CSV.
    run(f"{PY} tools/stream_sanpo_perception.py --session-fraction 1.0 --max-frames 300 --frame-stride 3 "
        f"--download-workers {args.download_workers} --out-dir data/processed/sanpo_real_462")
    n = len(list(CSV_DIR.glob("*.csv")))
    if n < 440:
        raise SystemExit(f"Only {n}/462 walks have a CSV (network failures?). Run this script again; "
                         "it retries only the missing ones.")

    # Part B
    step("b1_datasets",
         f"{PY} -m dataset.GRU.build --csv-dir data/processed/sanpo_real_462 --tracked-only --hazard-m 3 "
         "--folds 5 --overwrite",
         f"{PY} -m dataset.GRU.resplit --dataset {DS}_f5_s0 --overwrite",
         *[f"{PY} -m dataset.GRU.relabel --dataset {DS}_official --width {w} --depth 3 --horizon {h} --overwrite"
           for w in ("0.75", "1.5") for h in (13, 20, 25)])
    if not (DONE / "b2_kalman").exists():
        run(f"{PY} tools/fit_kalman_noise.py --objective ahead")
        apply_fit()
    step("b2_kalman", f"{PY} -m pytest -q tests/test_kalman_latency.py tests/test_threat_engine.py")
    step("b3_gru", *[
        f"{PY} -m models.GRU.train --arch {arch} --dataset {CORRIDOR} --time-bins 4 {extra}--seeds {args.seeds} "
        f"--run-name {name} --overwrite"
        for arch, extra, name in (("gru", "", "corridor_h8"), ("gru_attention", "", "corridor_h8"),
                                  ("gru", "--history-steps 4 ", "corridor_hist4"),
                                  ("gru", "--history-steps 1 ", "corridor_hist1"))])
    step("b4_ladder",
         f"{PY} -m models.GRU.benchmark --dataset {CORRIDOR} {RUNS}",
         f"{PY} -m models.GRU.benchmark --dataset {CORRIDOR} {RUNS} --reference cv_kalman_declared "
         f"--out-dir {LADDER}/ref_cv_kalman_declared")
    step("b4_replay",
         f"{PY} tools/replay_threat_engine.py --dataset {CORRIDOR} --gru-run models/GRU/GRU/runs/corridor_h8_seed0 "
         f"--name corridor_{args.run_tag} --announcer-sweep v2_gru v2_kalman nearest_in_corridor")
    step("b4_kalman_before_after", f"{PY} tools/compare_kalman_noise.py --split test")

    if status():
        raise SystemExit("NOT DONE: run this script again (it resumes).")
    zip_path = ROOT / f"cpe_rebuild_{args.run_tag}_results.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in OUTPUT_PATHS:
            src = ROOT / p
            for f in ([src] if src.is_file() else sorted(x for x in src.rglob("*") if x.is_file())):
                z.write(f, f.relative_to(ROOT).as_posix())
    print(f"{zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
    report = ROOT / LADDER / "report.md"
    if report.exists():
        print("".join(report.read_text(encoding="utf-8").splitlines(True)[:30]))
    print(f"ALL DONE: send {zip_path.name}")


if __name__ == "__main__":
    main()
