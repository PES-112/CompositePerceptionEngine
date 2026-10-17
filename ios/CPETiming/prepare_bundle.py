"""
prepare_bundle.py
=================
Fill ios/CPETiming/ with what the timing app bundles (all git-ignored):

  Resources/Frames/frame_0000.jpg ...   N SANPO frames (official TEST sessions, camera_head/left,
                                         every 3rd source frame = the 5 Hz processing rate), 1280x720 JPEG q90
  Resources/gru_x.bin, gru_cls.bin       recorded GRU inputs: the arrival dataset's test-split
                                         frames, padded to [N_obj=32, T, F] float32 / [32] int32,
                                         little-endian, M frames back to back
  Resources/bundle.json                  provenance (sessions, frame objects, dataset, models)
  Models/CPEDetector.mlpackage           --export: YOLO26n v3 (FP16 Core ML), via export_yolo26n_edge.py
  Models/CPEArrivalGRU.mlpackage         --export: the arrival GRU, via export_gru_coreml.py

The detector input is real SANPO imagery; the GRU input is real recorded
features (not the detector's own output on these frames), so the app measures
the cost of both models per frame, not end-to-end accuracy. --export needs
macOS (coremltools can't save ML Programs on Windows); the GitHub Actions
workflow .github/workflows/ios-timing.yml runs all of this on a macOS runner.

    python ios/CPETiming/prepare_bundle.py --frames 300 --export
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.GRU.samples import N_FEATURES, load_dataset  # noqa: E402
from tools.audit_sanpo_sessions import fetch, object_url  # noqa: E402
from tools.download_sanpo_valid_streams import BASE_PREFIX, gcs_list  # noqa: E402

DATASET = PROJECT_ROOT / "dataset/GRU/sanpo_real_462_tracked_d3_h8_current_m32_official_corridor_w0.75_d3_h20"
GRU_RUN = PROJECT_ROOT / "models/GRU/GRU/runs/corridor_h8_seed0"
STRIDE = 3
MAX_OBJECTS = 32


def session_frames(sid: str, n: int) -> list[str]:
    prefix = f"{BASE_PREFIX}/{sid}/camera_head/left/video_frames/"
    names = sorted(i["name"] for i in gcs_list(prefix) if i["name"].lower().endswith((".png", ".jpg")))
    return names[::STRIDE][:n]


def write_frames(sessions: list[str], n: int, out: Path) -> list[str]:
    from PIL import Image

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    names: list[str] = []
    for sid in sessions:
        names += session_frames(sid, n - len(names))
        if len(names) >= n:
            break
    for i, name in enumerate(names):
        img = Image.open(io.BytesIO(fetch(object_url(name)))).convert("RGB")
        img.thumbnail((1280, 1280))   # 1280x720: bundle size; the detector resizes to 640 anyway
        img.save(out / f"frame_{i:04d}.jpg", quality=90)
    return names


def write_gru_inputs(m: int, res: Path) -> dict:
    manifest, splits = load_dataset(DATASET, splits=("test",))
    samples = sorted(splits["test"], key=lambda s: (s["session"], s["frame_idx"]))[:m]
    hist = samples[0]["x"].shape[1]
    x = np.zeros((len(samples), MAX_OBJECTS, hist, N_FEATURES), dtype="<f4")
    cls = np.zeros((len(samples), MAX_OBJECTS), dtype="<i4")
    for i, s in enumerate(samples):
        k = min(MAX_OBJECTS, len(s["y"]))
        x[i, :k] = s["x"][:k]
        cls[i, :k] = s["cls"][:k]
    x.tofile(res / "gru_x.bin")
    cls.tofile(res / "gru_cls.bin")
    return {"dataset": manifest["name"], "split": "test", "frames": len(samples), "shape_x": list(x.shape[1:]),
            "objects_per_frame_mean": float(np.mean([len(s["y"]) for s in samples]))}


def export_models(models: Path) -> dict:
    models.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    yolo_root = PROJECT_ROOT / "models/yolo/cpe_yolo26n_hazards_v3_from_base/exports"
    subprocess.run([py, "training/scripts/export_yolo26n_edge.py", "--formats", "coreml", "--quantize", "16",
                    "--device", "cpu"], cwd=PROJECT_ROOT, check=True)
    det = max(yolo_root.glob("*.mlpackage"), key=lambda p: p.stat().st_mtime)
    gru_dir = PROJECT_ROOT / "models/GRU/exports"
    subprocess.run([py, "training/scripts/export_gru_coreml.py", "--run", str(GRU_RUN)], cwd=PROJECT_ROOT, check=True)
    gru = gru_dir / f"cpe_gru_{GRU_RUN.name}_fp16.mlpackage"
    for src, name in ((det, "CPEDetector"), (gru, "CPEArrivalGRU")):
        dst = models / f"{name}.mlpackage"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    return {"detector": det.name, "gru": gru.name, "gru_run": GRU_RUN.name}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--gru-frames", type=int, default=600)
    p.add_argument("--export", action="store_true", help="Also export both models to Core ML (macOS/Linux).")
    args = p.parse_args()

    split = json.loads((PROJECT_ROOT / "simulation/datasets/sanpo/official_split.json").read_text(encoding="utf-8"))
    _, test = load_dataset(DATASET, splits=("test",))
    in_dataset = sorted({s["session"] for s in test["test"]})
    sessions = [sid for sid in in_dataset if sid in set(split["test"])][:5]

    res = HERE / "Resources"
    res.mkdir(exist_ok=True)
    names = write_frames(sessions, args.frames, res / "Frames")
    info = {"frames": len(names), "frame_objects": names, "sessions": sorted({n.split("/")[3] for n in names}),
            "gru_inputs": write_gru_inputs(args.gru_frames, res)}
    if args.export:
        info["models"] = export_models(HERE / "Models")
    (res / "bundle.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"{len(names)} frames, {info['gru_inputs']['frames']} GRU input frames -> {res}")


if __name__ == "__main__":
    main()
