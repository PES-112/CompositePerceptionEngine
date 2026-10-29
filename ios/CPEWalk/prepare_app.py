"""
prepare_app.py
==============
Fill ios/CPEWalk/Models and ios/CPEWalk/AppData (both generated, git-ignored) for the on-device app.

  Models/CPEDetector.mlpackage     YOLO26n v3 (FP16 Core ML, raw [1, 21, 8400] head; NMS in Swift)
  Models/CPESegmenter.mlpackage    YOLO26n-sem walkable surface (FP16), class logits [1, 8, 80, 80] (stride 8,
                                   before the upsample and argmax) so the phone can add the road bias
                                   (tools/tune_surface_rules.py) before choosing a class
  Models/CPEArrivalGRU.mlpackage   the arrival GRU (training/scripts/export_gru_coreml.py), [1, 32, 8, 9] -> p
  AppData/cpe_config.json          every constant the Swift engine needs, read from the Python modules
                                   themselves (severity table, half-widths, fitted Kalman noise, announcer
                                   settings, segmentation rule), so the two implementations can't drift
  AppData/golden_session.json      one official-test SANPO walk through the Python engine, frame by frame
                                   (inputs, fused tracks, GRU inputs and P(arrive), ranking, announcements,
                                   phrases): the Swift unit tests replay it and compare

Run on macOS (coremltools' ML Program writer):

    .venv-seg.nosync/bin/python ios/CPEWalk/prepare_app.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, fields
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DETECTOR_PT = PROJECT_ROOT / "training/runs/cpe_yolo26n_hazards_v4_idd/weights/best.pt"
# Two-detector mode (methodology §4.26): set DETECTOR_PT to the IDD fine-tune and list here the classes
# the original v3 detector still answers for; empty = one detector.
LEGACY_DETECTOR_PT = PROJECT_ROOT / "training/runs/cpe_yolo26n_hazards_v3_from_base/weights/best.pt"
# From tools/compare_idd_detectors.py (evaluation/benchmarks/detector_idd/report.md).
LEGACY_CLASSES: list[str] = ["stop sign", "fire hydrant", "pole", "bollard", "stairs", "crosswalk", "pothole",
                             "puddle", "bench"]
SEGMENTER_PT = PROJECT_ROOT / "training/runs/cpe_yolo26n_sem_walkable_idd/weights/best.pt"
GRU_RUN = PROJECT_ROOT / "models/GRU/GRU/runs/corridor_h8_seed0"
DATASET = PROJECT_ROOT / "dataset/GRU/sanpo_real_462_tracked_d3_h8_current_m32_official_corridor_w0.75_d3_h20"
# The operating point the demo and viewer use (validation-chosen budget_5 for v2_gru, corridor_headfx).
THRESHOLD, COOLDOWN_S, SWITCH_MARGIN, STALE_S = 0.6, 8.0, 1.0, 0.6
GOLDEN_FRAMES = 120


def export_segmenter_logits(dst: Path) -> None:
    """The walkable-surface model with its stride-8 class logits as the output (no upsample, no argmax)."""
    import coremltools as ct
    import torch
    from ultralytics import YOLO

    net = YOLO(str(SEGMENTER_PT)).model.float().eval()

    class Logits(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = net

        def forward(self, x):
            y = self.net(x)
            return y[0] if isinstance(y, (list, tuple)) else y

    x = torch.zeros(1, 3, 640, 640)
    traced = torch.jit.trace(Logits().eval(), x, strict=False)
    m = ct.convert(traced, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS17,
                   inputs=[ct.ImageType(name="image", shape=x.shape, scale=1 / 255, color_layout=ct.colorlayout.RGB)],
                   outputs=[ct.TensorType(name="logits")], compute_precision=ct.precision.FLOAT16)
    m.user_defined_metadata["names"] = str(dict(YOLO(str(SEGMENTER_PT)).names))
    m.save(str(dst))


def export_detector(work: Path, pt: Path) -> Path:
    """FP16 Core ML, raw head (no NMS), one package per checkpoint (named after its run)."""
    run = pt.parents[1].name
    out = work / f"{run}.mlpackage"
    if not out.exists():
        from ultralytics import YOLO
        tmp = work / f"{run}.pt"
        shutil.copy2(pt, tmp)
        YOLO(str(tmp)).export(format="coreml", imgsz=640, half=True, device="cpu")
    return out


def export_models(models: Path) -> None:
    models.mkdir(parents=True, exist_ok=True)
    work = PROJECT_ROOT / "data/ios_build"
    work.mkdir(parents=True, exist_ok=True)
    det = export_detector(work, DETECTOR_PT)
    seg = work / f"{SEGMENTER_PT.parents[1].name}_logits.mlpackage"
    if not seg.exists():
        export_segmenter_logits(seg)
    gru = PROJECT_ROOT / f"models/GRU/exports/cpe_gru_{GRU_RUN.name}_fp16.mlpackage"
    if not gru.exists():
        subprocess.run([sys.executable, "training/scripts/export_gru_coreml.py", "--run", str(GRU_RUN)],
                       cwd=PROJECT_ROOT, check=True)
    pairs = [(det, "CPEDetector"), (seg, "CPESegmenter"), (gru, "CPEArrivalGRU")]
    shutil.rmtree(models / "CPEDetectorLegacy.mlpackage", ignore_errors=True)
    if LEGACY_CLASSES:
        pairs.append((export_detector(work, LEGACY_DETECTOR_PT), "CPEDetectorLegacy"))
    for src, name in pairs:
        dst = models / f"{name}.mlpackage"
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)
        print(f"  {name} <- {src.relative_to(PROJECT_ROOT)}")


def config() -> dict:
    from dataset.GRU.samples import CLASS_VOCAB, DEFAULT_HISTORY, DEFAULT_MAX_OBJECTS
    from src.narration import templates
    from src.perception_stack import depth_loader, pipeline, yolo_tracker
    from src.perception_stack.physics import class_severity
    from src.segmentation import context, edge_alert, labels
    from src.threat_engine import announcer, envelope, fusion, priority, stairs

    noise = fusion.NoiseModel()
    return {
        "classes": CLASS_VOCAB,
        "severity": {c: class_severity(c) for c in CLASS_VOCAB},
        "severity_gamma": priority.SEVERITY_GAMMA,
        "half_width_m": {**envelope.CLASS_HALF_WIDTH_M, "_default": envelope.DEFAULT_HALF_WIDTH_M},
        "static_classes": sorted(fusion.STATIC_CLASSES),
        "detector": {"imgsz": 640, "conf": yolo_tracker.DEFAULT_CONF, "nms_iou": 0.7, "max_det": 300,
                     "four_wheelers": list(yolo_tracker.FOUR_WHEELERS), "car_min_width_m": yolo_tracker.CAR_MIN_WIDTH_M,
                     "edge_margin_px": yolo_tracker.EDGE_MARGIN_PX, "legacy_classes": LEGACY_CLASSES},
        "depth": {"min_m": depth_loader.MIN_DEPTH_M, "max_m": depth_loader.MAX_DEPTH_M, "center_frac": 0.2,
                  "velocity_window": pipeline.VELOCITY_WINDOW},
        "gru": {"history": DEFAULT_HISTORY, "max_objects": DEFAULT_MAX_OBJECTS, "time_bins": 4, "horizon": 20,
                "step_s": 0.2},
        "noise": {f.name: getattr(noise, f.name) for f in fields(noise)},
        "fusion": {"max_gated": fusion.MAX_GATED, "max_misses": fusion.MAX_MISSES, "min_hits": fusion.MIN_HITS,
                   "ego_smoothing": fusion.EGO_SMOOTHING, "min_walk_speed": fusion.MIN_WALK_SPEED},
        "envelope": {"ttc_s": envelope.OVERRIDE_TTC_S, "corridor_m": envelope.CORRIDOR_WIDTH_M,
                     "envelope_corridor_m": envelope.ENVELOPE_CORRIDOR_WIDTH_M,
                     "min_closing_ms": envelope.MIN_CLOSING_MS, "max_coast_frames": envelope.MAX_COAST_FRAMES},
        "priority": {"ignore_threshold": THRESHOLD, "context_in_path": priority.CONTEXT_IN_PATH,
                     "context_near_boundary": priority.CONTEXT_NEAR_BOUNDARY,
                     "context_other_surface": priority.CONTEXT_OTHER_SURFACE},
        "announcer": {"cooldown_s": COOLDOWN_S, "escalation": announcer.ESCALATION, "switch_margin": SWITCH_MARGIN,
                      "override_repeat_s": announcer.OVERRIDE_REPEAT_S, "stale_s": STALE_S,
                      "clip_s": announcer.CLIP_S, "edge_stale_s": announcer.EDGE_STALE_S,
                      "surface_idle_s": announcer.SURFACE_IDLE_S},
        "narration": {"fast_closing_ms": templates.FAST_CLOSING_MS, "near_m": templates.NEAR_DISTANCE_M,
                      "bearing_center_deg": templates.BEARING_CENTER_DEG,
                      "bearing_far_deg": templates.BEARING_FAR_DEG,
                      "distance_buckets_m": list(templates.DISTANCE_BUCKETS_M),
                      "edge_max_m": templates.EDGE_MAX_M},
        "segmentation": {"labels": labels.CPE_CLASSES, "walker_rows": context.WALKER_ROWS,
                         "walker_cols": context.WALKER_COLS, "edge_cols": context.EDGE_COLS,
                         "depth_tol_frac": context.DEPTH_TOL_FRAC, "depth_tol_min_m": context.DEPTH_TOL_MIN_M,
                         "min_ground_px": context.MIN_GROUND_PX, "boundary_share": context.BOUNDARY_SHARE,
                         "curb_share": context.CURB_SHARE, "max_edge_m": context.MAX_EDGE_M,
                         "road_logit_bias": context.ROAD_LOGIT_BIAS,
                         "edge_alert_m": edge_alert.ALERT_M, "edge_confirm_frames": edge_alert.CONFIRM_FRAMES,
                         "edge_clear_s": edge_alert.CLEAR_S,
                         "surface_confirm_frames": edge_alert.SURFACE_CONFIRM_FRAMES},
        "stairs": {"enter_windows": [list(w) for w in stairs.ENTER_WINDOWS],
                   **{k.lower(): getattr(stairs, k) for k in ("WINDOW_S", "ENTER_SLOPE", "MIN_RUN_M", "ENTER_FRAMES",
                                                              "EXIT_WINDOW_S", "EXIT_RISE_M", "EXIT_RUN_M",
                                                              "EXIT_FRAMES", "WARMUP_S")}},
    }


def golden() -> dict:
    """One official-test walk through the Python engine, recorded frame by frame."""
    from dataset.GRU.samples import legacy_units, load_dataset
    from src.narration.templates import narrate
    from src.threat_engine.announcer import AnnouncementManager
    from src.threat_engine.engine import ThreatEngine
    from src.threat_engine.fusion import GroundTracker
    from src.threat_engine.priority import GRUArrivalPredictor
    from tools.replay_threat_engine import AUDIT, STEP_S, STRIDE, session_poses, session_rows

    manifest, splits = load_dataset(DATASET, splits=("test",))
    audit = {r["session_id"]: r for r in json.loads(AUDIT.read_text(encoding="utf-8"))["sessions"]}
    by = defaultdict(list)
    for s in splits["test"]:
        by[s["session"]].append(s)
    sid = max(by, key=lambda k: sum(len(s["track_ids"]) for s in by[k][:GOLDEN_FRAMES]))   # a busy walk
    frames = session_rows(by[sid], audit[sid]["fx"], audit[sid]["image_width"], legacy_units(manifest))
    poses = session_poses(sid)
    gru = GRUArrivalPredictor(GRU_RUN / "model.pt")
    engine = ThreatEngine(predictor=gru, ignore_threshold=THRESHOLD, tracker=GroundTracker(),
                          announcer=AnnouncementManager(cooldown_s=COOLDOWN_S, switch_margin=SWITCH_MARGIN,
                                                        stale_s=STALE_S))
    out = []
    for k in sorted(frames)[:GOLDEN_FRAMES]:
        pose = poses[min(STRIDE * k, len(poses) - 1)]
        rows = [{key: r[key] for key in ("track_id", "class", "confidence", "distance_m", "bearing_deg",
                                         "velocity_ms", "bbox_area_px")} for r in frames[k]]
        f = engine.step(k, k * STEP_S, frames[k], pose)
        live, x, _ = gru.encode(k, list(f.tracks))
        arrivals = gru.predict(k, list(f.tracks))     # observe() is idempotent within a frame
        a = f.announcement
        out.append({
            "k": k, "t": k * STEP_S, "pose": asdict(pose), "rows": rows,
            "tracks": [{"id": t.track_id, "lateral": t.lateral_m, "forward": t.forward_m,
                        "closing": t.closing_speed_ms, "lateral_speed": t.lateral_speed_ms,
                        "position_std": t.position_std_m, "confirmed": t.confirmed, "measured": t.measured}
                       for t in f.tracks],
            "gru_x": {t.track_id: x[i].tolist() for i, t in enumerate(live)},
            "p_arrive": {tid: a_.p_arrive for tid, a_ in arrivals.items()},
            "ranked": [[r.track.track_id, r.priority] for r in f.ranked],
            "overrides": [o.track.track_id for o in f.overrides],
            "announcement": None if a is None else {"kind": a.kind, "track": a.track_id,
                                                    "text": narrate(f.narrator_event())},
        })
    return {"session": sid, "fx": audit[sid]["fx"], "image_width": audit[sid]["image_width"],
            "threshold": THRESHOLD, "frames": out}


def main() -> None:
    print("models:")
    export_models(HERE / "Models")
    data = HERE / "AppData"
    data.mkdir(exist_ok=True)
    (data / "cpe_config.json").write_text(json.dumps(config(), indent=1), encoding="utf-8")
    g = golden()
    (data / "golden_session.json").write_text(json.dumps(g), encoding="utf-8")
    n_ann = sum(f["announcement"] is not None for f in g["frames"])
    print(f"AppData: cpe_config.json, golden_session.json ({g['session']}, {len(g['frames'])} frames, "
          f"{n_ann} announcements)")


if __name__ == "__main__":
    main()
