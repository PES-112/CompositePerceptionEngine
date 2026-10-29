"""
render_demo_video.py
====================
A demo video of the whole v2 pipeline on one SANPO walk, with the alerts spoken.

Every box in architecture.md §5 runs, as on the phone, at 5 Hz:
  SANPO RGB + depth ─> Stage 1 (YOLO26n v3 + ByteTrack, box depth, pinhole bearing)
  RGB ─> walkable-surface segmentation (YOLO26n-sem @640) ─> per-track SceneContext + edge ahead
       ─> EdgeAlerter ("Curb ahead, 3 meters."), SurfaceTracker ("Now on the road.", "Walking on the sidewalk.")
  ─> ThreatEngine: Kalman fusion with camera pose, safety envelope, GRU arrival, fused priority,
     announcement manager ─> narration phrase ─> speech (Windows' built-in voice or macOS `say`, a
     stand-in for the pre-rendered clip library), panned in stereo toward the object's bearing

The frame shows the tracked boxes (red: safety override, orange: ranked above the threshold,
grey: tracked, not a threat; a dashed outline marks objects that really do enter the walking
corridor within 4 s, the replay's ground truth), the segmentation tinted on the ground, the
engine's top threats with P(arrive) and context, and the caption of what is being said.
Audio rule, as on the phone: an override cuts whatever is playing, and any hazard alert cuts a
surface clip; any other alert waits for the current clip and is dropped if it would start more
than 1.2 s late.

    .venv-seg/Scripts/python tools/render_demo_video.py --session QVzjzsSertgX6WAHcL5wP6pnnEieXu8f
    .venv-seg/Scripts/python tools/render_demo_video.py --session <id> --frames 150 --title "Curb and road"

Writes data/demo/<name>.mp4 (with sound when it can be muxed: imageio-ffmpeg if installed,
else Windows' MediaComposition), <name>_silent.mp4, <name>.wav and <name>.json (the alert log).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.narration.spatial import pan_stereo  # noqa: E402
from src.narration.templates import narrate  # noqa: E402
from src.perception_stack.depth_loader import load_depth_map  # noqa: E402
from src.segmentation.context import ObjectRef, scene_frame  # noqa: E402
from src.segmentation.edge_alert import EdgeAlerter, SurfaceTracker  # noqa: E402
from src.segmentation.labels import CPE_CLASSES, CROSSWALK, CURB, NON_WALKABLE, ROAD, STAIRS, WALKABLE  # noqa: E402
from src.threat_engine.announcer import AnnouncementManager  # noqa: E402
from src.threat_engine.engine import ThreatEngine  # noqa: E402
from src.threat_engine.fusion import GroundTracker  # noqa: E402
from src.threat_engine.priority import GRUArrivalPredictor, context_multiplier  # noqa: E402
from tools.build_session_viewer import GRU_RUN, STEP_S, STRIDE, ground_truth  # noqa: E402
from tools.replay_threat_engine import session_poses  # noqa: E402
from tools.stream_sanpo_perception import DEFAULT_MODEL, fetch_session_frames, session_camera  # noqa: E402

SEG_WEIGHTS = PROJECT_ROOT / "training/runs/cpe_yolo26n_sem_walkable/weights/best.pt"
OUT_DIR = PROJECT_ROOT / "data/demo"
DOWNSAMPLE = 4                  # class maps scored at 1/4 resolution, as tools/build_scene_context.py
VIEW_W, PANEL_W, CAPTION_H = 960, 360, 92
FPS_OUT = 15                    # each 5 Hz processed frame is shown 3 times
SAMPLE_RATE = 22050
STALE_S = 1.2
GROUND_TINT = {WALKABLE: (80, 190, 80), CROSSWALK: (40, 210, 250), ROAD: (130, 120, 115),
               CURB: (40, 80, 240), STAIRS: (200, 70, 170), NON_WALKABLE: (40, 100, 140)}   # BGR
RED, ORANGE, GREY, WHITE = (60, 60, 230), (40, 150, 245), (170, 170, 170), (255, 255, 255)


def tts_clips(phrases: list[str], workdir: Path) -> dict[str, np.ndarray]:
    """One WAV per distinct phrase: Windows' built-in voice (System.Speech), or macOS `say`; no install."""
    workdir.mkdir(parents=True, exist_ok=True)
    jobs = {p: workdir / f"clip_{i:03d}.wav" for i, p in enumerate(phrases)}
    if sys.platform == "darwin":
        for phrase, path in jobs.items():
            subprocess.run(["say", "-r", "230", "-o", str(path), f"--data-format=LEI16@{SAMPLE_RATE}", phrase],
                           check=True)
        out = {}
        for phrase, path in jobs.items():
            with wave.open(str(path)) as w:
                out[phrase] = np.frombuffer(w.readframes(w.getnframes()), np.int16)
        return out
    script = ["Add-Type -AssemblyName System.Speech",
              "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer",
              "$s.Rate = 2",
              f"$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo({SAMPLE_RATE}, "
              "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)"]
    for phrase, path in jobs.items():
        text = phrase.replace("'", "''")
        script += [f"$s.SetOutputToWaveFile('{path}', $f)", f"$s.Speak('{text}')"]
    script.append("$s.Dispose()")
    ps1 = workdir / "tts.ps1"
    ps1.write_text("\n".join(script), encoding="utf-8-sig")
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1)], check=True)
    out = {}
    for phrase, path in jobs.items():
        with wave.open(str(path)) as w:
            out[phrase] = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    return out


def schedule_audio(alerts: list[dict], clips: dict[str, np.ndarray], duration_s: float) -> tuple[np.ndarray, list[dict]]:
    """Mix the clips into one track by the phone's rule; returns the track and what was actually heard."""
    track = np.zeros((int((duration_s + 3) * SAMPLE_RATE), 2), np.int16)
    heard, busy_until, playing = [], 0.0, None
    for a in alerts:
        clip = pan_stereo(clips[a["text"]], a.get("bearing_deg", 0.0))   # heard from the object's side
        start = a["t"]
        cuts = a["kind"] == "override" or (playing == "surface" and a["kind"] != "surface")
        if not cuts and start < busy_until:
            if busy_until - start > STALE_S:
                continue                                            # stale: dropped
            start = busy_until
        if cuts:                                                    # an override (or a hazard over a surface
                                                                    # clip) cuts the current clip and
            heard = [h for h in heard if h["spoken_at"] < start]   # any clip still waiting to start
            for h in heard:
                h["spoken_until"] = min(h["spoken_until"], round(start, 2))
        i = int(start * SAMPLE_RATE)
        track[i:] = 0
        track[i:i + len(clip)] = clip[: len(track) - i]
        busy_until, playing = start + len(clip) / SAMPLE_RATE, a["kind"]
        heard.append({**a, "spoken_at": round(start, 2), "spoken_until": round(busy_until, 2)})
    return track, heard


def write_wav(path: Path, track: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(track.shape[1] if track.ndim == 2 else 1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(track.tobytes())


def mux(video: Path, audio: Path, out: Path) -> bool:
    try:
        import imageio_ffmpeg
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video), "-i", str(audio),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)], check=True)
        return True
    except ImportError:
        if shutil.which("ffmpeg"):                                  # e.g. Homebrew on macOS
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-i", str(audio), "-c:v",
                            "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)], check=True)
            return True
    # Windows' own encoder (WinRT MediaComposition): H.264 + AAC, no install.
    ps = f"""
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {{ $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' }})[0]
$asTaskP = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {{ $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperationWithProgress`2' }})[0]
function Await($op, [Type]$t) {{ $task = $asTask.MakeGenericMethod($t).Invoke($null, @($op)); $task.Wait() | Out-Null; $task.Result }}
function AwaitP($op, [Type]$t, [Type]$p) {{ $task = $asTaskP.MakeGenericMethod($t, $p).Invoke($null, @($op)); $task.Wait() | Out-Null; $task.Result }}
[Windows.Storage.StorageFile,Windows.Storage,ContentType=WindowsRuntime] | Out-Null
[Windows.Media.Editing.MediaComposition,Windows.Media.Editing,ContentType=WindowsRuntime] | Out-Null
[Windows.Media.Transcoding.TranscodeFailureReason,Windows.Media.Transcoding,ContentType=WindowsRuntime] | Out-Null
$v = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync('{video}')) ([Windows.Storage.StorageFile])
$a = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync('{audio}')) ([Windows.Storage.StorageFile])
$dir = Await ([Windows.Storage.StorageFolder]::GetFolderFromPathAsync('{out.parent}')) ([Windows.Storage.StorageFolder])
$o = Await ($dir.CreateFileAsync('{out.name}', [Windows.Storage.CreationCollisionOption]::ReplaceExisting)) ([Windows.Storage.StorageFile])
$comp = New-Object Windows.Media.Editing.MediaComposition
$clip = Await ([Windows.Media.Editing.MediaClip]::CreateFromFileAsync($v)) ([Windows.Media.Editing.MediaClip])
[System.Collections.Generic.ICollection[Windows.Media.Editing.MediaClip]].GetMethod('Add').Invoke($comp.Clips, @($clip)) | Out-Null
$track = Await ([Windows.Media.Editing.BackgroundAudioTrack]::CreateFromFileAsync($a)) ([Windows.Media.Editing.BackgroundAudioTrack])
[System.Collections.Generic.ICollection[Windows.Media.Editing.BackgroundAudioTrack]].GetMethod('Add').Invoke($comp.BackgroundAudioTracks, @($track)) | Out-Null
$r = AwaitP ($comp.RenderToFileAsync($o)) ([Windows.Media.Transcoding.TranscodeFailureReason]) ([double])
Write-Output "render: $r"
"""
    script = out.with_suffix(".mux.ps1")
    script.write_text(ps, encoding="utf-8-sig")
    res = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
                         capture_output=True, text=True)
    script.unlink(missing_ok=True)
    ok = res.returncode == 0 and "render: None" in res.stdout and out.exists() and out.stat().st_size > 0
    if not ok:
        print("mux failed:", (res.stdout + res.stderr)[-800:])
    return ok


def dashed_rect(img, p1, p2, colour, thickness=2, dash=9):
    x1, y1 = p1
    x2, y2 = p2
    import cv2
    for xa in range(x1, x2, dash * 2):
        cv2.line(img, (xa, y1), (min(xa + dash, x2), y1), colour, thickness)
        cv2.line(img, (xa, y2), (min(xa + dash, x2), y2), colour, thickness)
    for ya in range(y1, y2, dash * 2):
        cv2.line(img, (x1, ya), (x1, min(ya + dash, y2)), colour, thickness)
        cv2.line(img, (x2, ya), (x2, min(ya + dash, y2)), colour, thickness)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", required=True, help="SANPO session id (use official-test walks).")
    p.add_argument("--frames", type=int, default=150, help="Processed frames (5 per second); 150 = 30 s.")
    p.add_argument("--threshold", type=float, default=0.6, help="Ignore threshold (v2_gru early_budget_5).")
    p.add_argument("--title", default="", help="Shown in the top-left corner.")
    p.add_argument("--name", default=None, help="Output name (default: the session id).")
    p.add_argument("--device", default="cpu", help="Segmentation model device: cpu, mps or 0.")
    args = p.parse_args()

    import cv2
    from ultralytics import YOLO

    started = time.time()
    sid, name = args.session, args.name or args.session
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scratch = PROJECT_ROOT / "data/viewer/_scratch" / sid
    camera = session_camera({"session_id": sid})
    rgb_dir, depth_dir, _ = fetch_session_frames({"session_id": sid}, scratch, SimpleNamespace(
        depth_kind="depth_maps", frame_stride=STRIDE, max_frames=args.frames, download_workers=6))
    images, depths = sorted(rgb_dir.iterdir()), sorted(depth_dir.iterdir())
    print(f"{len(images)} frames downloaded ({time.time() - started:.0f} s); Stage 1...", flush=True)

    from src.perception_stack import run_perception_stream
    frames = list(run_perception_stream(rgb_dir, depth_dir, camera["fps"] / STRIDE, source="sanpo", frame_step=1,
                                        model_path=str(DEFAULT_MODEL), focal_px=camera["fx"]))
    poses = session_poses(sid)
    truth = ground_truth(frames, poses)
    seg = YOLO(str(SEG_WEIGHTS))
    engine = ThreatEngine(predictor=GRUArrivalPredictor(GRU_RUN / "model.pt"), ignore_threshold=args.threshold,
                          tracker=GroundTracker(),
                          announcer=AnnouncementManager(cooldown_s=8.0, switch_margin=1.0, stale_s=0.6))
    edges, surfaces = EdgeAlerter(), SurfaceTracker()
    print(f"Stage 1 done ({time.time() - started:.0f} s); segmentation + engine...", flush=True)

    h0, w0 = cv2.imread(str(images[0])).shape[:2]
    scale = VIEW_W / w0
    view_h = round(h0 * scale)
    W, H = VIEW_W + PANEL_W, view_h + CAPTION_H
    silent = OUT_DIR / f"{name}_silent.mp4"
    # H.264 through Windows Media Foundation (opencv's ffmpeg build has no H.264 encoder); mp4v elsewhere
    writer = cv2.VideoWriter(str(silent), cv2.CAP_MSMF, cv2.VideoWriter_fourcc(*"H264"), FPS_OUT, (W, H))
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), FPS_OUT, (W, H))
    alerts, rendered = [], []
    for k, rows in enumerate(frames):
        t = k * STEP_S
        img = cv2.imread(str(images[k]))
        depth = load_depth_map(depths[k])
        r = seg.predict(img, imgsz=640, device=args.device, verbose=False)[0]
        cls = cv2.resize(r.semantic_mask.data.cpu().numpy().astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST)
        objs = [ObjectRef(str(row["track_id"]), (float(row["bbox_x1"]) + float(row["bbox_x2"])) / 2 / w0,
                          float(row["distance_m"]), max(0.005, (float(row["bbox_x2"]) - float(row["bbox_x1"])) / 2 / w0))
                for row in rows if row.get("bbox_x1") not in (None, "") and row.get("distance_m") not in (None, "")
                and not str(row["track_id"]).startswith("obs_")]
        sf = scene_frame(cls[::DOWNSAMPLE, ::DOWNSAMPLE], depth, objs)
        edge_alert = edges.step(t, sf.edge)
        f = engine.step(k, t, rows, poses[min(STRIDE * k, len(poses) - 1)], contexts=sf.contexts, edge=edge_alert,
                        surface=surfaces.step(sf.walker_surface))
        a = f.announcement
        if a is not None:
            tte = truth.get((k, a.track_id), -1)
            alerts.append({"t": round(t, 2), "frame": k, "kind": a.kind, "track": a.track_id, "class": a.class_name,
                           "text": narrate(f.narrator_event()),
                           "bearing_deg": round(f.narrator_event().bearing_deg, 1), "truth": a.kind if a.kind in ("edge", "surface") else
                           "true" if tte >= 1 else "late" if tte == 0 else "false"})

        # ── draw ──
        view = cv2.resize(img, (VIEW_W, view_h), interpolation=cv2.INTER_AREA)
        cls_v = cv2.resize(cls, (VIEW_W, view_h), interpolation=cv2.INTER_NEAREST)
        tint = view.copy()
        for c, col in GROUND_TINT.items():
            tint[cls_v == c] = col
        view = cv2.addWeighted(tint, 0.35, view, 0.65, 0)
        ranked = {rt.track.track_id: rt for rt in f.ranked}
        overrides = {o.track.track_id for o in f.overrides}
        for row in rows:
            tid = str(row["track_id"])
            if tid.startswith("obs_") or row.get("bbox_x1") in (None, ""):
                continue
            x1, y1, x2, y2 = (int(float(row[c]) * scale) for c in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"))
            colour = RED if tid in overrides else ORANGE if tid in ranked else GREY
            said = a is not None and a.track_id == tid
            cv2.rectangle(view, (x1, y1), (x2, y2), colour, 4 if said or tid in overrides else 2)
            tte = truth.get((k, tid))
            if tte is not None and tte >= 1:
                dashed_rect(view, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), (80, 220, 80), 2)
            d = row.get("distance_m")
            label = f"{row['class']} {float(d):.1f}m" if d not in (None, "") else row["class"]
            cv2.putText(view, label, (x1 + 2, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2, cv2.LINE_AA)
        if sf.edge is not None and sf.edge.kind in (CURB, STAIRS, ROAD):
            cv2.putText(view, f"edge ahead: {CPE_CLASSES[sf.edge.kind]} {sf.edge.distance_m:.1f} m",
                        (VIEW_W // 2 - 120, view_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)

        canvas = np.full((H, W, 3), 24, np.uint8)
        canvas[:view_h, :VIEW_W] = view
        x0 = VIEW_W + 14

        def text(s, y, size=0.5, colour=WHITE, bold=1):
            cv2.putText(canvas, s, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, size, colour, bold, cv2.LINE_AA)

        text(args.title or "CPE v2 demo", 28, 0.62, WHITE, 2)
        text(f"SANPO test walk {sid[:10]}  t = {t:4.1f} s", 52, 0.45, GREY)
        text(f"tracked {len(f.tracks)}   ranked {len(f.ranked)}   overrides {len(f.overrides)}", 76, 0.45, GREY)
        walker = CPE_CLASSES.get(sf.walker_surface, "?") if sf.walker_surface is not None else "?"
        text(f"walking on: {walker}", 100, 0.5)
        text("top threats  (P arrive 4 s, priority)", 134, 0.48, ORANGE)
        y = 158
        for rt in f.ranked[:5]:
            ctx = sf.contexts.get(rt.track.track_id)
            flags = "".join(s for s, on in (("path ", ctx and ctx.in_path), ("edge ", ctx and ctx.near_boundary),
                                            ("other ", ctx and ctx.other_surface)) if on)
            text(f"{rt.track.class_name} #{rt.track.track_id}  {rt.arrival.p_arrive:.2f}  {rt.priority:.2f}", y, 0.47)
            text(f"   {rt.track.distance_m:.1f} m  x{context_multiplier(ctx):.2g} {flags}", y + 18, 0.42, GREY)
            y += 42
        if not f.ranked:
            text("none above threshold", y, 0.45, GREY)
        y = view_h - 92
        for colour, s in ((RED, "safety override (TTC <= 1 s)"), (ORANGE, "ranked threat"),
                          (GREY, "tracked, not a threat"), ((80, 220, 80), "- - really enters path <= 4 s")):
            cv2.rectangle(canvas, (x0, y - 10), (x0 + 14, y + 2), colour, -1)
            cv2.putText(canvas, s, (x0 + 22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREY, 1, cv2.LINE_AA)
            y += 20
        rendered.append(canvas)
        print(f"  frame {k + 1}/{len(frames)}", end="\r", flush=True)

    duration = len(frames) * STEP_S
    clips = tts_clips(sorted({x["text"] for x in alerts}), OUT_DIR / f"_{name}_tts")
    track, heard = schedule_audio(alerts, clips, duration)
    for k, canvas in enumerate(rendered):                       # caption: what is being said now
        t = k * STEP_S
        now = [h for h in heard if h["spoken_at"] - 1e-6 <= t < max(h["spoken_until"], h["spoken_at"] + 1.0)]
        if now:
            h = now[-1]
            colour = {"override": RED, "edge": (240, 200, 60), "surface": (120, 210, 120)}.get(h["kind"], ORANGE)
            cv2.putText(canvas, f'"{h["text"]}"', (20, canvas.shape[0] - 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2,
                        cv2.LINE_AA)
        last = [h for h in heard if h["spoken_at"] <= t]
        cv2.putText(canvas, f"alerts so far: {len(last)}", (20, canvas.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    GREY, 1, cv2.LINE_AA)
        for _ in range(FPS_OUT // 5):
            writer.write(canvas)
    writer.release()
    wav = OUT_DIR / f"{name}.wav"
    write_wav(wav, track[: int(duration * SAMPLE_RATE)])
    shutil.rmtree(OUT_DIR / f"_{name}_tts", ignore_errors=True)
    (OUT_DIR / f"{name}.json").write_text(json.dumps({"session": sid, "frames": len(frames), "threshold": args.threshold,
                                                      "alerts": alerts, "heard": heard}, indent=1), encoding="utf-8")
    final = OUT_DIR / f"{name}.mp4"
    ok = mux(silent.resolve(), wav.resolve(), final.resolve())
    n_true = sum(x["truth"] == "true" for x in alerts)
    print(f"\n{len(frames)} frames, {len(alerts)} alerts ({n_true} before a real arrival, "
          f"{sum(x['kind'] == 'edge' for x in alerts)} edge, {sum(x['kind'] == 'surface' for x in alerts)} surface), "
          f"{len(heard)} spoken ({time.time() - started:.0f} s)")
    print(f"video: {(final if ok else silent).resolve()}")


if __name__ == "__main__":
    main()
