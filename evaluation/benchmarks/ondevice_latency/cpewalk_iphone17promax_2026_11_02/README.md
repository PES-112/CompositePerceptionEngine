# CPEWalk in-app performance, iPhone 17 Pro Max, 2026-11-02

The live app on the street: ARKit camera + LiDAR, two detectors (v4 IDD + v3 legacy), the segmenter on
every 2nd frame, tracking, fusion, GRU, announcer and narration. Logged by
`ios/CPEWalk/Sources/App/PerfLog.swift`; per-file tables in `report.md` (`ios/CPETiming/analyze_timing.py`).

6 runs, 744 processed frames after the 10-frame warm-up of each run (2.5 min at 5 Hz).
The runs are short (1–53 s), so this is a first look, not a sustained-run result.

## Per stage, all runs pooled (ms)

| Stage | Median | p95 | p99 | Max |
|---|---:|---:|---:|---:|
| ARKit frame → input (rotation, LiDAR copy) | 6.5 | 8.2 | 8.6 | 14.3 |
| Detector 1, v4 IDD (incl. letterbox) | 19.6 | 23.7 | 24.5 | 25.5 |
| Detector 2, v3 legacy (incl. letterbox) | 17.9 | 20.3 | 21.4 | 23.9 |
| Both detectors | 37.9 | 43.4 | 44.5 | 49.4 |
| Tracking, depth per box | 0.5 | 1.2 | 1.8 | 3.3 |
| Engine + narration | 0.4 | 0.8 | 1.0 | 5.7 |
| Segmentation, on the frames it runs (373) | 86.7 | 98.4 | 102.9 | 114.1 |
| **Total, frames without segmentation** (371) | 40.2 | 43.9 | 45.6 | 46.6 |
| **Total, frames with segmentation** (373) | 136.0 | 149.3 | 153.8 | 168.5 |
| **Total, all frames** | 119.6 | 145.4 | 152.2 | 168.5 |

## What it shows

- **The app keeps its 5 Hz rate.** No frame took longer than the 200 ms between frames (max 169 ms), and the gap between processed frames is 0.20 s (max 0.20 s).
- **Alerts are later than the 50 ms alert-path budget.** 373 of 744 frames exceed it, which is every frame that runs the segmenter. The alert is spoken at the end of the frame, so on those frames it is heard ~136 ms (p95 149 ms) after the camera frame. On frames without segmentation it is ~40 ms.
- **Segmentation is the largest cost** (87 ms median when it runs), then the two detectors (38 ms together).
- **Each detector takes ~18–20 ms in the app vs 5.5 ms for the model alone in CPETiming.** The in-app time includes letterboxing the full camera image with Core Image. CPETiming measures that resize separately (`pre_ms`), so most of the difference is likely preprocessing rather than the model. This was not measured separately.
- **Thermal state was nominal on every frame.** That is expected for runs this short.
- **Battery: no claim.** The longest run is 53 s; the level did not move (0.95, and 1.00 on later runs; unplugged throughout).

Not measured: what happens over a sustained walk (10+ min), and what speech adds after the frame (synthesis and audio output).
