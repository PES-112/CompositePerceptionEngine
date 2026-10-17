# On-device deployment (iPhone 17 Pro Max), SANPO-vs-LiDAR depth realism, and the SANPO camera-pose mismatch

Scope: (1) how to measure CPE alert-path latency on a borrowed iPhone 17 Pro Max credibly and cheaply; (2) how to bound the realism gap between SANPO depth and iPhone LiDAR/ARKit depth; (3) causes and fixes for the SANPO pose mismatch (docs/methodology.md §4.13, docs/architecture.md §9 item 5). Researched 2026-10-14. Part 3 includes my own measurements on the cached SANPO pose files (`data/sanpo_poses/`, 112 sessions) plus 12 `fixed_camera_poses.csv` files and one `description.json` fetched from the public bucket. Analysis code was run ad hoc in the session scratchpad; nothing in the repo was changed.

## 1. Measuring latency on the iPhone credibly and cheaply

### Takeaway
Ultralytics' own on-device numbers for YOLO26n on an iPhone 17 Pro (A19 Pro) put detect at about 3–5 ms of model time in isolation, ~7 ms of model time inside a live camera app, and ~11–16 ms per frame end to end. FP16 and INT8 are within noise, and the NMS-free end2end head is kept because it wins on the device. A 50 ms p95 alert path is therefore very likely achievable, but CPE still needs its own number. The cheapest credible path with no Mac is to build a small instrumented benchmark app on a free GitHub Actions macOS runner, either by forking Ultralytics' open-source yolo-ios-app or with a minimal SwiftUI harness. Install it on the friend's phone via TestFlight (needs a paid Apple Developer account), then log per-frame pre/inference/post times on SANPO frames over a sustained run of at least 20 minutes. The Xcode Core ML Performance Report is the "gold standard" for isolated model time, but it needs a Mac with the phone physically connected.

### Cited Findings
**Published on-device numbers (YOLO26n, iPhone 17 Pro, A19 Pro, 16-core Neural Engine):**
- Ultralytics' yolo-ios-app performance doc uses an iPhone 17 Pro (12 GB, A19 Pro, 6-core CPU, 6-core GPU, 16-core Neural Engine) as its "ground-truth device". The Core ML table ran on iOS 26.5.2. An Apple M4 Pro host running coremltools is used "for relative screening only". The doc warns that host results "predict the wrong winner — always confirm on device" — [Ultralytics yolo-ios-app docs/performance.md](https://github.com/ultralytics/yolo-ios-app/blob/main/docs/performance.md)
- Core ML, YOLO26n INT8, single-image burst (mean of 15 runs after 3 warm-ups on bus.jpg), `.cpuOnly` vs `.cpuAndNeuralEngine`:
  - Detect 640: 9.2 vs **3.2 ms**
  - Segment 640: 12.6 vs 4.8 ms
  - Semantic 640: 9.7 vs 4.6 ms
  - Depth 640: 25.0 vs 5.3 ms
  - Classify 224: 2.2 vs 1.9 ms

  Source: [yolo-ios-app performance.md](https://github.com/ultralytics/yolo-ios-app/blob/main/docs/performance.md); the same numbers appear on [Ultralytics CoreML integration docs](https://docs.ultralytics.com/integrations/coreml).
- iOS 27.0, Release build, Core ML YOLO26n detect with hardware acceleration (median of interleaved rounds, two cooled launches) — [performance.md](https://github.com/ultralytics/yolo-ios-app/blob/main/docs/performance.md):
  - INT8 end2end: 4.59 ms
  - FP16 end2end: 4.05 ms
  - FP16 raw head: 4.17 ms (3.58 ms inference + 0.58 ms NMS post)
  - CPU-only: about 10.3–11.5 ms
  - Run-to-run spread was a median of 3% and a worst case of 17%, so "gaps under ~10% are ties".
- **FP16 vs INT8:** "Core ML FP16 and INT8 are within noise on this device; INT8 only halves the download." INT8 weight palettization gives "no latency change" (the ANE computes in fp16), and INT8 activation quantization gives no benefit. Setting `minimum_deployment_target` to iOS 17/18+ "regresses detect latency (adds casts / CPU ops)", so the doc leaves it unset — [performance.md](https://github.com/ultralytics/yolo-ios-app/blob/main/docs/performance.md)
- **NMS-free (end2end) head:**
  - On the Mac host the end2end head is ~9% slower for n (2.60 vs 2.26 ms; a fixed ~0.25 ms CPU decode and ~19 extra CPU ops).
  - On the device (Xcode Performance Report, A19 Pro), end2end had a median of 1.81 ms vs 1.90 ms for legacy + NMS. "On device the sign flips and end2end is kept."
  - A legacy raw head plus Swift NMS adds ≈2.4 ms per frame.

  Source: [performance.md](https://github.com/ultralytics/yolo-ios-app/blob/main/docs/performance.md)
- **YOLO26n vs YOLO11n:** equal backbone latency on the host NE (2.26 vs 2.25 ms). YOLO26n is smaller (2.71 vs 2.89 MB). There is no device table for YOLO11n — [performance.md](https://github.com/ultralytics/yolo-ios-app/blob/main/docs/performance.md)

**In-app vs isolated latency (this is where p95 risk lives):**
- In-app model time was ≈7 ms vs ≈1.8 ms in the isolated Xcode Performance Report; the doc attributes this to thermal effects and live-pipeline contention — [performance.md](https://github.com/ultralytics/yolo-ios-app/blob/main/docs/performance.md)
- Full camera-to-result frame time for detect — [performance.md](https://github.com/ultralytics/yolo-ios-app/blob/main/docs/performance.md):
  - 15.9 ms with the `.photo` preset (Vision preprocessing ≈8 ms, inference ≈7 ms)
  - 11.3 ms for 720p preview with 360×640 inference (the shipped configuration)
  - 8.3 ms for VGA with manual vImage preprocessing
- Raw per-frame detect latency on the device:
  - Manual preprocessing + `.cpuAndNeuralEngine`: min/p90/max 10.7/15.0/18.5 ms
  - Vision + `.cpuAndNeuralEngine`: 13.2/17.3/17.8 ms

  Source: [performance.md](https://github.com/ultralytics/yolo-ios-app/blob/main/docs/performance.md)
- Core ML compiles for ~1–2 s on every launch; report warm-up separately. A 22,000-inference soak test was stable — [performance.md](https://github.com/ultralytics/yolo-ios-app/blob/main/docs/performance.md)
- Apple's Core AI (successor runtime, iOS 27) halves raw model time (detect 2.05 ms) but adds ~1.2 ms of CPU preprocessing per frame. The Core ML backend remains the fallback — [yolo-ios-app issue #320](https://github.com/ultralytics/yolo-ios-app/issues/320); [performance.md](https://github.com/ultralytics/yolo-ios-app/blob/main/docs/performance.md)

**Thermal behaviour, iPhone 17 Pro Max:**
- Apple claims the vapor chamber plus aluminium unibody gives 40% better sustained performance than the iPhone 16 Pro — [Tom's Guide](https://www.tomsguide.com/phones/iphones/iphone-17-pro-max-endurance-tested-does-the-new-design-actually-improve-performance)
- Tom's Guide ran 3DMark Wild Life Extreme Stress twice for 20 minutes each (a GPU load, not an ANE load). The iPhone 17 Pro Max's stability score was 65.2% after 20 min and 68.4% after 40 min, at ~100.7°F — [Tom's Guide](https://www.tomsguide.com/phones/iphones/iphone-17-pro-max-endurance-tested-does-the-new-design-actually-improve-performance)
- In gaming tests the 17 Pro held 115–120 fps for nearly 45 minutes and retained over 89% of its score after 90 minutes — [iGeeksBlog](https://www.igeeksblog.com/iphone-17-pro-gaming-review/). This is single-reviewer data.
- Benchmark harness advice: "Phones throttle quickly — let the device cool between runs". The first Core ML run compiles the model. iOS caps app memory at ~3.3 GB on a 12 GB iPhone 17 Pro Max — [lucario6607/Lc0-ios](https://github.com/lucario6607/Lc0-ios)
- On Apple's forums, the Xcode Performance tab was much faster than in-app Instruments timings. A reply suggested the tab runs far fewer predictions, so it avoids the thermal throttling that skews 10K-run loops — [Apple Developer Forums 765772](https://developer.apple.com/forums/thread/765772)

**Measurement tooling:**
- Xcode's Core ML Performance Report (Xcode 14+) runs on "any connected device". It reports median prediction, load and compile times and the per-layer compute unit. Xcode 16 added per-op estimated time and reasons for unsupported ops. The device must be unlocked — [WWDC22 10027](https://developer.apple.com/videos/play/wwdc2022/10027/); [WWDC24 10161](https://developer.apple.com/videos/play/wwdc2024/10161/)
- Known bugs (Xcode 16 / iOS 18): the report sometimes crashes on an iPhone with ANE/GPU compute units, or returns 0 ms times — [Apple Developer Forums 767412](https://developer.apple.com/forums/thread/767412)
- Ultralytics' instrumented app build logs `[perf]` pre/inf/post per frame (via `xcrun devicectl ... --console` and `os_log`). Compute units and camera preset are switchable through environment variables — [performance.md](https://github.com/ultralytics/yolo-ios-app/blob/main/docs/performance.md)
- **No-Mac build path:**
  - Several public repos build iOS apps on GitHub Actions macOS runners, which are free for public repos, and upload to TestFlight with App Store Connect API-key signing. Signing for TestFlight needs a paid Apple Developer account.
  - Since April 2026 uploads require Xcode 26 / iOS 26 SDK (a `macos-26` runner).

  Sources: [longle7/NextTrain PR #60](https://github.com/longle7/NextTrain/pull/60); [JW-Incorporated/foray #38](https://github.com/JW-Incorporated/foray/issues/38); [minemurak/headspeed PR #1](https://github.com/minemurak/headspeed/pull/1)
- A Core ML benchmark app (LeelaBench, lc0 chess engine) is built entirely on GitHub Actions. It offers CPU+NE / CPU+GPU / All selection and shows thermal state — [lucario6607/Lc0-ios](https://github.com/lucario6607/Lc0-ios)
- MobAI ios-builder wraps GitHub Actions / Codemagic / Bitrise for no-Mac iOS builds, including TestFlight upload or OTA install via QR code — [MobAI-App/ios-builder](https://github.com/MobAI-App/ios-builder)

### Inferences
**Budget estimate (inference from the published figures, not a measurement).** On the A19 Pro the published component costs are:
- YOLO26n detect: 4–7 ms of model time in-app
- A 31k-parameter GRU on CPU: well under 1 ms
- A YOLO26n-seg-class segmenter at 2–5 Hz async: ~5–6 ms per call on the ANE
- ARKit sceneDepth handling: 16.4 ms per frame on an iPhone 13 Pro, §2

The 50 ms p95 alert path should have at least 2× headroom. The risk sits in camera/ARKit pipeline contention and thermals, not in the networks.

**Recommended cheapest credible protocol.**
- **Option A (zero build effort).** Install the public Ultralytics YOLO iOS app on the friend's iPhone and record its on-screen latency or FPS for YOLO26n detect. This proxies the architecture only; it is not CPE's 17-class model.
- **Option B (CPE-specific, the one to report).** Use a public GitHub repo holding a minimal SwiftUI app:
  - Bundle the CPE `yolo26n.mlpackage` (end2end head, FP16 or INT8, no deployment target), the GRU as Core ML (or plain Swift/Accelerate), and ~300–1000 SANPO frames as JPEG.
  - Build on a GitHub Actions macOS runner and distribute via TestFlight. This needs a $99/yr developer account, or access to someone's account.
- **Option B's harness:**
  - Time every stage with `CACurrentMediaTime`/`os_signpost` and write a CSV to the app's Documents folder.
  - Use a warm-up block, then three interleaved cold/warm blocks, then a 20–40 minute sustained loop at 15 FPS pacing.
  - Log `ProcessInfo.thermalState` per frame.
  - Report median, p95, p99, p95 per 5-minute window and the thermal-state timeline.
- **What to report.** Report the in-app number, since it includes contention. The isolated model number can go in a footnote if a Mac ever becomes available. Ultralytics' finding that isolated time is ~4× lower than in-app time shows why the in-app number is the defensible one.
- **To include ARKit cost.** Run the loop while an `ARSession` with `.sceneDepth` is active, even if frames come from disk. This adds the LiDAR/VIO load to the thermal and contention picture.
- **Cloud Macs.** Cloud Macs (MacinCloud, GitHub runners) can build, but they cannot produce the Xcode Performance Report on a phone that is not physically connected or paired to them. They do not replace on-device timing.
- **Proxies to avoid.** Geekbench AI or AI-Benchmark scores are not a credible substitute for model-specific latency. At most they can show the borrowed unit is healthy.

### Gaps
- I found no independent (non-Ultralytics) measurement of YOLO26n on an iPhone 17 Pro **Max** specifically. The Pro and Pro Max share the A19 Pro; the Max has a larger chassis that may sustain slightly better.
- I found no published ANE-specific sustained-throttling curve (all thermal tests are GPU or gaming loads).
- Free-Apple-ID sideloading from Windows (e.g. Sideloadly or AltStore with 7-day certificates) was not researched or verified here. It could avoid the $99 account but should be checked before relying on it.
- I did not retrieve Geekbench AI numbers.

## 2. Bounding the realism gap: SANPO depth vs iPhone LiDAR (ARKit sceneDepth)

### Takeaway
ARKit `sceneDepth` is a 256×192 metric depth map with a per-pixel confidence (low/medium/high), fused from LiDAR plus RGB/VIO. It is accurate to about 1–2 cm at the centre within 1–3 m indoors, but errors at the image periphery reach 5–28 cm. Its usable range is ~4–5 m (6.4 m at most). It is reported at the camera rate, but the depth actually updates at ~15 Hz.

SANPO's `depth_maps` are dense CREStereo depth at 1280×720 up to 80 m. Its 15 FPS matches the LiDAR's effective update rate, so the gap is mainly range, resolution, peripheral error and dropouts. Simulate it by doing the following to SANPO depth, then rerunning the replay (the plan in architecture.md §4):
1. Crop to the iPhone aspect.
2. Downsample to 256×192.
3. Mask beyond 5 m, with 4 m and 3 m as sensitivity cases.
4. Add distance- and eccentricity-dependent noise.
5. Add random and edge dropouts.

There is no quantitative study of rear-LiDAR degradation in direct sun. Treat sunlight as a range-reduction sensitivity case and do not claim a number for it.

### Cited Findings
**ARKit sceneDepth format:**
- `ARDepthData` provides `depthMap` (metres) and `confidenceMap`, a UInt8 buffer at the same resolution with low/medium/high levels. Accuracy depends on surface reflectivity and absorbance. Apple sample code uses `frame.smoothedSceneDepth ?? frame.sceneDepth` — [WWDC20 Explore ARKit 4](https://developer.apple.com/videos/play/wwdc2020/10611/); [WWDC20 sample code mirror](https://github.com/artemnovichkov/wwdc20-samplecode/blob/master/CreatingAFogEffectUsingSceneDepth)
- ARKit gives developers a 256×192 depth map on every LiDAR iPhone — [Taylor & Francis 2026, lidar accuracy across iPhone models](https://www.tandfonline.com/doi/full/10.1080/2150704X.2026.2720055)
- In ARKitScenes the LiDAR depth is 256×192 uint16 mm with a uint8 0–2 confidence — [ARKitScenes arXiv 2111.08897](https://arxiv.org/pdf/2111.08897) (via search summary)

**Frame rate:**
- `sceneDepth` is delivered at 60 FPS, with 16.43 ± 0.60 ms processing time per image in the authors' app (iPhone 13 Pro). The raw `.builtInLiDARDepthCamera` stream is 320×240 at 24 FPS, with 6.14 ms processing — [PMC10939328, Accuracy and usability of smartphone distance estimation for visual assistive technology](https://pmc.ncbi.nlm.nih.gov/articles/PMC10939328/)
- Independent vibration-measurement work found the LiDAR depth map actually updates at 15 Hz although it is reported at the camera's 60 Hz. Its optimal phone-to-target range is 0.30–2.00 m, with higher standard deviation at object edges — [MDPI Sensors 23(18):7832](https://www.mdpi.com/1424-8220/23/18/7832)

**Accuracy vs distance (iPhone 13 Pro, indoor white door, 1/2/3 m, 30 readings each)** — [PMC10939328](https://pmc.ncbi.nlm.nih.gov/articles/PMC10939328/):

| Approach | Centre MAE at 1 / 2 / 3 m | Peripheral MAE |
|---|---|---|
| ARKit back (`sceneDepth`) | 1.37 / 0.48 / 1.40 cm | 7.08 and 4.75 cm at 1 m; 1.88 and 23.73 cm at 2 m; 7.21 and 20.78 cm at 3 m |
| Raw LiDAR | 1.36 / 0.87 / 0.81 cm | up to 27.87 cm at 3 m |

- All approaches were significantly worse at the periphery (p < .001).
- The authors conclude LiDAR/ARKit are "the best choice for assistive technology development when spatial accuracy is a top priority".
- The test was indoors only, with no lighting variation.

**Range:**
- Apple's figure is "up to 5 m" — [opencv.ai blog](https://www.opencv.ai/blog/depth-estimation); [arXiv 2310.14437](https://arxiv.org/pdf/2310.14437)
- Measured rear-sensor range was about 0.26 m minimum and about 6.42 m maximum (a temporal-consistency test, not an accuracy test) — [opencv.ai](https://www.opencv.ai/blog/depth-estimation)
- iPhone 12 Pro: reasonable accuracy up to 4 m; beyond that, readings varied click to click — [it-jim](https://www.it-jim.com/blog/iphones-12-pro-lidar-how-to-get-and-interpret-data/)
- An iPad Pro 2021 study vs a terrestrial laser scanner over 1–5 m found 2 m to be the ideal distance — [Taylor & Francis 2024](https://www.tandfonline.com/doi/full/10.1080/16874048.2024.2408839)

**Outdoor and sunlight:**
- Only blogs and vendor claims say direct sunlight reduces effectiveness (more solar IR):
  - [opencv.ai](https://www.opencv.ai/blog/depth-estimation)
  - "Shaded outdoor areas work; direct sun does not" — [Manifold vendor blog](https://www.scanmanifold.com/blog-posts/lidar-on-iphone-how-accurate-is-it-plus-the-biggest-errors-that-manifold-corrects)
  - [Leapp blog](https://leapp.nl/en/blogs/iphone/accuracy-lidar-scanner-iphone-what-you-need-to-know)
- An outdoor façade test (iPhone 12 Pro Max, 0.25–5 m) found RMSE 4.89 mm vs 3.44 mm for a terrestrial laser scanner, with errors rising slightly with range. Sun was not isolated as a variable — [ResearchGate, Abbas & Abed 2023](https://www.researchgate.net/publication/378351112_Evaluating_the_Accuracy_of_iPhone_Lidar_Sensor_for_Building_Facades_Conservation) (via search summary)
- Across iPhone models, a Zeb Horizon scanner's RMSEs were consistently a tenth of the iPhone Pro LiDAR's — [Taylor & Francis 2026](https://www.tandfonline.com/doi/full/10.1080/2150704X.2026.2720055) (via search summary)
- On ARKitScenes-type data, the ARKit confidence map averages ~1.63% low-confidence points — [arXiv 2310.14437](https://arxiv.org/pdf/2310.14437) (via search summary)

**SANPO depth (what the simulation currently uses):**
- Dense depth is CREStereo disparity converted to depth "using camera intrinsics". It is limited to 0–80 m, at 1280×720 "a result of CREStereo's architecture". A sparse ZED SDK depth is also provided. Video is 15 FPS and rectified with ZED software — [SANPO paper, arXiv 2309.12172v2](https://arxiv.org/html/2309.12172v2)
- Depth maps are in metres and relative to the left lens. `zed_depth_maps` is the sparse ZED API depth — [SANPO GitHub README](https://github.com/google-research-datasets/sanpo_dataset)
- The README says `.npz`, but the bucket listing for session 1rL8RTbI4jPqRkavZxrQbbM26iRfSLAH shows `camera_head/left/depth_maps/000000.float16.gz` (459 files = 459 video frames = 459 pose rows) — observed via the [GCS JSON listing](https://storage.googleapis.com/storage/v1/b/gresearch/o?prefix=sanpo_dataset/v0/sanpo-real/1rL8RTbI4jPqRkavZxrQbbM26iRfSLAH/camera_head/left/depth_maps/)

### Inferences
- **Depth semantics.** Disparity converted with focal length and baseline (Z = f·B/d) yields planar z-depth, not ray range. The code's assumption in `fusion.py` (z-depth, x = z·tanθ) is therefore consistent. ARKit `sceneDepth` is also generally treated as z-depth in the camera frame, but I did not find an Apple statement that settles this.
- **Frame-rate parity.** The LiDAR's true ~15 Hz update equals SANPO's 15 FPS. The replay's 0.2 s processed step (stride 3) is therefore not optimistic about depth refresh.
- **Suggested "LiDAR-ised SANPO" transform** (for `pending_work.md`'s optional experiment):
  1. **Field of view.** Centre-crop SANPO depth from 16:9 to 4:3 (1280×720 → 960×720) and downsample to 256×192 by block median. Note that SANPO's 51–60° HFOV differs from the iPhone wide camera's; keep the SANPO HFOV and just state it.
  2. **Range.** Set depth > R_max to invalid, with R_max ∈ {5 m (Apple spec), 4 m (it-jim), 3 m (sunlit sensitivity case)}.
  3. **Noise.** Gaussian noise with σ(d, r) = 0.01 m + k_d·d + k_r·r_norm²·d. Calibrate k_d so the centre error stays at about 1–2 cm through 3 m, and k_r so peripheral error reaches about 10–25 cm at 3 m (PMC numbers). Add extra noise at depth discontinuities (MDPI edge finding).
  4. **Dropout.** About 2% random low-confidence pixels plus edge pixels.
  5. **Temporal.** Hold each LiDAR map for 4 camera frames if replaying at 60 Hz. At SANPO's 15 FPS no hold is needed.

  Then recompute the Stage-1 box medians from the degraded map and rerun the held-out ranking and the system replay. Report the delta as the realism bound.
- **Label leakage.** Keep labels computed from the original SANPO depth (future arrival within 3 m). Only the model input should be degraded. This also partly addresses data-issue 6 (labels and inputs share one depth source).
- **Expected effect.** The 3 m arrival radius sits inside LiDAR range, so labels survive. Tracks will start later, because objects become measurable only once within ~5 m. At 1.4 m/s that is ~1.4 s of lead before reaching 3 m, versus the current 4 s horizon. Expect lower warning lead time and more "first seen already close" arrivals. This should be stated as the main realism cost.

### Gaps
- There is no peer-reviewed quantitative measurement of rear-LiDAR sceneDepth accuracy or range in direct sunlight vs shade.
- There is no per-pixel sceneDepth error-vs-distance curve beyond 3 m from a primary source; the ARKitScenes paper may have one, but I didn't retrieve it.
- I found no iPhone 17 Pro-specific LiDAR evaluation. The same 256×192 ARKit interface is reported across all LiDAR iPhones.
- I did not verify the iPhone 17 Pro Max main camera's HFOV or the ARKit capture resolution.

## 3. SANPO pose conventions and the "static objects move more in world frame" problem

### Takeaway
**Root cause found (from my own measurements on 12 sessions).** The two pose files differ in ways that each produce the reported symptom.
- **`camera_poses.csv` (the file the replay uses).** Its translations carry very large frame-to-frame jitter: path length is 2.5–25× the net displacement, with lateral and vertical steps of ±0.1–0.4 m per frame. Projecting box depth through these poses adds roughly walking-step-sized noise.
- **`fixed_camera_poses.csv`.** Its translations are smooth (path/net 1.02–1.2), but it uses a different camera convention: x right, y **down**, looking along **+z** (OpenCV), in a world whose up axis is +Z. `fusion.py` hard-codes the ZED convention (look along −z, up = +y). Applied to the fixed file, it points "forward" backwards, so a static object's world position moves by about 2× the walker's step. With 1.4 m/s × 0.2 s ≈ 0.28 m, that is ≈0.56 m, which matches the observed median of 0.52 m.

**Fix.** Use `fixed_camera_poses.csv` with `forward = R·(0,0,+1)`, `right = R·(1,0,0)` and world up = +Z (or the session mean of −R·(0,1,0)). Re-run the static-object check; the expected outcome is world-frame motion below camera-frame motion. A single sign flip alone would not fix the original file, because its problem is translation noise, not axes.

### Cited Findings (documentation)
- **README (2025-06-05 update):** the orientation of SANPO-Real camera poses "was not in sync with SANPO-Synthetic (right-handed, Y-up)"; corrected poses are in `fixed_camera_poses.csv`. The README gives no units, quaternion order, reference camera or tracking-state semantics — [SANPO GitHub README](https://github.com/google-research-datasets/sanpo_dataset)
- **SANPO paper:**
  - Real poses are "derived from fused IMU and VIO measurements provided by the ZED software". There is no statement on the reference frame, rate or accuracy.
  - The paper's 3D point-cloud figure says points from several seconds "are accumulated and aligned with ICP". Raw poses were apparently not sufficient on their own even for the authors' visualization.

  Source: [arXiv 2309.12172v2](https://arxiv.org/html/2309.12172v2)
- **SANPO paper on cameras:** the head camera is a "lightweight head-mounted ZED-M" and the chest camera a "ZED-2i, equipped with a 4mm lens". Both recorded at 15 FPS — [arXiv 2309.12172v2](https://arxiv.org/html/2309.12172v2)
  - Conflict: `camera_head/description.json` for session 1rL8RTbI… says `CAMERA_MODEL_ZED_2I`, fx 1909.7, HFOV 60.07°, stereo baseline 119.97 (ZED 2i ≈ 120 mm) — [description.json](https://storage.googleapis.com/gresearch/sanpo_dataset/v0/sanpo-real/1rL8RTbI4jPqRkavZxrQbbM26iRfSLAH/description.json)
  - The project audit also records all 462 `camera_head` sessions as ZED 2i.
  - So the data labelled "head" in the curated set is a ZED 2i with ~58° HFOV, not the ZED-M the paper describes.
- **ZED SDK:**
  - The pose is given for the **left eye** of the stereo camera. To get the centre, apply a half-baseline transform.
  - With `REFERENCE_FRAME::WORLD`, the world frame is where tracking started, oriented where the device was looking, unless an initial transform or area file is set. `REFERENCE_FRAME::CAMERA` gives the change since the last pose (odometry).
  - Camera and world frames share the coordinate system chosen at init, e.g. `RIGHT_HANDED_Y_UP`.

  Sources: [Stereolabs Coordinate Frames](https://www.stereolabs.com/docs/positional-tracking/coordinate-frames); [Using the Positional Tracking API](https://www.stereolabs.com/docs/positional-tracking/using-tracking); [ZED tutorial 4](https://github.com/stereolabs/zed-sdk/blob/master/tutorials/tutorial%204%20-%20positional%20tracking/README.md)
- **Consequences for CPE:**
  - The left-camera pose matches the left-lens depth maps, so the 6 cm half-baseline offset is irrelevant.
  - The original world frame is anchored at the first tracked frame. A per-session up axis is therefore expected when the IMU gravity alignment is not applied, which matches the team's observation.

### Cited Findings (my own measurements on public SANPO files; reproducible)
Sources:
- Cached `camera_poses.csv` files in `data/sanpo_poses/` (112 sessions; originally from `https://storage.googleapis.com/gresearch/sanpo_dataset/v0/sanpo-real/<sid>/camera_head/camera_poses.csv`)
- `fixed_camera_poses.csv` from the same paths for 12 sessions

All quaternions are read as (x, y, z, w) and converted to rotation R (camera-to-world), as in `fusion._quat_to_matrix`.

**Row alignment is fine.** For session 1rL8RTbI…, `camera_poses.csv` has 459 rows, `fixed_camera_poses.csv` 459, `left/video_frames` 459 PNGs and `left/depth_maps` 459 files. Shifting the fixed file ±1 row against the original does not improve agreement. The poses are one per left frame (observed via the [GCS listing](https://storage.googleapis.com/storage/v1/b/gresearch/o?prefix=sanpo_dataset/v0/sanpo-real/1rL8RTbI4jPqRkavZxrQbbM26iRfSLAH/camera_head/left/video_frames/)).

**`camera_poses.csv` translation jitter**, path length / net displacement over READY rows (original vs fixed, metres):

| Session | Original (path/net) | Fixed (path/net) |
|---|---|---|
| 2dVxdA3X | 38.2/8.5 | 9.4/8.5 |
| 3fNdc-1R | 39.8/17.6 | 18.0/17.6 |
| 5z7n1PIs | 44.3/1.8 | 11.4/1.8 |
| 6lm2fRHP | 68.9/14.3 | 15.4/14.3 |
| 8jLedz8b | 65.6/16.3 | 16.6/16.3 |
| AMWsWL1l | 42.8/7.9 | 9.6/7.9 |
| BZub4Nml | 74.2/13.7 | 14.0/13.7 |
| Ej_gHTCP | 68.9/24.7 | 25.5/24.7 |
| _4QWSKrm | 26.8/10.0 | 10.2/10.0 |
| h3TEaYCA | 59.5/25.8 | 26.9/25.8 |
| 1rL8RTbI | 134.0/34.7 | 35.8/34.7 |

- **Endpoints agree.** Net displacement is identical between the two files, so the endpoints match and only the per-frame path differs.
- **Example per-frame steps** (original, session 1rL8, rows 90–111, metres): (−0.11, 0.11, −0.09), (−0.16, −0.04, −0.12), (−0.06, −0.39, −0.18), (−0.30, 0.17, −0.18), (−0.30, 0.05, −0.10), and so on. The fixed file's steps over the same rows are (0.02, 0.13, −0.01), (0.01, 0.11, −0.01), …, i.e. steady ~0.11–0.13 m per frame (≈1.8 m/s) along one axis.
- **Jitter across all 112 cached original files:** over 1 s windows while moving (net > 0.5 m), path/net had a median of 1.79 and p90 of 3.83. A median of 3.7% of per-frame steps (p75 21%) exceed 0.3 m per frame, which implies more than 4.5 m/s.
- **Rotation is not the culprit in the original file.**
  - Its quaternions are near-identity: q_w ≈ 0.995–0.999, roughly a 10° head pitch.
  - Reading them as (x, y, z, w) or (w, x, y, z), transposed (world-to-camera) or conjugated, gives nearly the same camera axes (session-mean axis stability 0.98–1.0 for all variants).
  - This explains why "no axis sign flip fixes it".
  - With the original file, a 3-frame step expressed in the camera frame has a direction concentration (mean resultant length) of only ~0.47 under every convention, because the translation is noisy.
- **Fixed-file convention (all 10 sessions with walking; 2 near-stationary sessions excluded):**
  - Over 1 s windows, the 3D angle between the walking displacement and camera **+z** has a session median of 10–22°. Against camera −z it is 159–170°, so the camera looks along +z.
  - The session-mean world direction of camera +y is ≈ (0, 0, −1), with z-component −0.90 to −1.00 in every session. Camera +y therefore points down, and world +Z is up.
  - Net walking directions lie in the world XY plane (z-component |≤ 0.17|).

  The fixed file is thus OpenCV-camera (x right, y down, z forward) in a Z-up world. This contradicts the README's "right-handed, Y-up" wording, at least for the world frame. Trust the empirical check, not the README.
- **Fixed-file start.** The first READY rows of the fixed file sit exactly at (0, 0, 0) for ~3 rows while the original file already moves. The fixed file also uses NOT_READY for the first row. Treat leading rows as unreliable.
- **Walking vs heading.** The team's "walking direction vs camera heading ~30°" figure came from the noisy original translations. With the fixed file the 3D angle is 10–22°, which still includes the downward head pitch, so the horizontal angle is smaller.

### Inferences
**Why world-frame motion (0.52 m) exceeded camera-frame motion (0.32 m):**
- **(a) With `camera_poses.csv`.** Frame-level translation noise of ~0.1–0.4 m enters every world-frame position directly. Over a 3-frame step this adds noise of the same order as the 0.28 m walking step. Pose noise independent of measurement noise would need ≈0.41 m RMS to go from 0.32 to 0.52 (√(0.52² − 0.32²)), which is plausible given the observed jitter.
- **(b) With `fixed_camera_poses.csv` read through `EgoPose.from_sanpo`.** That method uses forward = R·(0,0,−1), and `sanpo_session_poses` uses up = mean R·(0,1,0).
  - Forward comes out reversed by 180°.
  - Up comes out as −Z: the same plane, but a flipped ground basis, which mirrors lateral.
  - A static object at camera depth z then maps to pos − f·z. As the walker advances s along f, its world estimate moves by 2s, i.e. ≈0.56 m per 0.2 s step at 1.4 m/s, matching the median of 0.52 m.
  - Flipping only one axis (e.g. forward sign without the matching right/up handling, or applying a flip to the noisy original file) would not cure both problems. This is consistent with "no frame offset or axis sign flip fixes it".
- **Recommended fix (concrete).** In `sanpo_session_poses`/`EgoPose.from_sanpo`, add a convention flag for the fixed file:
  - `right = R@(1,0,0)`
  - `forward = R@(0,0,1)`
  - `up = (0,0,1)`, or the session mean of `-R@(0,1,0)`
  - Ground-plane basis spanning world X–Y
  - Point `tools/replay_threat_engine.py` at `camera_head/fixed_camera_poses.csv`
- **Validation gates, in this order:**
  1. The median angle between the 1 s walking displacement (projected to the ground plane) and the projected forward vector is less than ~15° across sessions.
  2. For near (< 8 m) static-class tracks, the median per-step world-frame displacement is below the camera-frame displacement. A reasonable target is ≤ ~0.15–0.2 m, i.e. the box-depth noise floor, given σ_depth ≈ 0.05 + 0.01·d² m.
  3. Per-session sanity: the fixed-file path/net ratio is below 1.3. Flag sessions where it isn't, plus the ~2 near-stationary sessions per dozen.
- **If residual mismatch remains after the fix,** next suspects in order of likelihood:
  - Lateral position from bearing uses x = z·tanθ in the camera frame. This ignores head pitch and roll, which projects poorly onto the ground plane at ~10–15° pitch: a few-percent scale error, small.
  - Box-median depth jumping between object and background on thin or occluded objects (measurement noise, visible in camera frame too).
  - "Static" class membership, e.g. parked bicycles vs moving ones.
  - Holding NOT_READY rows: this repeats the last pose while the walker keeps moving, so static objects appear to approach in world frame during dropouts. Interpolating between the surrounding READY poses, or marking those frames "no pose" and falling back to camera-frame tracking, is better than holding.
- **Fallback (lower priority).** If the original `camera_poses.csv` must be used (e.g. for sessions lacking a fixed file), apply a zero-phase low-pass or constant-velocity smoother (e.g. a 1 s Savitzky–Golay filter) to its translations before use. Even then the fixed file is clearly the better source.
- **Paper framing.** The ego-motion ablation can probably move from "unvalidated" to "validated on SANPO" once gates 1–2 pass. Until then, keep it as an ablation, per methodology §4.13.

### Gaps
- No official SANPO document states the fixed file's camera convention, world up axis, quaternion order, or whether translations were re-estimated. My conclusions are empirical, from 12 sessions. Running the same check over all 462 sessions takes a few MB of CSV and is cheap.
- I could not determine why the original file's translations are so noisy. Possible causes include translation being expressed in a jittering frame, IMU-frame mixing, or VIO without loop smoothing; this is unconfirmed.
- I found no SANPO loader code in the public repo that reads `camera_poses.csv`. The README fetch showed only a `sanpo_dataset/` package folder, with contents not inspected.
- The conflict over which ZED model is on the head (paper: ZED-M; metadata: ZED 2i) is unresolved. It matters for HFOV and baseline but not for the pose fix.
- Whether SANPO `depth_maps` are z-depth or ray range is not stated anywhere. The z-depth reading is an inference from "disparity converted using intrinsics".
