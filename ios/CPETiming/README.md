# CPE Timing: on-device latency harness (iPhone 17 Pro Max)

A minimal SwiftUI app that times CPE's two learned models on the phone, per stage, under the load the
real app would add. It is a **latency benchmark, not the navigation app**:
- It runs YOLO26n v3 on bundled SANPO frames, and the arrival GRU on recorded SANPO features
  (built from SANPO's depth, as is).
- It doesn't use the phone's camera or LiDAR (the product's depth comes from an external LiDAR),
  and it doesn't speak.

It exists because "fits the 50 ms alert budget" is the one headline claim with no measurement yet
(`reports/CPE v2 next steps.md`, move 7). No one on the team has a Mac, so the app is built on a
GitHub Actions macOS runner and installed through TestFlight.

## What it measures

Per frame it writes one CSV row: `pre_ms` (JPEG decode + resize), `det_ms` (detector),
`det_post_ms` (reading the detector output), `gru_ms` and `total_ms`, plus the thermal state and
battery level. Each stage is also an `os_signpost` interval in Instruments'
Points of Interest.

- **Benchmark run:** a warm-up block (60 frames, excluded from analysis), then 10 blocks of 150
  frames paced at 15 FPS.
- **Sustained run:** 5–40 min (default 30) at 15 FPS. This exposes thermal throttling: report the p95
  per 5-minute window.

Models: `CPEDetector` is YOLO26n v3 FP16 Core ML from `training/scripts/export_yolo26n_edge.py`.
`CPEArrivalGRU` is `corridor_h8_seed0` FP16 from `training/scripts/export_gru_coreml.py`, softmax
inside the graph, fixed shapes [1, 32, 8, 9].

## Build

Everything in `Resources/` and `Models/` is generated (git-ignored):

```bash
python ios/CPETiming/prepare_bundle.py --frames 300 --export    # macOS: the export needs coremltools' native writer
brew install xcodegen && (cd ios/CPETiming && xcodegen generate)
open ios/CPETiming/CPETiming.xcodeproj
```

Without a Mac, run the **ios-timing** workflow (Actions tab → ios-timing → Run workflow). It always
does the following on a macOS runner:
1. Prepares the bundle.
2. Exports both models. The GRU export checks Core ML against PyTorch; the max |Δp| is in the
   `coreml-exports` artifact.
3. Builds the app unsigned.

Tick **testflight** to also archive, sign and upload.

## Run it on the iPhone with a free Apple ID (no Mac, no paid account)

Every workflow run also uploads `CPETiming-unsigned-ipa` (Actions → the latest green **ios-timing**
run → Artifacts). To install it from a Windows PC:

1. Install iTunes (the Apple website version, not the Microsoft Store one) and
   [Sideloadly](https://sideloadly.io). Plug in the iPhone and tap **Trust**.
2. Unzip the artifact, drag `CPETiming-unsigned.ipa` into Sideloadly, enter the Apple ID, Start.
3. On the phone: Settings → Privacy & Security → **Developer Mode** on (restarts), then Settings →
   General → VPN & Device Management → trust the Apple ID's developer profile.

Free-account installs expire after 7 days (re-install the same way). The timing is the same as a
TestFlight build: both are Release builds of the same app.

## Run it on an iPhone 17 Pro Max via TestFlight

One-time setup by the account holder (needs a paid Apple Developer Program membership, $99/yr):

1. In App Store Connect, create the app record with bundle ID `edu.pes.cpe.timing`, or your own. If
   you use your own, set it as the repository variable `CPE_BUNDLE_ID`.
2. In App Store Connect → Users and Access → Integrations, create an **API key** with the *Admin*
   role. Admin is needed so Xcode can create the distribution certificate and profile itself.
3. Add repository secrets (Settings → Secrets and variables → Actions):
   - `APPSTORE_KEY_ID` and `APPSTORE_ISSUER_ID`: shown next to the key.
   - `APPSTORE_KEY_P8_BASE64`: run `base64 -i AuthKey_XXXX.p8` and paste the output.
   - `CPE_TEAM_ID`: your 10-character team ID.
4. If the default runner's Xcode is too old for App Store uploads (Xcode 26 is required since April
   2026), set the repository variable `MACOS_RUNNER` to a runner image that has it.
5. Start a TestFlight build by pushing a tag: `git tag ios-testflight-1 && git push origin ios-testflight-1`
   (next time `-2`, and so on). The **Run workflow** button with **testflight** ticked works too, but
   GitHub only shows it once the workflow is on the default branch (`main`). Processing in App Store
   Connect takes 5–30 min.
6. In App Store Connect → TestFlight, add the phone owner as an internal tester. They install
   **TestFlight** from the App Store and accept the invite.

On the phone:

1. Charge to > 80 %, then unplug. Close other apps, keep the phone at room temperature, case off.
2. Open **CPE Timing**. It needs no permissions.
3. Run **Benchmark** twice. Then run **Sustained** for 30 min without touching the phone; the screen
   stays on.
4. Share the CSVs (the share button, or Files → On My iPhone → CPE Timing) and commit them under
   `evaluation/benchmarks/ondevice_latency/<device>_<date>/raw/`.

Then:

```bash
python ios/CPETiming/analyze_timing.py evaluation/benchmarks/ondevice_latency/<dir>/raw/*.csv \
    --out evaluation/benchmarks/ondevice_latency/<dir>
python tools/replay_threat_engine.py --gru-run models/GRU/GRU/runs/corridor_h8_seed0 \
    --latency-csv evaluation/benchmarks/ondevice_latency/<dir>/raw/cpe_timing_sustained30min_*.csv
```

Report median, p95 and p99 per stage, the p95 per 5-minute window, the
thermal timeline and battery drop per hour. Also report the latency-aware replay. State the
following as limitations:
- These are isolated per-frame timings of two models, not the whole app.
- SANPO frames are 1280×720 from a different camera; the phone's camera has a different field of
  view.
- Segmentation isn't included yet.
