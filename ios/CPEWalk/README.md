# CPE Walk: the hazard-alert app on the iPhone

The whole CPE v2 pipeline running live on the phone. The camera, LiDAR and ARKit tracking stand in
for SANPO's recorded frames, depth and camera poses. Everything runs on device, offline, with no
language model.

```
ARKit, phone held upright at chest height:
  camera image + LiDAR scene depth (confidence ≥ medium) + camera pose (world tracking)
→ Stage 1        YOLO26n v3 (Core ML, FP16) → per-class NMS → IoU tracker → LiDAR depth per box
                 (central 20% patch, closer boxes masked), least-squares closing speed, pinhole bearing
→ Segmentation   YOLO26n-sem (Core ML) every 2nd frame: walker surface, edge ahead, per-object context
→ Threat engine  Kalman fusion with ego-motion removed (fitted noise, Student-t update) → safety
                 envelope (TTC ≤ 1 s in the corridor) → arrival GRU (Core ML) → priority
                 P(arrive) × severity^γ × context → announcement manager
                 (override › "Curb ahead" › object threat › surface reminder)
→ Speech         on-device voice, panned toward the object's bearing; overrides cut in
```

Processing runs at 5 Hz, the replay's rate. The Swift engine is a line-by-line port of
`src/threat_engine/` and `src/segmentation/`. Every constant comes from `AppData/cpe_config.json`,
which `prepare_app.py` writes from the Python modules, so the two can't drift. A unit test replays a
recorded SANPO test walk through the Swift engine and compares the fused tracks, GRU inputs,
P(arrive), ranking and phrases with the Python engine (`Tests/GoldenSessionTests.swift`).

## Build and run on your iPhone (free Apple ID, Developer Mode)

Needs a Mac with Xcode (26 or later), the iPhone and a USB cable. No paid developer account.

1. One-time Mac setup:
   ```bash
   sudo xcodebuild -license accept     # in Terminal: sudo needs a password prompt
   xcodebuild -runFirstLaunch          # installs Xcode's system components
   brew install xcodegen
   ```
2. Generate the models, config and project (from the repo root):
   ```bash
   .venv-seg.nosync/bin/python ios/CPEWalk/prepare_app.py
   cd ios/CPEWalk && CPE_BUNDLE_ID=edu.pes.cpe.walk.$USER xcodegen generate && open CPEWalk.xcodeproj
   ```
3. In Xcode: Settings → Accounts → add your Apple ID. Select the **CPEWalk** target → Signing &
   Capabilities → Team: *your name (Personal Team)*. If Xcode says the bundle ID is taken, change it
   to something unique.
4. On the iPhone: plug it in and tap **Trust**. Then Settings → Privacy & Security → **Developer
   Mode** → on. The phone restarts; confirm Developer Mode.
5. In Xcode pick the iPhone as the run destination and press **Run** (⌘R). The first time, the phone
   asks you to trust the developer: Settings → General → VPN & Device Management → your Apple ID →
   Trust. Run again.

A free-account install works for 7 days; after that, press Run again. Without Xcode attached, the
app keeps working until then.

### Tests (simulator, no phone needed)

```bash
xcodebuild -downloadPlatform iOS       # once, ~8 GB: the simulator runtime
cd ios/CPEWalk && xcodebuild test -scheme CPEWalk -destination 'platform=iOS Simulator,name=iPhone 18 Pro Max' \
  -derivedDataPath ~/Library/Developer/Xcode/DerivedData/CPEWalk-cli
```

Keep the build folder outside the repo: the repo sits in iCloud-synced Documents, and iCloud's file
attributes make code signing fail ("resource fork, Finder information, or similar detritus not
allowed"). Xcode's own default DerivedData location is fine.

Result on 2026-10-26 (Xcode 27.1, iOS 27.0 simulator), 120 frames of a SANPO test walk: fused tracks
within 2e-14 of `fusion.py`, GRU inputs identical, max |Δ P(arrive)| 0.003 (FP16 Core ML), ranking
identical on every frame, 18/18 announcements word for word.

## Using it

- Hold the phone upright at chest height, camera facing forward (a chest strap or lanyard mount
  works best). Press **Start**. It says "C P E started".
- Alerts are short phrases: "Stop! Car very close ahead." (safety override), "Person left,
  5 meters.", "Curb ahead, 3 meters.", "Now on the road. Careful.". With nothing around, it says
  your surface every ~5 s ("Walking on the sidewalk."); switch that off with *Surface reminders
  when quiet*.
- On a flight of stairs it says "Going up stairs." or "Going down stairs. Careful." within about a
  second of starting, and "End of stairs." about a second after the last step. This comes from the
  phone's height, so it works even when the camera can't see the steps.
- Each phrase comes from the direction of the object: use headphones (bone-conduction keeps your
  ears open).
- *Show camera overlay* draws the boxes for sighted testers: red = override, orange = ranked
  threat, grey = tracked.

## Differences from the replay (state them when reporting results)

- **Depth:** iPhone LiDAR. It is reliable to ~5 m; beyond that ARKit's fused depth is an estimate,
  and low-confidence pixels are dropped. SANPO's stereo depth reaches further.
- **Pose:** ARKit world tracking instead of `fixed_camera_poses.csv`. If the phone points at the
  floor or sky, the last good pose is kept.
- **Tracker:** greedy IoU association with constant-velocity box prediction, standing in for
  Ultralytics' ByteTrack. Stage 1 runs at conf 0.30, so ByteTrack's low-score second pass rarely
  matters. The depth-guided rescoring pass of `yolo_tracker.py` is not ported; the box sanity
  filter (`is_valid_detection`) is.
- **Mounting:** chest, not SANPO's head camera. The GRU was trained on head-camera tracks.
- **Detector recall:** the detector finds people well but few poles and vehicles on SANPO-like
  views (methodology §4.21–4.22). The app inherits that.
- **Phone-only rules** (methodology §4.25), which the replay does not use:
  - the road bias on the segmenter's logits;
  - "Going up/down stairs." and "End of stairs." from ARKit's height;
  - a car narrower than 1.2 m is called a motorcycle.
- **Trained on US/Japan streets:** on Indian roads expect more road-called-sidewalk and
  scooter/autorickshaw mistakes until the models are fine-tuned on Indian data
  (`docs/pending_work.md` 2a).

This is a research prototype, not a mobility aid. Keep using a cane or guide dog.
