# Fix the geometry, then prove the forecaster

CPE's most publishable asset is a reproducible benchmark run on SANPO's official held-out split. For each tracked object it forecasts whether that object will reach the walker, and then budgets alerts against that forecast. None of the closest systems (Mobilio, WOAD, BlinkBud, the Pundlik collision-warning trial) publishes anything like it, because each was evaluated on private real-world runs. The headline result is fragile today. A one-line "nearest object in the cone" rule statistically ties the GRU (+0.013 AUROC, CI [−0.018, +0.046]). Lead time is about 0 s for every engine. The v1 vs v2 comparison sets two different operating points side by side, each chosen by an F1 score with no human-factors basis. All three weaknesses trace back to geometry, not to the model. At 3 m the ±30° "path" cone is about 3.5 m wide, nearly the whole 51–60° image, and the camera poses that would let CPE define a real walking corridor are being read with the wrong file and the wrong axis convention. The next four to six weeks should therefore go to the following, in order:

1. Fix the pose convention, which takes days.
2. Replace the cone with a body-width corridor and a closest-approach label.
3. Re-run the held-out table against a constant-velocity Kalman baseline, with paired session-bootstrap CIs.
4. Replace the single F1-picked point with operating curves at frozen alert budgets, plus lead time normalised by opportunity.

Only after that should the team add the LiDAR-realism bound, the in-app iPhone p95 and segmentation, the last as a measured ablation rather than a headline. With that package CPE can go to a CV-accessibility workshop or WACV now, and to ASSETS after a small BLV listening study. No version of the paper should claim safety.

## Nine moves, ordered by what they unblock

The ordering follows dependencies, not how much each item is talked about. Segmentation and on-device latency lead `pending_work.md`'s "Next three" ([pending_work.md](D:/Projects/CompositePerceptionEngine/docs/pending_work.md)). The research suggests neither is what currently limits a reviewer's verdict. The phone will almost certainly meet its 50 ms budget, and segmentation is where CPE overlaps most with prior work. What reviewers will attack is the label, the baselines and the operating point. Those are cheap to fix because the whole replay is already fast: 71 test sessions × 7 engines plus 41 validation sessions × 30 settings ran in **38 s on a laptop CPU** ([progress.md](D:/Projects/CompositePerceptionEngine/docs/progress.md)). Dense sweeps cost minutes, not days. Effort figures below are this report's estimates for one engineer.

| # | Move | Why it matters | Effort | Evidence it produces |
|---|---|---|---|---|
| 1 | Switch to `fixed_camera_poses.csv` read with the OpenCV / Z-up convention; interpolate NOT_READY rows | Fixes the static-object drift and enables a walking-direction frame | 1–2 days | Three pass/fail gates over all 462 sessions; ego-motion moves from "unvalidated" to "validated" |
| 2 | Redefine "in my path" as a ~0.75 m corridor plus a closest-approach (CPA) label; keep the cone as an "awareness" tier | The cone makes "nearest object" nearly unbeatable and pins lead time at ~0 s | 3–5 days | New held-out table, label-prevalence counts, sensitivity grid over width and horizon |
| 3 | Baseline ladder: Kalman constant-velocity (CV) extrapolation, a last-two-steps constant-velocity model (CVM), logistic regression or gradient-boosted trees on current features, GRU history 1/4/8, static vs moving strata | Reviewers treat CVM as mandatory, and it may tie the GRU | 3–4 days | Paired ΔAUROC/ΔAUPRC with session-bootstrap CIs |
| 4 | System Operating Characteristic (SOC) curves (recall vs false alerts/min); 2–3 operating points chosen on validation at fixed alert budgets; split each alert into true / nuisance / false | One F1-chosen point cannot show dominance | 2–3 days | v1 vs v2 compared at matched budget; curve bands |
| 5 | Lead time: opportunity-window distribution, normalised lead, Kaplan-Meier "not yet warned" curves, lead at matched recall | "0 s lead" is probably a data property, not an engine property | 2 days | Conditional lead-time table per class |
| 6 | Depth realism bound: re-run on SANPO depth degraded to iPhone-LiDAR quality ("LiDAR-ised") | Pre-empts "your depth is unrealistic" | 3–5 days | Δ of every headline metric under degraded depth |
| 7 | In-app iPhone timing via GitHub Actions + TestFlight; sustained run with ARKit live | The one headline claim with no number | 1–2 weeks, mostly logistics | Median/p95/p99 by stage, thermal timeline, latency-aware replay |
| 8 | Segmentation (YOLO26s-sem, coarse walkable scheme, chest-camera masks), judged by false-safe rate and a replay on/off ablation | Fills the "what" term and removes ground pseudo-objects | 2–3 weeks | Pixel, task and system-level tables |
| 9 | Redesign the VLM referee (order-swap, marked boxes, text-only control, human ceiling, n≈200–300); BLV listening study for the HCI version | Validity of the ranking and of the alert budget | 1–2 weeks; 4–8 weeks with ethics | Agreement with CIs; user ratings of v1 vs v2 audio |

Moves 1 and 2 are coupled: the corridor label needs a walking direction, and the walking direction needs trustworthy poses. Moves 3–5 are pure re-analysis on existing replay outputs once labels exist. Move 7's logistics (a borrowed phone and a paid Apple Developer account) should start in week one so the measurement is not blocked later.

## A misread pose file and a 3.5 m cone explain the weak results

### The pose mismatch has two separate causes, and both are fixable

The research team measured both SANPO pose files on 12 sessions. Each file produces the documented symptom (static objects moving 0.52 m per step in the world frame against 0.32 m in the camera frame), but for different reasons. The original `camera_poses.csv` has endpoints that match, but its per-frame translations jitter by ±0.1–0.4 m. Its **path length is 2.5–25× the net displacement**. Across all 112 cached files, 1 s windows show a median path/net ratio of 1.79, and a median 3.7% of steps imply speeds above 4.5 m/s. Its rotations are near-identity under every quaternion reading, which is why "no axis sign flip fixes it." The corrected `fixed_camera_poses.csv` is smooth (path/net 1.02–1.2) but follows the OpenCV convention: x right, y down, the camera looking along **+z**, in a world whose up axis is **+Z**. Measured walking displacement lies 10–22° from camera +z and 159–170° from −z. `fusion.py` hard-codes the ZED convention (forward −z, up +y). Applied to the fixed file, it points "forward" backwards, so a static object's world estimate moves by twice the walker's step. That predicts **≈0.56 m per 0.2 s step at 1.4 m/s, against the 0.52 m observed**. SANPO's README says only that the original orientation "was not in sync" and that corrected poses live in the fixed file. It gives no units or axis conventions, and its "right-handed, Y-up" wording does not match what the data shows for the world frame ([SANPO README](https://github.com/google-research-datasets/sanpo_dataset)). The SANPO paper itself aligned accumulated point clouds with ICP, a hint that the raw poses were never clean ([SANPO paper](https://arxiv.org/html/2309.12172v2)).

The fix is a convention flag in `EgoPose.from_sanpo` / `sanpo_session_poses` that sets right = R·(1,0,0), forward = R·(0,0,+1) and up = +Z (or the session mean of −R·(0,1,0)), with the replay pointed at `camera_head/fixed_camera_poses.csv`. NOT_READY rows should be interpolated between READY poses rather than held, because holding a pose while the walker keeps moving makes static objects appear to approach. The first few READY rows of the fixed file sit at the origin and should be dropped. Accept the fix only if three gates pass across all 462 sessions; the check needs only a few MB of CSV:

1. The median ground-plane angle between the 1 s walking displacement and the forward vector is below ~15°.
2. Near (<8 m) static-class tracks move less per step in the world frame than in the camera frame, ideally ≤0.15–0.2 m, the box-depth noise floor.
3. The per-session path/net ratio stays below 1.3, with exceptions flagged.

If residuals remain, the likely suspects in order are: the x = z·tanθ lateral projection, which ignores the ~10–15° head pitch; box-median depth jumping between an object and its background; and parked versus moving members of "static" classes. One related loose end needs recording but does not block the fix. The paper calls the head camera a ZED-M, while every curated `camera_head` session's metadata says ZED 2i with ~58° HFOV ([description.json](https://storage.googleapis.com/gresearch/sanpo_dataset/v0/sanpo-real/1rL8RTbI4jPqRkavZxrQbbM26iRfSLAH/description.json)).

### The ±30° cone labels almost everything nearby as "in my path"

At 3 m a ±30° cone has a half-width of 3·tan 30° ≈ **1.73 m**, about 3.5 m across. Pedestrian-planning references put comfortable lateral body spacing at **0.71–0.76 m** ([Pushkarev & Zupan, TRB 1975](https://onlinepubs.trb.org/Onlinepubs/trr/1975/538/538-001.pdf); [NYC DCP LOS appendices](https://www.nyc.gov/assets/planning/download/pdf/plans/transportation/td_pedlosappendices.pdf)). Two people passing need almost 1.5 m ([FHWA HRT-05-137](https://www.fhwa.dot.gov/publications/research/safety/pedbike/05137/chapter2.cfm)). BBeep, the closest assistive precedent, triggers when a pedestrian's linearly extrapolated path crosses an "emergency line" annotated as about 0.7 m in front of a blind user, with two horizons of **5.0 s and 2.5 s** ([BBeep CHI'19](https://wotipati.github.io/projects/BBeep/paper/CHI'19_BBeep_preprint.pdf)). With a real HFOV of 51–60°, the cone covers nearly the whole image ([progress.md](D:/Projects/CompositePerceptionEngine/docs/progress.md)). A label of "will be within 3 m and anywhere in view" is close to "is near and visible now". This is why proximity alone (nearest-in-cone, 0.875) nearly matches the GRU (0.888), and why objects are already "arriving" when first seen.

The replacement label has three tiers, all still computed from SANPO depth and the now-fixed poses:

1. **Corridor-entry label.** Define a path frame from the ego displacement over ~1 s, not the head's optical axis. An object "arrives" if its measured future position enters |x| ≤ w/2 + r_obj with 0 < z ≤ D within H. Use w = 0.75 m as the primary value, with a sensitivity grid w ∈ {0.7, 1.0, 1.5} m and horizons H ∈ {2.5, 4, 5} s chosen to match BBeep.
2. **CPA label for movers.** The minimum predicted distance under measured future relative motion falls below ~1.0 m, with time to closest approach inside the horizon. This is the standard variable in walker-avoidance research ([Gait & Posture 2013](https://www.sciencedirect.com/science/article/abs/pii/S0966636213001768)). It catches crossing pedestrians and cyclists who never sit "in front" for long.
3. **Awareness tier.** The old cone becomes a lower-urgency tier, giving a two-level structure analogous to BBeep's low- and intermediate-urgency beeps.

The cost is prevalence. Today's 88 test arrivals come from only 25 test sessions with arrivals, and the replay recall CI is already [0.18, 0.49]. A corridor label will cut positives further. The team should count arrivals per label before committing, and should report label studies on pooled validation and test data if the test-only count drops too low.

### Lead time must be measured against the time each object was visible

Lead time cannot exceed the window between first detection and arrival, W = t_arrival − t_first_seen. The first table in the lead-time section should be the distribution of W. If its median is a fraction of a second, "median lead ≈ 0 s" describes SANPO and YOLO's range, not the engine. Report lead three further ways:

- on subsets with W ≥ {1, 2, 4} s (with n and CIs);
- as normalised lead (lead / W);
- as a Kaplan-Meier curve of "probability not yet warned" against time-to-arrival, treating unwarned arrivals as right-censored.

The survival framing is standard statistics but the research found no assistive-navigation precedent for it. Comparisons across engines must use matched recall or matched budget. Accident-anticipation work reports time-to-accident at 80% recall (TTA@R80), and recent papers cap the false-positive rate because "indiscriminate positive predictions" inflate lead time ([arXiv 2510.22260](https://arxiv.org/html/2510.22260); [ROAR](https://arxiv.org/html/2511.06226)). v1's 0.6 s median lead over v2's 0 s is partly an artefact of v1 talking 7× more. Reviewers will also ask why CPE's safety envelope uses TTC ≤ 1 s when automotive forward-collision warning (FCW) requires the alert at **TTC ≥ 2.1 s** (stopped lead vehicle) and **≥ 2.4 s** (braking lead), in 5 of 7 trials ([NHTSA FCW test procedure](https://downloads.regulations.gov/NHTSA-2006-26555-0134/attachment_2.pdf)). No pedestrian standard exists. CPE should derive its own requirement from reaction time, speech duration and stopping distance at ~1.4 m/s, in the style of ISO 15623's warning-distance model ([ISO 15623:2013](https://www.iso.org/standard/56655.html)). It can cite BBeep's finding that 5 s warnings produced larger passing distances than 2.5 s ([BBeep](https://wotipati.github.io/projects/BBeep/paper/CHI'19_BBeep_preprint.pdf)).

## The GRU must beat a constant-velocity Kalman at matched alert budgets

### Add the baselines a forecasting reviewer considers mandatory

Trajectory-forecasting reviewers treat the constant-velocity model as a must-beat baseline. Schöller et al. showed that a CVM using only the last two timesteps **outperformed four state-of-the-art neural predictors** on ETH/UCY and that motion history was largely irrelevant ([Schöller et al., RA-L 2020](https://arxiv.org/abs/1903.07933)). A 2024 T-ITS evaluation found the CVM "a good approximation of the overall dynamics in the majority of cases" ([Uhlemann et al.](https://arxiv.org/abs/2308.05194)). CPE's TTC and cone-TTC are only partial proxies. The canonical baseline is nearly free because CPE already runs a Kalman filter: propagate the fused state 4 s and score the probability of entering the new corridor from the predicted position and covariance.

The held-out table should climb this ladder:

1. Nearest-in-corridor.
2. TTC and corridor-TTC.
3. Kalman CV extrapolation.
4. Schöller's two-step CVM.
5. Logistic regression or gradient-boosted trees on the current-frame features the GRU sees, which isolates the value of the 1.6 s sequence.
6. GRU with 1, 4 and 8 history steps, which tests Schöller's "history is irrelevant" claim directly.

A learned image-space TTC model is a different-paradigm comparison that CV venues may request and HCI venues will not, for example Binary TTC (per-pixel "collides within τ" from two frames, 6.4 ms) ([Badki et al., CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Badki_Binary_TTC_A_Temporal_Geofence_for_Autonomous_Navigation_CVPR_2021_paper.pdf)) or Manglik et al.'s egocentric TTC regression ([arXiv 1903.09102](https://arxiv.org/pdf/1903.09102)).

Results must be stratified into **static versus moving objects**. Most SANPO "arrivals" are probably static things the walker approaches. That makes the task largely ego-motion forecasting, which belongs to camera-wearer forecasting work ([Qiu et al., RA-L 2022](https://arxiv.org/abs/2111.00993)) rather than pedestrian-interaction models. Whether the GRU's edge sits in movers, in statics or in neither is likely the paper's most interesting finding. The existing null result for cross-object attention (significantly worse on arrival order) becomes a consistent finding once tied to Schöller's observation that adding neighbours' histories can hurt.

On statistics, keep the session-level paired bootstrap already in use. Add **AUPRC**, because arrivals are rare, and add DeLong tests as a cross-check. Apply a Holm correction across the ladder so that six comparisons do not produce one lucky "win". Freeze every threshold on validation before touching test, and report the spread across the three seeds. The same bar applies to the planned learned ensemble that replaces SLM-1: the stacked ensemble's held-out 0.893 is not significantly better than the GRU alone ([progress.md](D:/Projects/CompositePerceptionEngine/docs/progress.md)), so each added model must win a paired test before it enters the paper.

The steelman case against the GRU deserves a plan. If Kalman-CV or nearest-in-corridor still ties it under the corridor label, CPE should say so plainly and move the learned model's claim to what it demonstrably buys: **ranking stability** (flicker 0.56 against 0.71, and 5× fewer topic switches at system level). A benchmark showing that geometry suffices for arrival on SANPO while learning stabilises announcements is still a contribution. Hiding a tie is not.

### Replace the F1 point with SOC curves and frozen alert budgets

The collision-alerting field's native evaluation tool is the **System Operating Characteristic curve**, which plots successful alerts against unnecessary alerts as the threshold sweeps. Design changes are judged by how they shift the curve ([Yang & Kuchar 1997](https://www.mit.edu/~jkkuchar/Y&K1997.pdf); [Kuchar 1996](https://arc.aiaa.org/doi/pdf/10.2514/3.21637)). CPE's main figure should plot warned (recall of arrivals) against false alerts per minute on a log axis for each engine. The GRU threshold, cooldown and switch margin should be swept densely (20–50 values, not 6), with session-resampled bootstrap bands. If v2's curve passes above v1's point, v2 dominates. Today's pairing of 0.59 precision at 17/min with 0.24 at 119/min cannot show that.

Operating points should be chosen by constraint rather than by F1 of (warned, precision), which weights a missed arrival and a false alert equally without justification. The rule is to maximise warned subject to false alerts/min ≤ B, for B ∈ {2, 5, 10}, chosen on validation and frozen for test. This mirrors the user-selectable FCW sensitivity levels that NHTSA analysed separately ([NHTSA DOT HS 812 611](https://rosap.ntl.bts.gov/view/dot/41002/dot_41002_DS1.pdf)).

Every alert should also be classed as true, nuisance or false. NHTSA counted 98% of FCW alerts as in-path, but flagged the **22% of in-path alerts issued while the driver was already braking** as likely nuisance ([NHTSA DOT HS 812 611](https://rosap.ntl.bts.gov/view/dot/41002/dot_41002_DS1.pdf)). CPE's precision currently counts "already inside the zone" as correct, so it overstates usefulness. Alerts per minute should also be reported conditional on scene density, so crowded scenes do not dominate the average.

Precision matters for a human reason. Bliss et al. found about 90% of participants **matched their response rate to the alarm's true-alarm probability** ([Bliss, Gilson & Deaton 1995](https://doi.org/10.1080/00140139508925269)). By that logic, v1 alerts would be heeded about a quarter of the time and v2's a little over half. That supports a modelled "expected avoided arrivals per minute of alert burden" curve, provided it is labelled as a model and not a measurement. The burden is real: FCW drivers turned their sensitivity down at around **4 alerts per 100 miles** ([NHTSA DOT HS 812 611](https://rosap.ntl.bts.gov/view/dot/41002/dot_41002_DS1.pdf)), and BBeep beeps only when a collision is predicted because constant alarms made blind users self-conscious ([BBeep](https://wotipati.github.io/projects/BBeep/paper/CHI'19_BBeep_preprint.pdf)). The research found **no published tolerable alert rate for blind pedestrians**, so 17 alerts/min (one every 3.5 s) must be presented as an open question, not a target.

### The VLM referee needs controls and roughly twice the human labels

The referee is a supplementary validity check on announcement order, not a pillar of the paper. If it is run, its design should follow the judge literature. Multimodal judges agree with humans reasonably on pairwise comparison but poorly on batch ranking: GPT-4V scored 0.773 similarity without ties on pairs, and consistency on batch ranking was only 0.418 ([MLLM-as-a-Judge](https://arxiv.org/html/2402.04788v3.pdf)). Agreement also "varies considerably across models and datasets" ([JUDGE-BENCH](https://arxiv.org/abs/2406.18403)). The referee should therefore pose a forced choice over numbered boxes drawn on the image. Each frame should run twice with the candidate order reversed, and only order-consistent picks should count; GPT-4 was consistent on just 65% of swapped pairs by default ([Zheng et al. 2023](https://arxiv.org/abs/2306.05685)). A text-only control (the object list with no image) shows whether the VLMs are judging the scene or just reading the distance column.

Cohen's κ fits poorly when the number of candidates varies per frame. Report raw agreement against a per-frame 1/k chance baseline and Krippendorff's α for the panel, and keep κ for binary reductions. Pre-register these choices to avoid coefficient shopping ([CASRAI guide](https://casrai.org/guides/inter-rater-reliability-choosing-a-coefficient)). Two humans should label at least 50 frames to set a human-human ceiling, because a κ > 0.6 gate cannot be reached if humans agree only at 0.5.

Sample size also needs revisiting. From the standard large-sample formula, with p_o ≈ 0.8 and ~3 candidates, 100 items give a 95% CI of about ±0.12 and 150 about ±0.10. A point estimate of 0.6 therefore cannot cleanly clear a 0.6 gate. Either restate the gate as "lower bound > 0.4" or label about 200–300 frames. Agreement on disagreement-selected frames will be lower than on random frames and should be reported as such.

## LiDAR range, not phone latency, is the realism threat

### Degrade SANPO depth to LiDAR quality and report the delta

SANPO's dense depth is CREStereo output at 1280×720 out to 80 m ([SANPO paper](https://arxiv.org/html/2309.12172v2)). ARKit's `sceneDepth` is 256×192 with a per-pixel confidence map ([WWDC20](https://developer.apple.com/videos/play/wwdc2020/10611/)). Its centre error is **1–2 cm at 1–3 m**, but peripheral error reaches **~21–28 cm at 3 m** ([PMC10939328](https://pmc.ncbi.nlm.nih.gov/articles/PMC10939328/)). Its range is about 5 m by Apple's figure ([opencv.ai](https://www.opencv.ai/blog/depth-estimation)), and it actually updates at ~15 Hz ([MDPI Sensors 23(18):7832](https://www.mdpi.com/1424-8220/23/18/7832)). That refresh rate matches SANPO's 15 FPS, so the gap is range, resolution, peripheral noise and dropouts, not timing.

The transform is:

1. Centre-crop to 4:3.
2. Block-median downsample to 256×192.
3. Invalidate depth beyond R_max ∈ {5, 4, 3} m. The 3 m case stands in for direct sun, which has no peer-reviewed number, so the paper should not invent one.
4. Add noise that grows with distance and eccentricity, calibrated to the centre and peripheral errors above, plus extra noise at edges.
5. Drop about 2% of pixels at random, plus edge pixels.

Then recompute the box depths and re-run both the held-out ranking and the system replay. Labels must stay on the original SANPO depth so that only the model inputs degrade, which also partly breaks the current input/label depth sharing.

The expected cost is concrete. If objects become measurable only inside ~5 m, a walker at 1.4 m/s gets **about 1.4 s of track before the 3 m zone**, against today's 4 s horizon. The tighter corridor zone from move 2 helps here, because a 1.5 m event zone leaves roughly 2.5 s of opportunity under the same 5 m range. That is BBeep's short horizon, and the combination is worth stating in the paper.

### Measure in-app p95 under sustained load, not isolated model time

Published numbers make the 50 ms alert-path budget very likely achievable. On an iPhone 17 Pro, YOLO26n detection takes **3.2 ms** of isolated ANE time, about **7 ms** of model time inside a live camera app, and **11.3 ms** end to end in the shipped 720p configuration. FP16 and INT8 are within noise ([Ultralytics yolo-ios-app performance.md](https://github.com/ultralytics/yolo-ios-app/blob/main/docs/performance.md)). The same document shows isolated timings understate in-app time roughly fourfold, so the defensible number is the in-app one.

The practical route without a Mac is a minimal SwiftUI harness in a public repo. It bundles the end2end `yolo26n.mlpackage` (FP16, no minimum deployment target, since setting one regresses latency), the GRU in Core ML or Accelerate, and 300–1000 SANPO frames. It is built on a GitHub Actions macOS runner and installed through TestFlight, which needs a $99/yr developer account, and since April 2026 an Xcode 26 runner ([NextTrain PR #60](https://github.com/longle7/NextTrain/pull/60); [Lc0-ios](https://github.com/lucario6607/Lc0-ios)). The harness should do the following:

1. Time each stage with `os_signpost` to a CSV.
2. Run a warm-up block, then interleaved blocks, then a **20–40 minute** sustained loop at 15 FPS pacing.
3. Keep an `ARSession` with `.sceneDepth` live throughout, so LiDAR and visual-inertial odometry (VIO) contend for resources as they would in use.
4. Log `thermalState` every frame.

Report median, p95, p99 and p95 per 5-minute window. Logging battery drain is worthwhile because BlinkBud and WOAD both report power ([BlinkBud](https://arxiv.org/abs/2512.01366); [WOAD](https://www.nature.com/articles/s41467-025-58085-x)). Feed the measured latencies, plus segmentation staleness, into the replay. Installing the public Ultralytics app is a zero-effort sanity check but does not measure CPE's model. The only published sustained-load figure for the 17 Pro Max is a GPU stress test (65.2% stability after 20 minutes) ([Tom's Guide](https://www.tomsguide.com/phones/iphones/iphone-17-pro-max-endurance-tested-does-the-new-design-actually-improve-performance)), so CPE's own ANE thermal timeline is new data rather than a formality. Consistent with the replay-only scope, this is a latency benchmark, not a field demo. The SANPO-vs-iPhone HFOV difference should be stated as a limitation, not simulated away.

## Segmentation earns its place only through a replay ablation

### There are more labelled sessions than the audit counted

The audit's "53 sessions with masks" counts the `camera_head` stream only ([session_audit.json](D:/Projects/CompositePerceptionEngine/simulation/datasets/sanpo/session_audit.json)). Session `-5OCPnbrw…`, for example, is marked mask-free in the audit, yet its `camera_chest/left/segmentation_masks/` holds 528 frames, 88 of them human-annotated ([GCS listing](https://storage.googleapis.com/storage/v1/b/gresearch/o?prefix=sanpo_dataset/v0/sanpo-real/-5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG/); [annotation-type JSON](https://storage.googleapis.com/gresearch/sanpo_dataset/v0/sanpo-real/-5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG/camera_chest/left/frame_segmentation_annotation_type.json)). SANPO reports panoptic labels on **237 sessions**: 146 on the chest camera, **18,787 human-annotated frames** and ~94k propagated ([SANPO paper](https://arxiv.org/html/2309.12172)). Recounting masks across both cameras is the single biggest data lever for segmentation. Training can use every labelled session outside the official test split.

The catch is that chest masks do not overlay the head-camera frames the replay uses. Task-level evaluation therefore stays on the 10 head-masked test sessions unless the team takes the optional step of moving Stage 1 to the chest stream, which is also the more phone-like mount. Evaluation should use human-annotated frames only, as SANPO does. Masks decode as class = R channel, with 0 mapped to ignore.

### Use a coarse scheme, a small first-party model, and safety-asymmetric metrics

Fine-grained SANPO segmentation is hard: Kmax-DeepLab reaches only 43–48 mIoU over 31 classes. Synthetic-only training scores **14.6–17.6 mIoU** on real data, and synthetic data helps by at most ~2 points at a ≤50% mix ([SANPO paper](https://arxiv.org/html/2309.12172)). Zero-shot Cityscapes models score sidewalk IoU as low as 0.12. The realistic target is SANPO's own on-device 3-class MobileNetV3, which scores safe-for-walking **83.4**, not-safe 64.5 and obstacles 85.9 IoU.

The recommended recipe is:

1. **Model.** Start from YOLO26s-sem Cityscapes weights (80.8 mIoU on Cityscapes); its nano sibling runs in **4.86 ms on an iPhone 17 Pro** with argmax inside the Core ML graph ([Ultralytics semantic docs](https://docs.ultralytics.com/tasks/semantic); [yolo-ios-app #322](https://github.com/ultralytics/yolo-ios-app/issues/322)).
2. **Classes.** Remap to roughly eight classes: walkable, crosswalk, road, curb, stairs, non-walkable ground, obstacles, sky.
3. **Rare classes.** Oversample curb, stairs and crosswalk.
4. **Input.** Train at a 16:9 input such as 1024×576, because thin curb edges need resolution and latency has headroom at 2–5 Hz.
5. **Export.** Ship FP16 Core ML. Avoid Core AI's in-graph argmax, which produced wrong output under hardware acceleration ([yolo-ios-app #322](https://github.com/ultralytics/yolo-ios-app/issues/322)), and skip INT8, which gives no speed gain and risks thin boundaries.

Evaluation should run at three levels:

- **Pixel level:** mIoU on the scheme, per-class IoU, boundary F for curb and stairs, and above all the **false-safe rate** (road or stairs predicted walkable). SENSATION-DS found "the highest-mIoU model was not the safest" ([SENSATION-DS](https://arxiv.org/html/2607.21137v1)).
- **Task level:** per-track surface accuracy, edge-distance error in metres, and fewer ground pseudo-obstacles at unchanged true-obstacle recall.
- **System level:** the replay with context on versus off.

No prior segmentation-for-blind-navigation work reported that last level, so it is where CPE's segmentation becomes novel. It should not be the paper's headline, because Mobilio (iPhone + LiDAR, n = 14 BVI users, 41% fewer contacts) appears to train segmentation on SANPO-type data ([Mobilio summary](https://pith.science/paper/2605.29120)). If CPE leads with segmentation it looks like a weaker Mobilio, while leading with arrival forecasting under an alert budget does not.

## Claim alert-policy properties, not safety, and pick venues accordingly

Open-loop replay can mislead. On nuPlan, open-loop and closed-loop scores were **negatively correlated** ([Dauner et al., CoRL 2023](https://arxiv.org/abs/2306.07962)). In CPE's replay the recorded walker never heard an alert, so warned arrivals cannot become avoided ones. Every competitor that claims safety measures it with people: Pundlik's double-masked trial cut contacts by about 37% over 368 h of walking ([JAMA Ophthalmology 2021](https://www.researchgate.net/publication/353414917_Home-Use_Evaluation_of_a_Wearable_Collision_Warning_Device_for_Individuals_With_Severe_Vision_Impairments_A_Randomized_Clinical_Trial)), and Mobilio and WOAD ran BVI studies ([WOAD](https://www.nature.com/articles/s41467-025-58085-x)).

Pundlik's silent-mode arm is the right framing for CPE's replay: the system runs and nobody hears it. That supports these claims:

- held-out predictive superiority (or a documented tie) against named baselines;
- precision, recall and nuisance at frozen alert budgets;
- opportunity-normalised lead time;
- announcement stability;
- a depth-realism bound;
- measured on-device latency.

It does not support "safer", "reduces collisions", "usable by blind people" or "17 alerts/min is acceptable".

On venue, the best immediate fit is a CVPR or ICCV accessibility workshop, which accepts benchmark and algorithm papers without user studies ([CVPR 2025 AVA](https://cvpr.thecvf.com/virtual/2025/workshop/32372); [ICCV 2025 CV4A11y](https://openaccess.thecvf.com/content/ICCV2025W/CV4A11y/papers/Tian_Introduction_to_the_First_Workshop_on_Vision_Foundation_Models_and_ICCVW_2025_paper.pdf)). A WACV applications paper is the other option. Either should be framed as "hazard-arrival forecasting on SANPO", with released labels and code; SANPO's CC BY 4.0 licence permits this ([SANPO GitHub](https://github.com/google-research-datasets/sanpo_dataset)). NeurIPS Datasets & Benchmarks took a synthetic BLV walking benchmark with no users ([mmWalk](https://arxiv.org/abs/2510.11520)). RA-L or IROS fits if the baseline ladder is strong.

ASSETS 2026 softened its old rule. It now "recommends" validation with representative users "when possible" and accepts mainly technical contributions grounded in prior formative work, but reviewers still ask whether claims are supported "with the target users" ([ASSETS 2026 CFP](https://assets26.sigaccess.org/call_for_papers.html)). The cheapest bridge is a remote study with BLV participants that needs no walking. They can rate pre-rendered v1 and v2 audio sequences on SANPO clips, or state tolerable alert rates, which answers the open alert-budget question directly. This makes the P4 clip rendering a dependency of the HCI paper, not of the workshop paper. It also needs ethics approval, so it belongs on the timeline now.

Before drafting, fix two citation errors in `related_work.md`. SANPO's first author is **Sagar M. Waghmare** and the venue is **WACV 2025** ([WACV PDF](https://openaccess.thecvf.com/content/WACV2025/papers/Waghmare_SANPO_A_Scene_Understanding_Accessibility_and_Human_Navigation_Dataset_WACV_2025_paper.pdf)). Mobilio's listed Nature BME venue could not be confirmed and may be arXiv-only. BlinkBud shares Kalman tracking and head-motion compensation with CPE ([arXiv 2512.01366](https://arxiv.org/abs/2512.01366)), so it must be cited and distinguished, along with Pundlik.

## Conclusion

The research changes CPE's to-do list. CPE's ceiling is set by geometry, not by model capacity. Every symptom the team has logged traces to coordinate frames and zone definitions: the GRU tying a proximity rule, lead time near zero, ego-motion that "doesn't reconcile", and an operating point nobody can defend. These are days of work, and they must come before segmentation, because segmentation results measured under a cone label would need redoing. The constraint that actually binds the product is sensor range. Under realistic LiDAR range, a 4 s horizon for static obstacles at walking pace cannot be met. A body-width corridor with a ~1.5 m event zone is both more honest and closer to what BBeep found behaviourally useful.

The most likely research outcome is mixed, and still publishable. Simple geometry may predict arrivals on SANPO as well as the GRU does, while learning delivers steadier, fewer and more precise announcements on an SOC curve. Stated plainly, with a released benchmark, that is a stronger paper than a 0.013 AUROC margin presented as a win. It also asks the right question for the follow-up: what alert rate blind pedestrians will actually tolerate.
