# CPE positioning against the literature, expected baselines/metrics, and venues

Scope: 2020-2026 work relevant to CPE (phone, SANPO replay, YOLO26n + ByteTrack + depth -> Kalman -> TTC envelope -> GRU arrival probability x severity^0.25 x scene context -> announcer -> pre-rendered audio). Research done 2026-10-14 with web search plus fetches. Sources marked "(not fetched this session)" are well-known papers cited from prior knowledge. Check them before they go into `docs/related_work.md`, which requires live verification.

CPE numbers referenced below come from `docs/progress.md`: held-out AUROC GRU 0.888 / K0 0.571 / TTC 0.661 / cone-TTC 0.765 / nearest-in-cone 0.875. Flicker: GRU 0.56 vs 0.71. Replay: v2 precision 0.59 at 17 alerts/min and 26% of arrivals warned, against v1 0.24 at 119/min and 65%. Lead time is about 0 s for every engine because objects first appear near the 3 m zone. Processing runs at 5 Hz with a 4 s horizon.

## Q1. Closest prior work (2020-2026) on assistive hazard detection/prioritisation for blind pedestrians, and how it was evaluated

### Takeaway
The closest systems are Mobilio (Liu & Slade 2026, iPhone with LiDAR, trained partly on SANPO), WOAD (Nature Comms 2025), the Pundlik/Luo collision-warning RCT (JAMA Ophthalmology 2021), BlinkBud (IMWUT 2025) and the WalkVLM line (2024-25, about reminder redundancy). Every high-impact one is evaluated with BLV or blindfolded users and reports collisions/contacts, plus FP/FN rates or latency. None publishes an offline, held-out "will this object reach me" prediction benchmark with alert-rate vs. recall curves on a public dataset. That benchmark is CPE's defensible niche.

### Cited Findings
**Smartphone / wearable systems with obstacle warning**
- **Mobilio, Liu & Slade (2026), arXiv:2605.29120.** `related_work.md` lists it as Nature Biomedical Engineering; the Pith page lists only the arXiv preprint with no venue, so the venue needs re-checking. Hardware: iPhone 12 Pro chest-strapped, camera + IMU + GPS + LiDAR, open-ear headphones, white cane. Survey of 112 BVI respondents named turn-by-turn directions, path guidance and obstacle avoidance as the three needs. User study n=14 BVI on a community path vs. Google Maps + cane: 13 ± 3% less time, 41 ± 5% fewer environmental contacts. — [Pith summary of arXiv:2605.29120](https://pith.science/paper/2605.29120)
  - Reference snippets on that page suggest the segmentation model was trained on SANPO/Cityscapes data and run on device via ONNX/Unity Sentis. This is unconfirmed from the full text but important: if true, Mobilio is a direct SANPO-using competitor. — [Pith summary](https://pith.science/paper/2605.29120)
  - A news write-up reports LiDAR obstacle-distance error of 8.0% vs 12.2% for camera-only. In the obstacle course, cane + Mobilio gave 41% fewer contacts but took 17% longer. — [The Brighter Side of News](https://www.thebrighterside.news/post/smartphone-app-helps-blind-users-navigate-faster-and-avoid-obstacles/) (secondary source)
  - An AI-generated review critique flagged one path plus one obstacle course, n=14, missing definitions of "contacts", and no failure-rate or edge-case engineering tests (rain, low light, crowds). This shows the bar even for user-study papers. — [Pith](https://pith.science/paper/2605.29120)
- **Pundlik, Baliutaviciute, Moharrer, Bowers, Luo (2021).** "Home-Use Evaluation of a Wearable Collision Warning Device for Individuals With Severe Vision Impairments: A Randomized Clinical Trial", *JAMA Ophthalmology* 139(9):998-1005. Chest-mounted camera detects collision hazards. Double-masked RCT with active vs. silent mode, 31 completers, 368 h of walking video over 4 weeks of daily use. All-contact rate fell about 37% and body contacts about 36% in active mode (median contacts 9.3 vs 13.8). — [ResearchGate](https://www.researchgate.net/publication/353414917_Home-Use_Evaluation_of_a_Wearable_Collision_Warning_Device_for_Individuals_With_Severe_Vision_Impairments_A_Randomized_Clinical_Trial); [ClinicalTrials NCT03057496](https://clinicaltrials.gov/study/NCT03057496)
  - This is the closest conceptual match to CPE: a camera-based collision warning (not just obstacle presence) with a silent-mode design. The silent-mode design is the clinical analogue of CPE's replay, where the system runs but nobody hears it.
- **WOAD, Gao et al. (2025).** "A wearable obstacle avoidance device for visually impaired individuals with cross-modal learning", *Nature Communications* 16:2857. Glasses with RGB + ToF depth and an FPGA, plus a smartphone running cross-modal (feature-level RGB-depth) obstacle detection with sound/haptic warnings. Reports 100% collision avoidance in indoor/outdoor experiments, under 320 ms delay and about 11 h battery. — [Nature Communications](https://www.nature.com/articles/s41467-025-58085-x); [PubMed](https://pubmed.ncbi.nlm.nih.gov/40128555/)
  - The search summary says 20 visually impaired volunteers took part. — [search snippet via Nature Comms](https://www.nature.com/articles/s41467-025-58085-x)
- **BlinkBud, Li et al. (2025),** *IMWUT* 9(4) Art. 191. A single earbud camera plus phone detects hazards approaching from behind (vehicles). Uses Kalman-filter trajectory estimation, an RL policy for image sampling, and head pitch/yaw compensation. Reports 4.90% average false-positive ratio, 1.47% false-negative ratio, 29.8 mW earbud / 702.6 mW phone power. No lead time or precision/recall curve in the abstract. — [arXiv:2512.01366](https://arxiv.org/abs/2512.01366)
  - It shares Kalman tracking + trajectory-based hazard triggering + head-motion compensation with CPE. Its evaluation is "extensive real-world experiments" with FP/FN ratios and power.
- **AI Guide Dog, Jadhav et al. (2025),** AAAI 2025 Spring Symposium. Smartphone, egocentric, vision-only multi-label classification of directional commands, GPS for outdoor routes. The abstract gives no baselines, metrics or BLV study. — [arXiv:2501.07957](https://arxiv.org/abs/2501.07957)
- **Co-Ego guide-dog robot, Liu et al. (2026),** arXiv:2603.20121, preprint. Fuses the robot's ground-level view with the user's egocentric view for body-height hazards. Metrics are collision times and cognitive load. The study used **blindfolded sighted participants**, not BLV users. It cites SANPO, GuideDog (arXiv:2503.12844), SideGuide (IROS 2020) and NavWear (2025). — [arXiv:2603.20121](https://arxiv.org/abs/2603.20121); [search listing](https://arxiv.org/pdf/2603.20121)
- Consumer iOS LiDAR apps ("Obstacle Detector – For Blind", "Super Lidar") give distance tones/speech. No formal evaluation; reviews note LiDAR needs a Pro iPhone. — [App Store: Obstacle Detector](https://apps.apple.com/us/app/obstacle-detector-for-blind/id6461118479); [App Store: Super Lidar](https://apps.apple.com/us/app/super-lidar-lidar-for-blind/id1543706309)

**VLM "walking assistant" line (directly relevant to alert fatigue)**
- **WalkVLM, Yuan et al. (2024),** arXiv:2412.20903. Contributes the Walking Awareness Dataset (WAD, 12k video-annotation pairs, Europe and Asia). Uses CoT hierarchical planning plus a **Temporal-Aware Adaptive Prediction** module: a lightweight model over past frames decides when a reminder is needed, to cut temporal redundancy. Metrics are ROUGE, TF-IDF similarity, GPT score and **Temporal Redundancy F1 (TRF)**. Baselines: LLaVA, DeepSeek-VL, Yi-VL, MiniCPM-V2.6, Qwen2-VL. — [arXiv:2412.20903](https://arxiv.org/pdf/2412.20903); [project page](https://walkvlm2024.github.io/)
- **WalkVLM-LR, Li et al. (2025),** arXiv:2508.16070. Uses GRPO with human-preference rewards for conciseness, plus an **environment-awareness discriminator** that scores scene risk and suppresses unneeded reminders. It argues that leaving the "when to speak" decision to the VLM adds latency. — [arXiv:2508.16070](https://arxiv.org/abs/2508.16070)

**Datasets/benchmarks around BLV navigation**
- mmWalk (Ying et al., NeurIPS 2025 D&B): synthetic, 120 trajectories, about 62k frames, 559k panoramic RGB/depth/semantic images, mmWalkVQA with 69k QA in 9 categories. VLMs did poorly on **risk assessment and navigation**. — [arXiv:2510.11520](https://arxiv.org/abs/2510.11520)
- Islam et al. (2024), a dataset for crucial object recognition in BLV navigation. — [arXiv:2407.16777](https://arxiv.org/pdf/2407.16777)
- Synthetic data augmentation for robotic mobility aids for BLV (2024) cites SANPO. — [arXiv:2409.11164](https://arxiv.org/pdf/2409.11164)
- A 2025 scoping review found that of 646 papers, only 38 involved BLV people in design or requirements. Most involved them only in evaluation. In its focus study, obstacle detection got the most first preferences (6 participants). — [arXiv:2505.19325](https://arxiv.org/html/2505.19325v1)

### Inferences
- CPE's differentiators:
  1. It forecasts per object whether the object will arrive, rather than detecting obstacles or segmenting a path (WOAD, Mobilio, consumer apps).
  2. It suppresses alerts explicitly with an operating point (17 vs 119 alerts/min). Among the papers found, only WalkVLM/WalkVLM-LR treat "when to speak" as a measured objective, and they use VLM reminders and text metrics, not hazard-arrival ground truth.
  3. It has a fully reproducible held-out benchmark on a public dataset (the SANPO official split). Mobilio, WOAD, Pundlik and BlinkBud all evaluate on private real-world runs.
  4. It is phone-only with a tiny model (31k-param GRU).
- CPE overlaps with BlinkBud (Kalman + trajectory triggering + head-motion compensation) and with Pundlik et al. (collision warning, not obstacle presence). Both must be cited and differentiated: BlinkBud handles rear hazards, Pundlik uses a dedicated chest device with an RCT.
- If Mobilio really trained on SANPO, reviewers will ask for a comparison against "segmentation + LiDAR free-space" style obstacle warning. CPE's planned walkable-surface segmentation (P1.1) moves it closer to Mobilio, so the GRU arrival predictor and alert budget must be the stated novelty, not segmentation.
- **What CPE can claim without a user study:** it predicts per-object arrival better than physics heuristics on held-out SANPO; it reduces alert rate at higher precision in open-loop replay; and it fits on-device latency (once P3 is measured). **What it cannot claim:** fewer collisions, better safety, usability, acceptable workload, or that 17 alerts/min is tolerable. Every competitor above that makes safety claims backs them with contacts/collisions measured with humans.
- Even 17 alerts/min (one every 3.5 s) is likely high for a BLV user. WalkVLM framing their whole contribution around redundancy suggests reviewers will see alert rate as central. This inference is not quantified by any source found.

### Gaps
- Mobilio's exact venue (Nature BME per `related_work.md` vs. arXiv-only per Pith) and whether it truly trains on SANPO are not confirmed from the full text.
- WOAD's participant count (20) comes from a search snippet, not the article text, because the fetch was blocked.
- Not fetched: NavWear (2025), GuideDog (arXiv:2503.12844), Apple's built-in Magnifier People/Door Detection (LiDAR). Their evaluation protocols are unknown.
- No CHI/ASSETS 2023-2026 paper was found that does offline per-object collision forecasting for BLV pedestrians. The search was not exhaustive (ACM DL not queried directly).

## Q2. Baselines a reviewer will expect for "will this object reach the walker"

### Takeaway
Reviewers in trajectory forecasting treat a constant-velocity model (CVM) as a strong, mandatory baseline. Schöller et al. (RA-L 2020) showed it beats several neural predictors on ETH/UCY, and a T-ITS 2024 study found it a good approximation in most cases. CPE's TTC and cone-TTC baselines are partial CV proxies, but CPE lacks the canonical one: Kalman constant-velocity extrapolation of the fused track, tested for entry into the 3 m corridor within 4 s. Reviewers will also expect a simple learned non-sequential baseline (logistic regression / GBM on current features), a learned TTC/collision forecaster from the egocentric literature, and statistical tests showing the GRU's 0.888 vs nearest-in-cone 0.875 is a real difference.

### Cited Findings
- **Schöller, Aravantinos, Lay, Knoll (2020),** "What the Constant Velocity Model Can Teach Us About Pedestrian Motion Prediction", *IEEE RA-L* 5(2):1696-1703. A CVM using only the last two timesteps outperforms four state-of-the-art neural models on ETH/UCY (8 observed / 12 predicted steps, 3.2 s / 4.8 s). They found motion history largely irrelevant and that adding neighbours' histories can hurt. — [arXiv:1903.07933](https://arxiv.org/abs/1903.07933); [code](https://github.com/cschoeller/constant_velocity_pedestrian_motion)
  - Later work calls the CVM "a standard comparison metric". — [search summary citing follow-ups](https://arxiv.org/html/1903.07933v3)
- **Uhlemann, Fent, Lienkamp (2024),** "Evaluating Pedestrian Trajectory Prediction Methods with Respect to Autonomous Driving", *IEEE T-ITS*. The CVM "provides a good approximation of the overall dynamics in the majority of cases". The paper ablates observation-history length and measures inference time vs. agent count, using ADE/FDE on ETH/UCY. — [arXiv:2308.05194](https://arxiv.org/abs/2308.05194)
- **Egocentric forecasting:**
  - Qiu et al. (2022), "Egocentric Human Trajectory Forecasting with a Wearable Camera and Multi-Modal Fusion", *IEEE RA-L* 7(4):8799-8806. Forecasts the **camera wearer's** trajectory in crowds from past ego trajectory, neighbours and scene semantics/depth, with a Transformer and cascaded cross-attention. The stated application is helping visually impaired people. — [arXiv:2111.00993](https://arxiv.org/abs/2111.00993); [TISS dataset](https://github.com/Jianing-Qiu/TISS)
  - The same group wrote "Indoor Future Person Localization from an Egocentric Wearable Camera" (IROS 2021). — [search summary](https://arxiv.org/abs/2111.00993)
  - EgoTraj (2026, real-world egocentric trajectory dataset, arXiv:2605.19004) and EgoCogNav (arXiv:2511.17581) are recent egocentric human navigation forecasting works. Only titles were seen. — [EgoTraj](https://arxiv.org/pdf/2605.19004); [EgoCogNav](https://arxiv.org/pdf/2511.17581)
  - Survey: "Predicting the Future from First Person (Egocentric) Vision: A Survey" (Rodin et al., arXiv:2107.13411). — [arXiv](https://arxiv.org/pdf/2107.13411)
- **Collision / time-to-contact from egocentric video:**
  - Manglik, Weng, Ohn-Bar, Kitani (IROS 2019), "Forecasting Time-to-Collision from Monocular Video". Regresses time to near-collision between a user-pushed suitcase robot and pedestrians, without explicit depth or velocity. 13,658 egocentric snippets with LiDAR ground truth; multi-stream CNN. Pre-2020, but it is the canonical "assistive egocentric TTC forecasting" paper. — [arXiv:1903.09102](https://arxiv.org/pdf/1903.09102)
  - COPILOT (2022) predicts and localizes human-environment collisions from egocentric video and treats Manglik et al. as closest prior work. — [arXiv:2210.01781](https://arxiv.org/html/2210.01781)
  - Badki, Gallo, Kautz, Sen (CVPR 2021 oral), "Binary TTC: A Temporal Geofence". Gives per-pixel "will collide within τ" binary classification from two monocular frames in 6.4 ms, with quantized/continuous TTC available. — [CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Badki_Binary_TTC_A_Temporal_Geofence_for_Autonomous_Navigation_CVPR_2021_paper.pdf); [code](https://github.com/NVlabs/BiTTC)
  - oTTC (2024) does object-level TTC and criticizes per-pixel Binary TTC for needing detections and wasting effort on background. — [arXiv:2405.07698](https://arxiv.org/html/2405.07698)
- **Accident anticipation from dashcams** is the closest analogue to "P(arrive within 4 s)" as a classification-over-time task. It uses AP over the precision-recall curve plus **mean time-to-accident (mTTA)** and TTA@R80. Papers report two operating points (best AP, best mTTA) because the two trade off: DSTA/UString lose about 1.5 s of mTTA when AP is pushed toward 60%. — [DSTA arXiv:2106.10197](https://arxiv.org/pdf/2106.10197); [RARE arXiv:2505.17449](https://arxiv.org/pdf/2505.17449)
- Not fetched this session (well-known; verify before citing): Trajectron++ (Salzmann et al., ECCV 2020, arXiv:2001.03093); Social-LSTM (Alahi et al., CVPR 2016); Yagi et al. "Future Person Localization in First-Person Videos" (CVPR 2018); Park et al. "Egocentric Future Localization" (CVPR 2016); PIE (Rasouli et al., ICCV 2019) and JAAD for pedestrian intention.

### Inferences
- **The constant-velocity Kalman baseline is considered strong and reviewers will demand it.** CPE already runs a Kalman filter, so this baseline is nearly free: propagate the fused state for 4 s and score P(enter corridor) via the predicted-position covariance, or use a hard yes/no. The GRU must beat it on the official split with CIs. Given CVM folklore and nearest-in-cone already at 0.875, there is real risk the gap is small. Reporting it honestly is better than having a reviewer ask.
- Recommended baseline ladder for the held-out table:
  1. Nearest-in-cone (done).
  2. TTC and cone-TTC (done).
  3. **Kalman-CV extrapolation into the 3 m corridor** (missing).
  4. **CVM with last-two-steps velocity**, the Schöller variant (missing).
  5. **Logistic regression / gradient-boosted trees on the current-frame features** the GRU sees (missing; isolates the value of the 1.6 s sequence).
  6. **GRU with history 1 / 4 / 8 steps** (partly planned in P2.1; directly tests Schöller's "history is irrelevant" claim).
  7. Optionally a Trajectron++-style or Transformer forecaster run on ground-plane tracks, then thresholded into arrival.
  - Add DeLong / bootstrap CIs on AUROC and AUPRC, since arrivals are likely imbalanced.
- Most "arrivals" in SANPO are probably static objects the walker approaches, so the task is largely **ego-motion forecasting** (where the walker is heading). The relevant literature is egocentric camera-wearer forecasting (Qiu 2022, Park 2016), not pedestrian-pedestrian interaction models. CPE should stratify results by static vs. moving objects. If the GRU's gain is only on moving objects, or only on static ones, that is the key finding either way.
- A learned image-based TTC (Binary TTC / Manglik-style) is a "different paradigm" baseline that reviewers at CV venues may ask about. It is optional at HCI venues.
- Cross-object attention not helping agrees with Schöller et al.'s finding that neighbour histories can hurt. Cite it to turn the null result into a consistent finding.

### Gaps
- Not found: any paper that uses a "will object X enter the user's personal zone within T s" binary label for BLV pedestrians, so there is no direct prior benchmark number to compare against.
- Whether recent egocentric forecasting papers (EgoTraj 2026) report CVM as competitive was not checked.

## Q3. Metrics for warning/alert systems: lead time, false alarms, alert fatigue, operating curves, standards

### Takeaway
Automotive FCW defines warning adequacy by **TTC at alert onset**: NHTSA NCAP requires TTC ≥ 2.1 s (stopped lead) and ≥ 2.4 s (braking lead), in at least 5 of 7 trials. ISO 15623 frames warning distance from driver reaction time plus braking. Anticipation research reports AP + mean time-to-accident at matched recall. Assistive papers report FP/FN ratios, collisions/contacts and latency. CPE should report a **recall-vs-alerts/min curve with a lead-time axis (TTA at fixed recall)**, not a single point.

### Cited Findings
- **ISO 15623** (2002; current edition 2013, "Intelligent transport systems — Forward vehicle collision warning systems — Performance requirements and test procedures").
  - Warning timing is judged from subject speed, distance to the obstacle vehicle and relative speed.
  - Warnings should come in time for drivers to avoid most common rear-end crashes by braking alone.
  - Variables include driver brake reaction times (T_max, T_min, T_resp) and braking-system response T_b.
  - Tests cover detection zone, warning-distance range/accuracy, target discrimination and curves.
  - — [ISO 15623:2013](https://www.iso.org/standard/56655.html); [ISO 15623:2002](https://www.iso.org/standard/27835.html); [2002 preview](https://cdn.standards.iteh.ai/samples/27835/7cad0eb49fc443e5b3b7529fbf4cd5a4/ISO-15623-2002.pdf)
  - The commonly quoted T_max ≈ 1.5 s / T_min ≈ 0.4 s values were **not confirmed** (paywalled).
- **NHTSA NCAP FCW confirmation test.**
  - Stopped lead vehicle at 45 mph: the alert must occur at TTC ≥ 2.1 s.
  - Lead vehicle braking at 0.3 g: TTC ≥ 2.4 s.
  - Slower lead vehicle: TTC ≥ 2.0 s per an NHTSA slide, but **1.8 s per an NTSB report** (the sources conflict).
  - A pass needs 5 of 7 trials. Only the earliest user-selectable timing setting is tested.
  - — [NHTSA FCW confirmation test procedure (2013)](https://downloads.regulations.gov/NHTSA-2006-26555-0134/attachment_2.pdf); [2023 NCAP FCW procedure](https://downloads.regulations.gov/NHTSA-2023-0060-0001/attachment_1.pdf); [NTSB SIR-15/01](https://www.ntsb.gov/safety/safety-studies/Documents/SIR1501.pdf)
- **Accident anticipation metrics.** AP = ∫P(R)dR. TTA = time between the first confident prediction and the event. mTTA averages TTA over thresholds. TTA@R80 is TTA at 80% recall. Example DAD results: DSTA AP 59.2 / mTTA 2.07 s; UString 58.6 / 2.12 s. — [DSTA arXiv:2106.10197](https://arxiv.org/pdf/2106.10197); [RARE arXiv:2505.17449](https://arxiv.org/pdf/2505.17449)
- **Assistive systems' reported metrics:**
  - BlinkBud: FP ratio 4.90%, FN ratio 1.47%, power. — [arXiv:2512.01366](https://arxiv.org/abs/2512.01366)
  - WOAD: collision-avoidance rate, under 320 ms latency, battery life. — [Nature Comms](https://www.nature.com/articles/s41467-025-58085-x)
  - Pundlik RCT: contact rates (body, cane) per hour, active vs. silent. — [ResearchGate](https://www.researchgate.net/publication/353414917_Home-Use_Evaluation_of_a_Wearable_Collision_Warning_Device_for_Individuals_With_Severe_Vision_Impairments_A_Randomized_Clinical_Trial)
  - Mobilio: contacts, completion time, heading error, workload surveys. — [Pith](https://pith.science/paper/2605.29120)
  - Co-Ego: collision times, cognitive load. — [arXiv:2603.20121](https://arxiv.org/abs/2603.20121)
  - WalkVLM: Temporal Redundancy F1 for "speaking at the right times". — [arXiv:2412.20903](https://arxiv.org/pdf/2412.20903)

### Inferences
- CPE's 1 s TTC envelope is shorter than automotive FCW thresholds (2.0-2.4 s). Walking speeds and stopping distances are far smaller, but reviewers may ask for a justification. A pedestrian analogue of ISO 15623 Annex A (reaction time + stopping distance at about 1.4 m/s) gives a principled lead-time requirement. This is a derivation CPE can do itself. No pedestrian standard was found.
- CPE's lead time of about 0 s is an artifact of a 3 m zone, given SANPO depth range and when objects first appear. Following the anticipation literature, redefine the event at the 1.5 m / contact zone (P2.3) and report **TTA at fixed recall (e.g. TTA@R50, R80)** and **the distribution of TTC-at-alert-onset**. Then compare against the FCW-style "≥ X s before contact in ≥ 5/7 of events" criterion.
- The metric set reviewers will expect, mapped from these sources:
  - (a) AUROC + **AUPRC** per object for the predictor.
  - (b) A system-level curve of recall vs. alerts/min, and precision vs. recall, swept over announcer thresholds (P2.2).
  - (c) Lead time at matched recall.
  - (d) FP alerts per minute or per km, the "nuisance" rate.
  - (e) Temporal redundancy / topic switches (CPE already has flicker and switches/min).
  - (f) On-device latency, p95 included, plus power. BlinkBud and WOAD report power/battery, so CPE should report energy per minute on the iPhone, not just latency.
- Report v1 and v2 at **matched alert rate or matched recall**. The current comparison (0.59 @ 17/min vs 0.24 @ 119/min) mixes two operating points, and reviewers will flag it.

### Gaps
- No standard or empirical study was found on acceptable alert rates or nuisance-alarm tolerance for BLV pedestrians. The automotive nuisance-alarm literature (e.g. Lee et al. on FCW timing and trust) was not searched.
- Exact ISO 15623 Annex A formulas and values are unverified (paywalled; a possible full copy at dig.watch was not fetched: https://dig.watch/wp-content/uploads/ISO_15623_2013_EN.pdf).
- Euro NCAP AEB-pedestrian / VRU test protocols were not checked.

## Q4. Published work using SANPO for navigation/hazard tasks; SANPO benchmarks

### Takeaway
SANPO was published at **WACV 2025**. Its benchmarks are depth estimation and (video) panoptic segmentation. No paper was found that uses SANPO for per-object arrival/collision forecasting or alert evaluation. Uses so far are Mobilio's segmentation training (unconfirmed), Google's Project Guideline, and Sanpo-D VLM spatial QA (2026).

### Cited Findings
- Waghmare, Wilber, Hawkey, Sirotenko et al., "SANPO: A Scene Understanding, Accessibility and Human Navigation Dataset", WACV 2025, pp. 7866-7875, DOI 10.1109/WACV61041.2025.00764. Earlier preprint title: "...Navigation, Pathfinding, Obstacle Avoidance Dataset".
  - 701 stereo real videos (≥30 s, 4 US locations) with per-frame depth, 112k frames with temporally consistent video panoptic labels, and 1961 synthetic videos.
  - The dataset is used by Project Guideline to train mobile models.
  - **`related_work.md` cites the arXiv preprint title and lists the authors as "Waqar, Z., et al."; the actual first author is Sagar M. Waghmare. Fix before drafting.**
  - — [arXiv:2309.12172](https://arxiv.org/abs/2309.12172); [WACV 2025 PDF](https://openaccess.thecvf.com/content/WACV2025/papers/Waghmare_SANPO_A_Scene_Understanding_Accessibility_and_Human_Navigation_Dataset_WACV_2025_paper.pdf)
- **Sanpo-D** ("Spatial-Conditioned Reasoning in Long-Egocentric Videos", arXiv:2601.18100, 2026). Fine-grained re-annotation of 647 SANPO samples for navigation-oriented spatial QA, used to benchmark VLMs. Depth-fused inputs help on safety-critical categories, and VLM-3R had the best pedestrian-detection accuracy (0.8487). — [arXiv:2601.18100](https://arxiv.org/html/2601.18100); [dataset listing](https://awesomepapers.io/multimodal/datasets/sanpo-d)
- Citing works found: mmWalk (NeurIPS 2025 D&B), synthetic augmentation for robotic mobility aids (arXiv:2409.11164), Co-Ego guide-dog robot (arXiv:2603.20121). — [search listing](https://arxiv.org/pdf/2510.11520); [arXiv:2409.11164](https://arxiv.org/pdf/2409.11164); [arXiv:2603.20121](https://arxiv.org/pdf/2603.20121)
- Mobilio's reference snippets indicate SANPO/Cityscapes segmentation training (unconfirmed). — [Pith](https://pith.science/paper/2605.29120)

### Inferences
- **CPE could position its SANPO-derived arrival labels plus the official-split evaluation protocol as a new benchmark task** ("hazard arrival forecasting on SANPO"). Release code and labels so others can compare, which is attractive to WACV/CVPR-workshop and D&B-style reviewers. Novelty here looks real given what was found, but a direct Google Scholar "cited by" sweep has not been done.
- For P1.1 segmentation, report mIoU on the SANPO panoptic classes in the same protocol as the SANPO paper so the numbers are comparable. The SANPO paper's exact baseline numbers were not extracted here.
- Dataset-depth caveat: SANPO depth comes from ZED stereo, not iPhone LiDAR. Mobilio reports about 8% LiDAR distance error. The planned depth-truncation/noise experiment (P2.5) is the way to pre-empt "your depth is unrealistic".

### Gaps
- No full "cited-by" list for SANPO was obtained (Google Scholar not queried). There may be 2025-26 segmentation/depth papers using SANPO that were not found.
- The SANPO paper's official benchmark numbers (depth AbsRel, panoptic PQ/mIoU baselines) were not extracted.

## Q5. Venues and whether they require BLV user studies

### Takeaway
ASSETS 2026 softened its rule. It now "recommends" validation with representative users "when possible" and allows primarily technical contributions grounded in prior formative work, but reviewers still ask whether claims are supported "with the target users". CHI/ASSETS/IMWUT papers making safety claims almost always include BLV studies. CVPR/ICCV accessibility workshops (AVA, CV4A11y) and WACV accept dataset/benchmark/algorithm papers without user studies. The realistic path is a CV-workshop or benchmark-style paper now and an ASSETS/IMWUT paper after a BLV study.

### Cited Findings
- **ASSETS 2026** (Porto, Oct 25-28 2026; technical-paper deadline Apr 22 2026).
  - The call says: "We recommend that papers with a technical contribution be validated through research activities with representative users when possible". Projects may build on in-depth formative work or prior research with representative users, so a paper's key contributions may be primarily technical.
  - Reviewing guidelines ask whether claims are supported "by proper background and proper evaluation (with the target users…)" and whether results "could make a difference in the lives of people with disabilities".
  - — [ASSETS 2026 CFP](https://assets26.sigaccess.org/call_for_papers.html); [CFP mirror with full text](https://www.myhuiban.com/conference/893?lang=en_us); [Technical papers page](https://assets26.sigaccess.org/technical_papers.html)
  - Contrast with ASSETS 2019: technical papers "without being validated through research activities with representative users are unlikely to be accepted". — [CFP mirror / search summary](https://www.myhuiban.com/conference/893?lang=en_us)
- **CVPR AVA workshop: "Accessibility, Vision, and Autonomy Meet"** (organizers Ohn-Bar, Gurari, Kacorri, Kitani, Asakawa, Mankoff). Editions at CVPR 2022/2023/2024/2025, with challenges: 2023 synthetic instance segmentation/pose with accessibility classes; 2025 a multimodal challenge with synthetic and real benchmarks. — [CVPR 2025 AVA](https://cvpr.thecvf.com/virtual/2025/workshop/32372); [workshop site](https://accessibility-cv.github.io/)
- **ICCV 2025 CV4A11y**: First Workshop on Vision Foundation Models and Generative AI for Accessibility. — [ICCVW 2025 intro paper](https://openaccess.thecvf.com/content/ICCV2025W/CV4A11y/papers/Tian_Introduction_to_the_First_Workshop_on_Vision_Foundation_Models_and_ICCVW_2025_paper.pdf)
- **IMWUT** published BlinkBud (2025) with real-world prototype experiments and power measurement. Its abstract mentions no BLV user study. — [arXiv:2512.01366](https://arxiv.org/abs/2512.01366)
- **IEEE RA-L / IROS** host egocentric-forecasting and TTC work evaluated purely on datasets (Schöller 2020; Qiu 2022; Manglik IROS 2019). — [arXiv:1903.07933](https://arxiv.org/abs/1903.07933); [arXiv:2111.00993](https://arxiv.org/abs/2111.00993); [arXiv:1903.09102](https://arxiv.org/pdf/1903.09102)
- **High-impact journals** (Nature Comms, Nature BME, JAMA Ophthalmology) publish assistive systems only with human evaluation: WOAD with volunteers, Mobilio n=14 BVI, Pundlik RCT n=31. — sources above
- **Blindfolded sighted participants** are still used in robotics venues (Co-Ego 2026), but this is contested in accessibility research. — [arXiv:2603.20121](https://arxiv.org/abs/2603.20121)
- **NeurIPS Datasets & Benchmarks** accepted mmWalk, a synthetic BLV walking benchmark with no user study. — [arXiv:2510.11520](https://arxiv.org/abs/2510.11520)

### Inferences
- Venue fit for CPE as it stands (replay only, no users):
  1. **CVPR/ICCV accessibility workshops (AVA, CV4A11y) or a WACV main-track applications paper.** Best fit. The framing is "hazard-arrival forecasting benchmark on SANPO + on-device engine", and no user study is required.
  2. **IEEE RA-L / IROS (with ICRA/IROS assistive-robotics workshops).** Fits the forecasting angle. Needs the strong CVM/Kalman and learned-forecaster baselines from Q2.
  3. **IMWUT.** Needs on-device latency and power plus some real-world data. The SANPO replay alone is probably thin for IMWUT. A small real iPhone capture would help, and the memory note says no hardware demo is planned.
  4. **ASSETS/CHI/TACCESS.** Technically possible under the 2026 wording if CPE grounds design choices (alert budget, verbosity) in prior BLV formative research: the 2025 scoping review's obstacle-detection preference, Mobilio's 112-person survey, and Pundlik's findings. Reviewers will still likely ask for BLV input. Safety or usability claims are not acceptable without one.
  5. **Clinical/Nature-family journals.** Out of reach without a controlled study with contacts/collisions as the outcome.
- The cheapest "user evidence" that strengthens an HCI submission without a full field study:
  - A remote formative study or survey with BLV participants on tolerable alert rates and phrasing, which directly calibrates the 17 alerts/min operating point.
  - Or a listening study with BLV participants rating replayed audio sequences from v1 vs v2 on SANPO clips.
  - Neither involves walking or safety risk. Ethics approval is still needed.
- Statements CPE should avoid in any venue: "safer", "reduces collisions", "usable by BLV people", "appropriate alert rate". Statements CPE can make: held-out predictive superiority over the named physics baselines, precision and recall at alert-rate operating points in open-loop replay, a reproducible benchmark, and on-device feasibility.

### Gaps
- CHI 2026/2027 and IMWUT reviewing criteria were not fetched, and TACCESS author guidelines were not checked.
- The AVA 2025 challenge tasks and the deadlines for 2026/2027 workshops were not found.
- No primary source was found for the formal accessibility-community critique of blindfolded simulation (e.g. Sears & Hanson; Bennett & Rosner CHI 2019, not verified this session).
