# CPE v2 evaluation methodology: alert operating point, path labels, lead time, VLM referee validity

Context, from the repo docs (no source needed, internal): replay of SANPO's official test split (71 sessions, 88 arrivals). v2-GRU: precision 0.59 [0.47, 0.69], 17 alerts/min, warned 0.26 [0.18, 0.49]. v1-K0: precision 0.24, 119 alerts/min, warned 0.65. Median lead time is about 0 s for every engine. The label is "within 3 m inside a ±30° cone within 4 s". Thresholds were picked from a 6-value grid by validation F1 of (warned, precision). Planned referee: 3 VLMs plus 100-150 human-labelled frames scored with Cohen's κ (gates: κ<0.4 reject, κ>0.6 accept).

---

## Q1. What do the alerting / alarm-fatigue literatures say about acceptable alert rates and false-alarm tolerance (blind users, ADAS FCW)?

### Takeaway
No source gives a validated "alerts per minute" target for blind pedestrians. The ADAS and alarm literatures agree on three points. (a) Users discount alarms in proportion to their false-alarm rate. (b) Users switch off or desensitise systems that alert too often, even at rates far below CPE's. (c) Deployed FCW achieves about 98% alert validity. CPE should therefore report the full trade-off and treat alert rate and precision as first-class constraints, not as by-products of F1.

### Cited Findings
**ADAS / FCW (strongest quantitative evidence)**
- NHTSA naturalistic study (Nodine et al., May 2019, DOT HS 812 611), FCW field test:
  - **98% of FCW alerts were for in-path vehicles**. All out-of-path ("false") categories together were 2% of alerts.
  - Validity was 99% on straight roads vs 86% on curves, and 99% for moving targets vs 41% for stopped targets.
  - **22% of in-path alerts came when the driver was already braking.** The report classes these as likely nuisance alerts.
  - Automatic braking was only 26% valid.
  - Source: [NHTSA DOT HS 812 611 (2019)](https://rosap.ntl.bts.gov/view/dot/41002/dot_41002_DS1.pdf)
- The same report: male drivers received "a fairly high rate of FCW alerts (around **4 alerts for every 100 miles**)". Long- and medium-term drivers' alert rates fell over time **because drivers changed the setting to be less sensitive**, "likely … due to a desire to receive fewer alerts." [NHTSA DOT HS 812 611 (2019)](https://rosap.ntl.bts.gov/view/dot/41002/dot_41002_DS1.pdf)
- The same report: the FCW system offered user-selectable sensitivity settings, and alert rate was analysed per setting. This is precedent for shipping several operating points rather than one. [NHTSA DOT HS 812 611 (2019)](https://rosap.ntl.bts.gov/view/dot/41002/dot_41002_DS1.pdf)
- Parasuraman et al. are cited as arguing that collisions are rare, so even precise collision alerting will produce many false alarms, and that driver acceptance needs a false-alarm rate "ideally indistinguishable from zero". This is secondary, via the 2022 *Safety* review "Evaluating the Associations between FCW Severity and Driving Context". [Safety 8(1):5, 2022](https://www.mdpi.com/2313-576X/8/1/5) (the page returned 403 to the fetcher, so this is search-snippet level).
- Truck FCW is reported as underused largely because of an **18% false-alarm rate** (Grove et al., 2016). False alarms lead drivers to ignore warnings (12%) or stop using the system (11%) (Stevens, 2016). Secondary, cited in [Accident Analysis & Prevention 2023 truck FCW paper](https://www.sciencedirect.com/science/article/abs/pii/S0001457523004979).
- Alarm timing interacts with false alarms (simulator study). Drivers given early alarms tend to respond to false alarms and respond late to critical events when an alarm is missed. Drivers given late alarms are reluctant to respond to false alarms. [Behaviour & Information Technology 25(5), 2006](https://www.tandfonline.com/doi/abs/10.1080/01449290500167824)
- A Chalmers thesis separates **false** warnings (no real threat) from **nuisance** warnings (a real threat the driver already knows about). It argues that annoyance comes from an excess of both. [Chalmers, "Nuisance vs False Collision Warnings During Driving"](https://publications.lib.chalmers.se/records/fulltext/118609.pdf)
- Standard CWS effectiveness metrics include "false alarms per mile driven or per hour of driving", with a threshold on threat level that sets the trade-off. [US patent 6853919](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/6853919)

**Cry-wolf / probability matching (foundational, pre-2018)**
- Bliss, Gilson & Deaton (1995), Ergonomics 38(11):2300-2312. About 90% of participants did not respond to every alarm. Instead they **matched their response rate to the alarm's true-alarm probability** (tested at 25/50/75% reliability). [DOI 10.1080/00140139508925269](https://doi.org/10.1080/00140139508925269); details via [Bliss dissertation, UCF](https://stars.library.ucf.edu/rtd/3614/)
- Bliss & Dunn (2000), Ergonomics 43(9), replicated probability matching under workload. [T&F](https://www.tandfonline.com/doi/abs/10.1080/001401300421743)
- Aviation (TCAS): false alarms "may cause pilots to lose trust in the system and ignore alerts", and missed or late detections lead to collisions. [Kuchar & Drumm, MIT Lincoln Lab](https://www.ll.mit.edu/sites/default/files/publication/doc/traffic-alert-collision-avoidance-system-kuchar-ja-11284.pdf)

**Blind / low-vision assistive systems (qualitative only)**
- BBeep (Kayukawa et al., CHI 2019) beeps **only when a future collision is predicted**, because "constant emission of loud alarm sounds can be socially disruptive and make the blind user feel overly self-conscious". Participant P3 valued that "it doesn't beep when there are no people". Participants would not use it in quiet places (hospitals, libraries). n=6 blind participants in an airport. [BBeep CHI'19 preprint](https://wotipati.github.io/projects/BBeep/paper/CHI'19_BBeep_preprint.pdf); [ACM DOI 10.1145/3290605.3300282](https://dl.acm.org/doi/fullHtml/10.1145/3290605.3300282)
- On walking assistants: generating VLM reminders frame by frame or at fixed intervals "will produce a substantial amount of temporal redundancy for the user, resulting in a suboptimal user experience". WalkVLM adds a module that decides *when* to speak and scores it with a Temporal Redundancy F1 (TRF). [WalkVLM, arXiv 2412.20903 (2024)](https://arxiv.org/abs/2412.20903). A follow-up adds a scene-risk discriminator to skip unnecessary reminders. [WalkVLM-LR, arXiv 2508.16070 (2025)](https://arxiv.org/abs/2508.16070)
- Auditory interfaces: continuous audio "may contribute to auditory overload", and moving spatial information to haptics can reduce reliance on continuous audio. [AIDEN, arXiv 2511.06080 (2025)](https://arxiv.org/pdf/2511.06080). Headphones can mask ambient sound. "Delays or overlapping communications can confuse users and obscure crucial environmental cues". [NavVI, arXiv 2507.15072 (2025)](https://arxiv.org/pdf/2507.15072)
- Early ETAs were criticised for over-tasking users, who must decode device output while also attending to ambient sound. Speech and vibrotactile feedback performed similarly, and the authors recommend user-selectable modality. [PLOS ONE 2017, feedback modalities for wearable visual aids](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0170531)
- Why head-level warnings matter: Manduchi & Kurniawan (2011) surveyed more than 300 blind or legally blind respondents about head-level and fall accidents. Secondary reports say 13% had head-level accidents more than once a month and 23% of head injuries needed medical attention; another secondary source says 33%, and this conflicts. [UCSC tech report UCSC-SOE-10-24](https://tr.soe.ucsc.edu/sites/default/files/technical-reports/UCSC-SOE-10-24.pdf); [Semantic Scholar](https://www.semanticscholar.org/paper/Mind-Your-Step-%3A-Mobility-Related-Accidents-by-with-Manduchi-Kurniawan/609b3d62282eaa5c2e4a89c803c0b401a789e27d)

### Inferences
- **The scale mismatch is the headline.** FCW drivers turned sensitivity down at roughly 4 alerts per 100 miles, which is about 1-2 alerts per hour of driving at typical speeds. v2-GRU's 17 alerts/min is about 1,000 per hour, and v1's 119/min is about 7,000 per hour. Pedestrian contexts are denser than highways, so a direct transfer is unfair. Still, the literature gives no support for rates of tens per minute, and the WalkVLM/BBeep line treats *silence when nothing is relevant* as a design goal.
- **Probability matching turns precision into a predicted compliance rate.** If users comply at about the alert's precision, v1 (0.24) would be heeded about a quarter of the time and v2-GRU (0.59) a bit over half. That is the strongest citable argument for a precision floor. Note this is about announcement precision, not recall.
- **Separate "false" from "nuisance" alerts in the replay metrics**, following NHTSA's split:
  - *False:* the object never enters the path.
  - *Nuisance:* the object is in the path but was already announced and is unchanged, or it is already inside the zone (the walker "is already braking").
  - Today's precision counts "inside the zone now" as correct. Under NHTSA's taxonomy many of those would be nuisance, so precision is optimistic as a usefulness measure.
- **Recommended reporting (operating point):**
  1. Report a curve, not a point (see Q2).
  2. Pick operating points by *constraint*, not F1. Example: "maximise warned subject to false alerts/min ≤ B" for 2-3 budgets (say B ∈ {2, 5, 10}/min), chosen on validation, then frozen for test. This mirrors FCW's user-selectable sensitivity levels ("quiet / default / sensitive").
  3. Break alerts/min into true / nuisance / false per minute.
  4. Report alerts per minute *conditional on scene density* (objects in view), since a flat budget is unfair across crowded and empty scenes.
- F1 of (warned, precision) has no human-factors justification. It weights a missed arrival and a false alert equally, while the cited literature says false alerts carry a compounding trust cost.

### Gaps
- I found **no published quantitative tolerance (alerts/min or false-alarm rate) for blind pedestrians**. Neither ASSETS/CHI/TACCESS papers nor commercial-ETA studies gave one. The searches found qualitative evidence only, so this has to be stated as an open question, not imported.
- I could not fetch the ISO 15623 text (FCW performance requirements), so its warning-timing and false-alarm clauses are not verified here.
- The MDPI *Safety* 2022 review and the Chalmers thesis were seen only at snippet level (403 / not fetched).
- The Manduchi & Kurniawan percentages (23% vs 33% needing medical attention) conflict across secondary sources. Check the primary report before quoting.

---

## Q2. How do papers report warning systems: PR/ROC over thresholds, SOC/DET curves, time-to-collision at warning, lead-time distributions? Curve or single point?

### Takeaway
The collision-alerting field's native tool is the **System Operating Characteristic (SOC) curve** (Kuchar 1996): probability of a successful (timely) alert vs probability of an unnecessary alert, swept over the alert threshold. Accident-anticipation work in vision reports **AP over a threshold sweep plus time-to-accident at fixed recall** (mTTA, TTA@R80). CPE should present an operating curve with 2-3 marked points and lead time measured *at matched recall or alert rate*.

### Cited Findings
- Kuchar, "Methodology for Alerting-System Performance Evaluation", *J. Guidance, Control, and Dynamics* 19(2), 1996, introduced the SOC curve: "a plot showing the tradeoff of false alarms for collisions", generated by varying one alerting parameter. [AIAA DOI 10.2514/3.21637](https://arc.aiaa.org/doi/pdf/10.2514/3.21637); description in [Winder & Kuchar, NASA/CR-2000-210099](https://ntrs.nasa.gov/api/citations/20000085966/downloads/20000085966.pdf)
- Yang & Kuchar (JGCD 20(4), 1997): the SOC curve "explicitly shows the expected safety level and unnecessary alert rate as a function of the alert threshold". Its shape depends on sensor accuracy, path uncertainty and human performance, so design changes are compared by how they *shift the curve*. [Yang & Kuchar 1997 PDF](https://www.mit.edu/~jkkuchar/Y&K1997.pdf)
- The SOC builds on signal detection theory and accounts for sensor, threshold, display, operator and dynamics uncertainty. [Kuchar & Hansman, IFAC 1995](https://www.sciencedirect.com/science/article/pii/S1474667017452785)
- Accident anticipation from dashcam video (DAD/CCD/A3D benchmarks) uses three standard metrics:
  - **AP** over a threshold sweep.
  - **TTA**: time from the first threshold crossing to the event.
  - **mTTA**: TTA averaged over thresholds.
  - **TTA@R80**: TTA at the threshold that gives 80% recall.
  - Example numbers on DAD: ROAR AP 79.5 / mTTA 3.72 s / TTA@R80 3.62 s, against older methods at about 1.3-1.9 s mTTA. Sources: [ROAR, arXiv 2511.06226](https://arxiv.org/html/2511.06226); [CRASH, arXiv 2407.17757](https://arxiv.org/pdf/2407.17757)
- A known caveat: a high TTA "might be misleading if the model overfits … resulting in indiscriminate positive predictions". For this reason one 2025 paper reports **mTTA only under a false-positive-rate cap** (mTTA^λ with FPR ≤ λ) and AUC at fixed horizons. [Accident Anticipation via Temporal Occurrence Prediction, arXiv 2510.22260](https://arxiv.org/html/2510.22260)
- NHTSA's FCW evaluation reports **distributions** rather than single numbers: range at alert onset, a cumulative TTC distribution per sensitivity setting, and headway at alert. [NHTSA DOT HS 812 611 (2019)](https://rosap.ntl.bts.gov/view/dot/41002/dot_41002_DS1.pdf)
- Walking-assistant VLM work scores "when to speak" separately from "what to say" (TRF vs ROUGE/GPT-4 score). [WalkVLM (2024)](https://arxiv.org/abs/2412.20903)

### Inferences
- **Recommended main figure:** an SOC-style curve per engine.
  - x-axis: false (or unnecessary) alerts per minute, log scale.
  - y-axis: warned (recall of arrivals).
  - Sweep the GRU ignore threshold densely (for example 20-50 values, not 6) *and* the announcer constants (cooldown, hysteresis).
  - Mark the validation-chosen points. Overlay v1 and physics as points or curves.
  - This is the FROC analogue (sensitivity vs false positives per unit time). It is more honest than one F1-chosen point, given that the warned CI [0.18, 0.49] is very wide.
  - The point (0.26 warned, 17/min) vs (0.65, 119/min) cannot show whether v2 *dominates* v1. A curve can: if v2's curve passes above v1's point, v2 dominates.
- Also report **PR-AUC (AP) of the per-frame arrival score**. With few positives, AP is the standard for imbalanced anticipation tasks. Report it alongside the existing AUROC.
- Report lead time the way anticipation papers do, as **TTA at matched recall** (e.g. lead@warned=0.5) or **at matched alert budget**, never at engine-specific thresholds. Otherwise a chattier engine trivially gets more lead (v1's 0.6 s median vs v2's 0 s is partly this).
- Bootstrap CIs on the curve: resample sessions, recompute the whole curve, and draw pointwise bands. This keeps the existing session-level bootstrap (§4.8).

### Gaps
- I did not retrieve the full text of Kuchar 1996 itself, only descriptions in Yang & Kuchar 1997 and the NASA report.
- I found no assistive-navigation paper that reports a full SOC/PR curve for user alerts. Most report single-condition user studies.

---

## Q3. How should "in my path" be defined: corridor width, personal space, collision course via CPA/TCPA? What should replace the ±30° cone?

### Takeaway
Pedestrian-dynamics and assistive-collision work defines "collision course" in two ways. One is **predicted minimum distance (closest approach) under constant-velocity extrapolation** within a time horizon. The other is crossing a **narrow body-width line or corridor in front of the user** (BBeep: an "emergency line" of about 0.7 m, with 2.5 s and 5 s horizons). Body-width figures are about 0.46 × 0.61 m (Fruin body ellipse), with about 0.71-0.76 m lateral spacing for comfortable walking. A ±30° cone is about 3.5 m wide at 3 m, roughly 5× too wide.

### Cited Findings
- **BBeep (CHI 2019):**
  - Predicts each pedestrian's future position by linear extrapolation of tracked 3D positions (with camera-rotation compensation). It flags a collision when the predicted path segment **intersects an "emergency line"** in front of the user. Figures annotate the line as 0.7 m.
  - Two horizons: **s1 = 5.0 s** (time to close about 10 m of detection range at a relative 4 km/h) and **s2 = 2.5 s**.
  - Three-stage policy: a low-urgency beep for a collision within 5 s, an intermediate one within 2.5 s, and a stop sound.
  - In a 399-trajectory corridor study, 5 s timing deflected pedestrians more than 2.5 s; sound type had no significant effect. The outcome measure was "minimal distance" between suitcase and trajectory.
  - Source: [BBeep CHI'19 preprint](https://wotipati.github.io/projects/BBeep/paper/CHI'19_BBeep_preprint.pdf)
- **Minimum Predicted Distance (MPD)**, the future closest-approach distance from linear extrapolation of each walker's position, heading and speed. It is the standard variable in walker-walker avoidance studies. Walkers anticipate collision risk and adapt, with role-dependent strategies (the walker giving way contributes more). [Collision avoidance between two walkers: role-dependent strategies, Gait & Posture 2013](https://www.sciencedirect.com/science/article/abs/pii/S0966636213001768)
- Walkers can predict the future closest-approach distance even for curved paths (5 m and 10 m radius), with few collisions or crossing-order changes (n=22, VR). [Exp Brain Res 239(1):329-340, 2021](https://pubmed.ncbi.nlm.nih.gov/33175191/)
- **Body and lateral spacing:**
  - Fruin's body ellipse is 45.7 × 61.0 cm (0.28 m²).
  - "People require a lateral space of 28 to 30 inches [0.71-0.76 m] … for comfortable movement". Longitudinal walking spacing is 8-10 ft (2.4-3.0 m).
  - Lateral spacing below 60 cm between strangers occurs only in jammed conditions.
  - Sources: [NYC DCP Pedestrian LOS appendices](https://www.nyc.gov/assets/planning/download/pdf/plans/transportation/td_pedlosappendices.pdf); [Pushkarev & Zupan, TRB Record 538 (1975)](https://onlinepubs.trb.org/Onlinepubs/trr/1975/538/538-001.pdf)
- Two people walking side by side or passing need almost 1.5 m. [FHWA HRT-05-137](https://www.fhwa.dot.gov/publications/research/safety/pedbike/05137/chapter2.cfm)
- The HCM review recommends a 0.75 m² body buffer zone for walking. [FHWA HCM Ch.13 recommended procedures](https://highways.dot.gov/media/8521)
- ADAS precedent: FCW validity is defined by whether the target is **in-path (in-lane)**, and out-of-path targets count as false. False alerts rise on curves (86% vs 99% in-path) because path prediction fails. [NHTSA DOT HS 812 611 (2019)](https://rosap.ntl.bts.gov/view/dot/41002/dot_41002_DS1.pdf)
- Predictive pedestrian simulation models use time-to-collision fed into an urgency function with a maximum anticipation time. [Real-time Collision Avoidance for Pedestrian and Bicyclist Simulation](https://www.researchgate.net/publication/260016683_Real-time_Collision_Avoidance_for_Pedestrian_and_Bicyclist_Simulation_A_Smooth_and_Predictive_Approach)

### Inferences
- **Cone geometry problem:** at 3 m a ±30° cone has lateral half-width 3·tan30° ≈ 1.73 m, so it is about 3.5 m wide. That is about 5× BBeep's 0.7 m line and more than 2× FHWA's two-person passing width. Combined with the camera's 51-60° HFOV, nearly every visible object within 3 m counts as "in the path". This explains both the low lead time and the weak separation between "nearest in cone" and the GRU.
- **Recommended replacement label (tiered, all computed from SANPO depth + camera pose, still formula-free):**
  1. **Path frame.** Define the walking direction from the camera pose's ego displacement over about 1 s, not the head's optical axis. SANPO is head-mounted, so head yaw ≠ walking direction. Project each object to (lateral x, forward z) in that frame using depth and bearing.
  2. **Corridor-entry label ("in my path"):** the object's *measured* future position enters |x| ≤ w/2 + r_obj with 0 < z ≤ D within H.
     - w ≈ 0.75 m (Fruin lateral spacing) as the primary value; sensitivity grid w ∈ {0.7, 1.0, 1.5} m (BBeep / FHWA passing width).
     - r_obj: half the object's width from its bbox and depth, if available.
     - D = 3 m, H = 4 s as now, plus H = 2.5 s and 5 s to match BBeep's two stages.
  3. **CPA label (for moving objects):** from measured future relative tracks, min-distance (MPD) < r_safe (e.g. 1.0 m) with time-to-CPA ∈ [0, H]. This captures crossing pedestrians and cyclists who never sit "in front" for long.
  4. **Keep the 3 m cone as a separate "awareness" tier.** Do not use it as the "hazard" tier. This gives a two-level urgency structure analogous to BBeep's low and intermediate beeps.
- Report the existing (H, D, θ) sensitivity grid extended with w. Label prevalence will drop sharply under the corridor label, so expect wider CIs. Check how many test arrivals remain (88 now). The 25 test sessions with arrivals may be too few, and pooling validation and test for label studies may be needed.
- Stationary obstacles (poles, bollards) can only enter the corridor through ego-motion. That makes the label depend on the walker's recorded path, which is open-loop-valid only for that trajectory (see Q6).

### Gaps
- I could not retrieve primary text for the 2012 "minimal predicted distance" metric paper (Olivier et al.), only the 2013 role-dependent follow-up and the 2021 curvature study. No numeric "comfortable MPD" threshold was verified.
- I did not verify proxemics (Hall) distances, and I found no 2018-2026 source that gives a blind-pedestrian-specific corridor width.
- The BBeep "0.7 m" value is read from figure annotations in the preprint. The text does not state the emergency-line width explicitly.

---

## Q4. How to evaluate lead time when objects appear late (conditioning on first-seen distance, per-class, survival analysis)?

### Takeaway
Lead time is bounded by the *warning opportunity window* (first detection → arrival). Anticipation papers report lead time at matched recall or a false-positive cap. CPE should report lead time **conditional on objects whose window is long enough** (e.g. first seen ≥ 2 s or ≥ 5 m before arrival), as a fraction of the available window, and as a censored time-to-warning distribution.

### Cited Findings
- TTA is the gap between the first threshold crossing and the event. TTA@R80 fixes recall at 80%. mTTA under an FPR cap prevents "indiscriminate positive predictions" from inflating lead time. Sources: [ROAR (2025)](https://arxiv.org/html/2511.06226); [arXiv 2510.22260 (2025)](https://arxiv.org/html/2510.22260)
- BBeep tied its 5 s horizon explicitly to sensor range: about 10 m of detection range at 4 km/h relative speed. Lead time is capped by detection range divided by closing speed. [BBeep CHI'19](https://wotipati.github.io/projects/BBeep/paper/CHI'19_BBeep_preprint.pdf)
- BBeep found that 5 s timing produced larger minimum distances than 2.5 s. Earlier warnings were behaviourally more effective in its corridor study. [BBeep CHI'19](https://wotipati.github.io/projects/BBeep/paper/CHI'19_BBeep_preprint.pdf)
- FCW reporting uses cumulative distributions of TTC at alert onset per sensitivity setting, not medians. [NHTSA DOT HS 812 611 (2019)](https://rosap.ntl.bts.gov/view/dot/41002/dot_41002_DS1.pdf)
- Early-warning timing changes how operators respond to later false and missed alarms. [Behaviour & IT 25(5), 2006](https://www.tandfonline.com/doi/abs/10.1080/01449290500167824)

### Inferences (recommended protocol, none of it found pre-packaged in the literature)
1. **Define the opportunity window W = t_arrival − t_first_seen** per arrival, and report its distribution first. If the median W is about 0.2-0.6 s, then "median lead ≈ 0 s" is a data property, not an engine property. Say so explicitly.
2. **Conditional lead-time subsets:** report warned and lead only on arrivals with W ≥ {1, 2, 4} s, or first-seen distance ≥ {5, 8} m. Report n per subset: with 88 arrivals the subsets will be small, so give CIs.
3. **Normalised lead = lead / W** (0 = warned at arrival, 1 = warned at first sight). This removes the late-appearance confound and is comparable across classes.
4. **Survival / time-to-event view:**
   - Treat each arrival as a subject at risk from first sighting.
   - The "event" is the first warning, and arrival without warning is right-censoring.
   - Plot a Kaplan-Meier curve of "probability not yet warned" vs time-to-arrival (counting backwards from arrival: 4, 3, 2, 1, 0 s) per engine.
   - This uses every arrival, including unwarned ones, and avoids the median of a mostly-zero distribution.
   - Note: this is a standard method (Kaplan-Meier with censoring); I did not find an assistive-navigation paper using it.
5. **Matched-budget lead:** compare engines' lead at equal alerts/min or equal warned (the TTA@R80 analogue, e.g. lead@warned=0.5), taken from the SOC sweep in Q2.
6. **Per-class:** split arrivals by class (person, vehicle, bicycle, static obstacle) and by mover vs static. Static obstacles "arrive" only through ego-walking, and their window depends on detector range for small objects.
7. **Fixing the data limit:** the window is set by YOLO's detection range on SANPO frames. The lead-time question is partly a detector-range question (small or far objects). A cheap diagnostic is "first-seen distance" by class.

### Gaps
- I found no assistive-tech paper that reports censored or survival-style warning analysis. This is a methodological recommendation, not established field practice.
- I found no empirical source for the minimum lead time a blind pedestrian needs to act on a spoken warning (reaction plus speech duration). BBeep's 2.5/5 s horizons are the closest citable values. Speech announcements take about 1 s to utter, which also eats into lead (inference, not sourced).

---

## Q5. Validity of VLMs as judges/referees: agreement with humans, biases, number of human items, which agreement statistic, kappa CI sample size

### Takeaway
LLM/MLLM judges reach human-level agreement on *pairwise* comparisons in some domains (GPT-4 85% vs human-human 81% on MT-Bench). They are much weaker on scoring and batch ranking, and agreement varies strongly by task. Position bias is pervasive. The planned design (pairwise or forced choice, blinding, human calibration before use) is consistent with best practice. Recommended changes:
- Randomise and swap object order.
- Include a human-human agreement ceiling.
- Use a statistic suited to a variable candidate set (raw agreement + Gwet's AC1 / Krippendorff's α alongside Cohen's κ).
- Size the human set for CI width, not just κ thresholds.

### Cited Findings
**Agreement**
- Zheng et al. (NeurIPS 2023 D&B), MT-Bench / Chatbot Arena: GPT-4 judges reach "over 80% agreement, the same level of agreement between humans". This was 85% vs 81% human-human without ties, on about 3K expert and 3K crowd votes. [arXiv 2306.05685](https://arxiv.org/abs/2306.05685)
- MLLM-as-a-Judge (Chen et al., ICML 2024):
  - MLLMs align with humans on **pair comparison** but differ noticeably on scoring and **batch ranking**.
  - GPT-4V reached 0.636 similarity with human judgments on pair comparison with ties and 0.773 without; Pearson correlation for scoring was 0.490.
  - Consistency on repeated queries is low: 0.611 for scoring, 0.418 for batch ranking.
  - Biases found: egocentric, position, length, and hallucination.
  - Source: [arXiv 2402.04788](https://arxiv.org/html/2402.04788v3.pdf)
- JUDGE-BENCH (Bavaresco et al., ACL 2025; preprint 2024): 11 LLMs on 20 human-annotated datasets. Performance "varies considerably across models and datasets". Agreement depends on the property evaluated and the expertise of the human judges. The authors conclude "LLMs should be carefully validated against human judgments before being used as evaluators." [arXiv 2406.18403](https://arxiv.org/abs/2406.18403)
- Recent 2026 preprints report further MLLM-judge biases: model-preference bias [arXiv 2604.11589](https://arxiv.org/pdf/2604.11589); "informativeness bias", where VLMs judge without using the image [arXiv 2604.17768](https://arxiv.org/pdf/2604.17768); and perceptual judgment bias [arXiv 2606.02578](https://arxiv.org/html/2606.02578v1). Seen at title and abstract level only.

**Biases and mitigations**
- Position bias: all LLM judges tested showed it, mostly favouring the first position. With the default prompt, GPT-4 was consistent in 65% of swapped pairs, Claude-v1 in 23.8% and GPT-3.5 in 46.2%. Mitigation: **swap order and count a win only if it holds in both orders**, or randomise; few-shot prompting raised GPT-4's consistency to 77.5%. [Zheng et al. 2023](https://arxiv.org/abs/2306.05685)
- Verbosity bias: Claude-v1 and GPT-3.5 failed a "repetitive list" attack 91.3% of the time, GPT-4 8.7%. [Zheng et al. 2023](https://arxiv.org/abs/2306.05685). A 2026 audit reports very small verbosity bias for 21 current judges on MT-Bench, while cautioning against generalising. [arXiv 2606.19544](https://arxiv.org/pdf/2606.19544)
- Position bias is strong for LLaVA and CogVLM, and GPT-4V and Gemini show length bias. [MLLM-as-a-Judge](https://arxiv.org/html/2402.04788v3.pdf)

**Agreement statistics and sample size**
- Sim & Wright (2005), *Physical Therapy* 85(3):257-268, is the standard methods reference:
  - Kappa magnitude depends on **prevalence and bias**.
  - Landis-Koch benchmarks are "arbitrary".
  - CIs should be reported.
  - Sample sizes are tabulated by null κ, target κ, and proportion of positive ratings.
  - Example from a study using the tables: κ0 = 0.40 vs κ1 = 0.80 with 50% positives, 80% power, two-tailed gives **≥41 items**. Another needed 175 observations to separate κ 0.60 from 0.40 with 90% prevalence.
  - Sources: [Sim & Wright 2005 (Birmingham record)](https://research.birmingham.ac.uk/en/publications/the-kappa-statistic-in-reliability-studies-use-interpretation-and/); examples via [search summary of studies citing the tables](https://www.scribd.com/document/408146084/The-Kappa-statistic-in-reliability-studies-Use-interpretation-and-sample-size-requirements-Sim-Wright-2005-pdf)
- Kappa paradox: under skewed marginals, high raw agreement can yield low κ. Gwet's AC1 is less affected by prevalence. Sources: [PMC5712640 (2017)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5712640/); [Wongpakaran et al. 2013, PMC3643869](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3643869/)
- AC1's chance definition is contested. One guide recommends reporting AC1 *alongside* κ and raw agreement, and warns that switching coefficients after a disappointing κ is "coefficient shopping". [CASRAI guide](https://casrai.org/guides/inter-rater-reliability-choosing-a-coefficient)
- Landis-Koch labels are inappropriate for AC1. ICC, Krippendorff's α and AC1 are options for nominal data with 2+ raters. [Geijer et al., Acta Radiologica 2025](https://journals.sagepub.com/doi/10.1177/02841851251319666)
- Krippendorff's α conventions: rely on α ≥ 0.800, draw tentative conclusions at 0.667-0.800, discard below 0.667 (secondary guide, citing Krippendorff's conventions). [Kappa paradox guide](https://edu.koji.so/docs/agreement-coefficients-kappa-paradox-gwet-ac1-krippendorff-course-feedback-coding)

### Inferences
- **The planned task fits the stronger regime.** A pairwise or forced-choice "which object is the top threat" is closer to MLLM-as-a-Judge's pair comparison (good agreement) than to batch ranking (poor). Do *not* ask VLMs to rank the full object list.
- **Mandatory controls:**
  1. Present the object list in randomised order and run each frame twice with the order reversed. Count a VLM pick only if it is order-consistent, and report the consistency rate (as in Zheng et al.).
  2. Mark candidates on the image (numbered boxes) so the judge must look at the image. This addresses "informativeness bias", where VLMs answer from text alone.
  3. Run a **text-only ablation**: the object list without the image. If VLM agreement barely changes, the referee is reading the distance column, not judging the scene.
  4. Keep numeric fields (distance, closing speed) identical in format to avoid length or format cues.
- **Statistic choice.** The referee picks one of k objects, and k varies per frame. Cohen's κ assumes a fixed category set and fixed marginals, so it is ill-defined or awkward here. Recommended:
  - Report **raw agreement** with a per-frame chance baseline of 1/k_i (mean over frames).
  - Report a **chance-corrected agreement using per-item chance**, which is a Bennett-S / AC1-style correction.
  - Report **Krippendorff's α (nominal)** for the 3-VLM + human panel, since α handles more than 2 raters and missing ratings.
  - If the question is reduced to binary "does arm A's pick match the referee", Cohen's κ applies directly.
  - Pre-register the statistic so the choice cannot be made after seeing results.
- **Human ceiling.** Have **2 humans** label at least a 50-frame subset. Human-human agreement is the ceiling for VLM-human agreement, as in MT-Bench, and "top threat" may be genuinely ambiguous. If human-human κ is about 0.5, a VLM gate of κ>0.6 is unreachable by construction. Express the gate relative to the human-human ceiling.
- **Sample size (approximate, standard large-sample SE).** SE(κ) ≈ √(p_o(1−p_o)) / ((1−p_e)√n).
  - With p_o = 0.8 and p_e ≈ 0.33 (about 3 candidates): n = 100 gives a 95% CI of about ±0.12; n = 150 gives about ±0.10; n = 300 gives about ±0.07.
  - At n = 100-150, a κ point estimate of 0.6 has a CI that straddles 0.5-0.7, so it cannot cleanly clear a 0.6 gate. Either frame the gate as "lower CI bound > 0.4", or raise n toward about 200-300.
  - Use a frame-level bootstrap clustered by session for the CI, consistent with §4.8.
  - This is my calculation from the standard formula, not a sourced number.
- **Disagreement-frame selection inflates difficulty.** Frames where engines disagree are the hardest. Agreement measured there will be lower than on random frames, so report it as such and do not generalise.
- **Correlated VLM errors:** the repo's stance (majority vote is not a substitute for calibration) is supported by JUDGE-BENCH's "validate per task" conclusion and by shared-bias findings (position and length biases recur across model families).

### Gaps
- I found no published VLM-as-judge study on *hazard ranking or egocentric safety judgments* specifically, so domain transfer of the 70-85% agreement figures is unverified.
- I did not verify Sim & Wright's tables directly (paywalled). The quoted n values come from studies applying them.
- I found no source giving a standard "number of human-labelled items" for LLM-judge validation. Practice ranges from tens to thousands; JUDGE-BENCH datasets and MT-Bench's 3K votes are much larger than 150.

---

## Q6. Limitations of open-loop / simulation-only evaluation; how assistive-tech papers justify results without blind participants

### Takeaway
Open-loop replay metrics can be uncorrelated or even *negatively* correlated with closed-loop performance (Dauner et al., CoRL 2023). Assistive-tech precedents without blind-user studies rely on dataset metrics (WalkVLM) or on sighted proxy behaviour (BBeep's corridor study). CPE should present replay results as *alert-policy* properties (precision, rate, stability, opportunity) and explicitly not as safety outcomes.

### Cited Findings
- Dauner, Hallgarten, Geiger & Chitta, "Parting with Misconceptions about Learning-based Vehicle Motion Planning" (CoRL 2023):
  - Found a **negative correlation between open-loop and closed-loop scores** on nuPlan.
  - Open-loop evaluation "lacks dynamic feedback". In open-loop the ego is replayed from human logs, so the planner's output does not influence the scene.
  - The best open-loop results came from ignoring most scene context.
  - Sources: [arXiv 2306.07962](https://arxiv.org/abs/2306.07962); [PMLR v229](https://proceedings.mlr.press/v229/dauner23a/dauner23a.pdf)
- WalkVLM evaluated a blind walking assistant on a 12k-video annotated dataset using automatic metrics (TRF for temporal redundancy, ROUGE, TF-IDF, a GPT-4 score). This is dataset-only precedent in the same application domain. [arXiv 2412.20903](https://arxiv.org/abs/2412.20903)
- BBeep designed its alert timing from a **sighted-pedestrian observational study**: 399 trajectories, outcome = minimal distance. It then ran a small (n=6) blind-user airport study. [BBeep CHI'19](https://wotipati.github.io/projects/BBeep/paper/CHI'19_BBeep_preprint.pdf)
- A systematic review of 75 wearable obstacle-avoidance ETA studies found trial sample sizes from 1 to 70. Small n is normal in the field. [Xu et al., IEEE Access 2023 (ResearchGate)](https://www.researchgate.net/publication/371517045_Wearable_obstacle_avoidance_electronic_travel_aids_for_blind_and_visually_impaired_individuals_A_systematic_review)
- A 2026 survey notes that most navigation systems focus on output and under-use user input. [Navigation Solutions for BVI Persons survey, IJHCI 2026](https://www.tandfonline.com/doi/full/10.1080/10447318.2026.2643358)
- Precedent from the alerting field: SOC analysis models human response *probabilistically* inside the evaluation instead of running user trials. [Kuchar & Hansman 1995](https://www.sciencedirect.com/science/article/pii/S1474667017452785); [Yang & Kuchar 1997](https://www.mit.edu/~jkkuchar/Y&K1997.pdf)

### Inferences
- **Specific open-loop threats in CPE:**
  1. The recorded walker never heard alerts, so "arrivals" include events a warned user would have avoided, and vice versa.
  2. Ego-motion-induced arrivals (static obstacles) are trajectory-specific.
  3. Inputs and labels share SANPO depth (already noted in the report).
  4. Sighted walkers' paths differ from cane or guide-dog users' paths.
- **Defensible claims without a user study:**
  - (a) Alert-policy statistics: precision, alerts/min, nuisance fraction, topic switches. These are independent of user response *given* the scene.
  - (b) Opportunity-normalised lead time (Q4).
  - (c) SOC curves showing dominance.
- **Not defensible:** collision reduction or safety. State this directly in the paper.
- **Partial closed-loop proxy:**
  - Model user compliance with probability matching (Bliss 1995). Assume a warned user avoids the arrival with probability p = precision of the engine's alerts, or a swept p.
  - Compute "expected avoided arrivals per minute of user burden". This yields a single utility-vs-burden curve in the spirit of Kuchar's SOC with a human-response model.
  - Label it as a model, not a measurement.
- Optionally add a **sighted-proxy or blindfold pilot**, or an **expert review by O&M (orientation & mobility) instructors** of sampled alerts. This is a cheaper intermediate step used in the literature (BBeep used sighted pedestrians for policy design). Inference: I did not find a specific O&M-expert-review precedent in this search.

### Gaps
- I found no paper that quantifies open-loop vs closed-loop correlation for *alerting* (as opposed to planning). Dauner et al.'s negative correlation is for motion planning, so transfer is by analogy.
- I found no assistive-navigation paper that explicitly argues for the validity of replay-only evaluation. Precedents (WalkVLM) simply report dataset metrics.
- The tool-call budget did not allow searching ASSETS/TACCESS 2022-2026 more deeply for simulation-based blind-navigation evaluations (e.g. VR or sighted-blindfolded proxies) and their stated limitations.
