# Related Work and Literature References

Verified external references — every entry below was checked against a live search (title, authors,
year, venue/arXiv ID confirmed, not recalled from memory) on 2026-08-27, so citations here are safe to
carry into a paper draft as-is. Organized by which CPE component or design decision each paper
grounds, with an explicit note on whether CPE currently uses the finding, or whether it motivates
planned-but-not-built work (cross-referenced to `pending_work.md`). This is the paper's future
"Related Work" section in citation-ready form, not prose — write the narrative from `methodology.md`
and cite from here.

Do not add an entry to this file from memory alone. If you need a new citation, verify it against a
live search first — a wrong arXiv ID or misattributed author is worse than no citation.

**Section references.** `architecture.md §N` references in this file were written against the v1
design and point to v1 `architecture.md` (git `a6e37a7`). The v2 design (2026-10-14) still needs
citations for its new parts: Kalman tracking with ego-motion compensation, walkable-surface
segmentation (e.g. SegFormer), and GRU sequence models. Verify and add them before drafting.

---

## 1. Positioning: comparable assistive-navigation systems

The direct comparison class for CPE's Introduction/Related Work section (`methodology.md` Appendix
§A.2 already drafts this comparison; these are the citations to back it).

- **Dakopoulos, D. & Bourbakis, N. G. (2010). "Wearable Obstacle Avoidance Electronic Travel Aids for
  Blind: A Survey."** *IEEE Transactions on Systems, Man, and Cybernetics, Part C*, 40(1), 25–35.
  DOI: 10.1109/TSMCC.2009.2021255.
  Foundational survey of the ETA/EOA/PLD taxonomy for blind-navigation aids and the camera- vs.
  ultrasonic-sensing split. **Use:** cites the "traditional sensory substitution devices lack semantic
  understanding" claim CPE's Introduction already makes — this is the paper that claim should point to.
- **Liu, R. & Slade, P. (2026). "Improving outdoor navigation for people with blindness using an
  AI-driven smartphone application and personalized audio guidance."** arXiv:2605.29120. *Venue
  unconfirmed:* an earlier draft of this file listed *Nature Biomedical Engineering*, which could not be
  verified (2026-10-17); cite the arXiv version until a published version is found.
  A contemporary (2026) Harvard smartphone app ("Mobilio") combining ML, sensor fusion, and
  personalized audio guidance for blind/low-vision outdoor navigation — the closest published system
  to CPE's own premise. Their user survey of 100+ blind/low-vision people identifying turn-by-turn
  directions, continuous path guidance, and obstacle avoidance as the three essential capabilities is
  directly citable as independent validation of CPE's problem framing. **Use:** this is the single most
  important related-work citation to add — CPE's Related Work section should explicitly differentiate
  against it (CPE's hard reflex-latency guarantee and physics-verification arbitration vs. their
  approach) rather than only comparing against older ETA surveys and cloud-VLM systems.

- **Calim, H. et al. (2026). "Safety-oriented sidewalk and road segmentation for smartphone-based
  assistive navigation."** arXiv:2607.21137 (submitted to *Assistive Technology*). SENSATION-DS:
  2,752 chest-height pedestrian-view images, 9 classes; mobile models on Android; road-as-sidewalk
  error 0.079–0.099 at mIoU 0.690–0.715 (281 test images). **Use:** the closest published false-safe
  measure to ours; paper §6.5 compares, with the caveats (different data, taxonomy, no formula given).

## 2. Perception: detection, tracking, and depth

- **Zhang, Y., Sun, P., Jiang, Y., et al. (2022). "ByteTrack: Multi-Object Tracking by Associating
  Every Detection Box."** ECCV 2022. arXiv:2110.06864.
  Associates low-confidence detection boxes with tracklets instead of discarding them, recovering
  occluded/fragmented tracks. **Use:** CPE's tracking stack (`YoloTracker`) uses ByteTrack directly
  (`architecture.md` §11.1) — this is the primary citation for that component, not just a name-drop.
- **Sapkota, R., Karkee, M., et al. (2025). "Ultralytics YOLO Evolution: An Overview of YOLO26,
  YOLO11, YOLOv8 and YOLOv5 Object Detectors for Computer Vision and Pattern Recognition."**
  arXiv:2510.09653.
  Independent (non-Ultralytics) technical overview of the YOLO26 architecture CPE fine-tunes — covers
  DFL removal, NMS-free inference, and the small-target-aware label assignment relevant to detecting
  small hazards (poles, potholes) at range. **Use:** Ultralytics has not published a formal YOLO26
  paper; this is the closest citable technical reference for the detector backbone, and should replace
  any informal "YOLO26n" mention with an actual citation in the paper.
- **Yang, L., et al. (2024). "Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data."**
  arXiv:2401.10891.
  Monocular depth foundation model trained on ~62M unlabeled images, offering strong zero-shot
  generalization. **Use — not yet, planned:** CPE currently uses SANPO's ground-truth depth maps for
  both training and evaluation, which is a deliberate simplification for isolating the kinetic score's
  correctness (`methodology.md` §2.1). A real phone deployment without a depth sensor needs a
  monocular depth model; this is the natural citation and starting point when that work begins
  (`pending_work.md` — not currently tracked as an item; add if physical deployment work starts).
- **Liu, S., et al. (2023). "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set
  Object Detection."** ECCV 2024. arXiv:2303.05499.
  Open-vocabulary detector that can localize arbitrary categories from text prompts, zero-shot.
  **Use:** `architecture.md` §9 names Grounding DINO as an example "massive offline Teacher model" for
  the SLM-1 knowledge-distillation pipeline (generating pseudo-labels/rich descriptions offline for
  SFT data) — this is the citation for that specific claim, not a detector CPE runs at inference time.

## 3. Dataset

- **Waghmare, S. M., et al. — Google Research (2025). "SANPO: A Scene Understanding, Accessibility and
  Human Navigation Dataset."** *WACV 2025*. Preprint arXiv:2309.12172 (2023).
  701 real + synthetic egocentric stereo video sessions with dense depth, panoptic masks, and camera
  pose, purpose-built for outdoor human-navigation assistive technology (used by Google's own Project
  Guideline). **Use:** this is CPE's primary training/evaluation corpus (`sanpo_dataset.md`) — cite
  directly rather than describing it only informally. Note the paper states SANPO is *already used* for
  a comparable assistive application, which is independent validation that it's an appropriate corpus
  choice, worth stating explicitly in a paper's Dataset subsection.

## 4. Kinetic score design: grounding the formula in traffic-safety literature

CPE's `K = severity(c) · v² / d` currently justifies `v²` and `mass^λ` from internal reasoning only
(a value judgment, `methodology.md` §4.7). These two papers are real traffic-safety
literature that independently supports the same functional form, and are worth citing in the paper's
Methodology to show the design isn't arbitrary, even though the specific λ compromise still is.

- **Rosén, E. & Sander, U. (2009). "Pedestrian fatality risk as a function of car impact speed."**
  *Accident Analysis & Prevention*, 41(3), 536–542.
  Derives an empirical pedestrian fatality-risk curve as a function of impact speed from the largest
  in-depth pedestrian accident study at the time. **Use:** direct empirical support for velocity
  (not just proximity) being the dominant driver of collision *consequence* — the same argument
  `methodology.md` §4.1 makes for keeping `v²` over a proximity/TTC-only score, now backed by
  real accident data rather than only the internal architectural argument.
- **Rosén, E., Stigson, H., & Sander, U. (2011). "Literature review of pedestrian fatality risk as a
  function of car impact speed."** *Accident Analysis & Prevention*, 43(1), 25–33.
  Follow-up literature review reaffirming the monotonic risk-vs.-speed relationship while correcting
  for sampling bias in earlier studies. **Use:** citable caveat — earlier risk curves overestimated
  risk from severity-biased sampling, a methodological parallel worth a footnote given CPE's own
  ablation corpus-composition concerns (`pending_work.md`, stratified severity check).
- Kinetic-energy-as-injury-mechanism is also the stated basis of the **Vision Zero** road-safety
  framework (kinetic energy must stay below the threshold of human biomechanical tolerance) —
  referenced across the traffic-safety literature search results but without one single originating
  paper; cite via the Rosén & Sander papers above rather than Vision Zero policy documents directly,
  since those are policy framework citations rather than primary research.

- **Geisslinger, M., Poszler, F., Betz, J., Lütge, C. & Lienkamp, M. (2021). "Autonomous Driving
  Ethics: from Trolley Problem to Ethics of Risk."** *Philosophy & Technology*, 34(4), 1033–1055.
  DOI: 10.1007/s13347-021-00449-4. Defines risk as collision probability × estimated harm. **Use:**
  the v2 priority `P(arrive) · severity^γ · context` is this expected-harm form (methodology §4.20).
- **Geisslinger, M., Poszler, F. & Lienkamp, M. (2023). "An ethical trajectory planning algorithm for
  autonomous vehicles."** *Nature Machine Intelligence*, 5(2), 137–144. DOI:
  10.1038/s42256-022-00607-z. Code: `github.com/TUMFTM/EthicalTrajectoryPlanning`. Harm is a logistic
  injury curve of Δv, with Δv from conservation of momentum (`m_other / (m_self + m_other) · v_rel`).
  **Use:** the principled replacement for our `mass^λ` severity, which grows without bound where Δv
  saturates (§4.20).
- **Tefft, B. C. (2013). "Impact speed and a pedestrian's risk of severe injury or death."**
  *Accident Analysis & Prevention*, 50, 871–878. US crashes reweighted for sampling bias; risk is a
  function of impact speed. **Use:** with Rosén & Sander, the speed-not-mass basis for harm.
- **Westhofen, L. et al. (2023). "Criticality Metrics for Automated Driving: A Review and Suitability
  Analysis of the State of the Art."** *Archives of Computational Methods in Engineering*, 30(1),
  1–35. DOI: 10.1007/s11831-022-09788-7. TTC, Δv and energy-based severity are separate measures,
  each valid for its own scenario class. **Use:** why v2 keeps TTC (envelope) and harm (ranking)
  apart instead of fusing them into one K0-style formula.
- **Status (2026-10-20).** K0 is retired as a ranking (AUROC 0.537 on the corridor label). Its
  severity term survives with γ = 0.25, but a γ = 0 replay shows it doesn't change the alerts
  measurably on SANPO, so it is a declared value reported as a limitation (methodology §4.20).

## 5. Evaluation methodology: ranking and agreement without ground truth

Directly grounds the ablation and referee design (`methodology.md` §4).

- **Bradley, R. A. & Terry, M. E. (1952). "Rank Analysis of Incomplete Block Designs: I. The Method
  of Paired Comparisons."** *Biometrika*, 39(3-4), 324–345. DOI: 10.1093/biomet/39.3-4.324.
  The original forced-choice pairwise-comparison ranking model. **Use:** `methodology.md` §4.7
  explicitly invokes "the same machinery behind Bradley-Terry / Elo leaderboards" for the blinded
  referee's win-rate methodology — this is the primary citation for that claim, currently missing one.
- **Cohen, J. (1960). "A Coefficient of Agreement for Nominal Scales."** *Educational and
  Psychological Measurement*, 20(1), 37–46.
  Chance-corrected inter-rater agreement statistic. **Use:** the human-calibration step
  (`pending_work.md` P2.4) reports Cohen's κ between each VLM referee and the human gold set — cite
  the original paper, not just the statistic by name.
- **Zheng, L., et al. (2023). "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena."** NeurIPS
  2023 Datasets & Benchmarks. arXiv:2306.05685.
  Systematic study showing strong LLM judges can match human preference agreement (~80%), but also
  documents judge biases (position, verbosity, self-preference) that motivate calibration before
  trusting judge output. **Use:** directly grounds the "VLM as blinded referee, calibrated against a
  human gold set before any number is trusted" design (`methodology.md` §4.7) — also the
  source of the biases CPE should explicitly check for (e.g. does object presentation order bias a
  VLM referee's pick, given `collect_disagreements()` already randomizes object order for exactly
  this reason).

## 6. Cognitive Layer (SLM-1) and knowledge distillation

> **v1 only.** The SLM layers were dropped in the v2 design (2026-10-14, `architecture.md` §3).
> Kept for the paper's "why not a language model" discussion; section references below point to v1
> `architecture.md` (git `a6e37a7`).

- **Hinton, G., Vinyals, O., & Dean, J. (2015). "Distilling the Knowledge in a Neural Network."**
  arXiv:1503.02531.
  Introduces training a small "student" model to match a large "teacher" model's soft output
  distribution rather than only hard labels. **Use:** the foundational citation for
  `architecture.md` §9's teacher-student distillation pipeline (large offline VLM teacher → SLM-1
  student on edge-generated fact sheets) — currently described without this citation.
- **Yang, A., et al. — Qwen Team, Alibaba (2025). "Qwen3 Technical Report."** arXiv:2505.09388.
  Dense/MoE model family (0.6B–235B) with a unified thinking/non-thinking mode switch. **Use:**
  primary citation for the SLM-1 model choice (`architecture.md` §11.2) — the non-thinking mode is the
  specific feature CPE's design depends on for latency, so cite the mechanism, not just the model name.
- **Microsoft (2025). "Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language Models
  via Mixture-of-LoRAs."** arXiv:2503.01743.
  3.8B model matching larger models on reasoning via curated synthetic training data. **Use:** citation
  for the SLM-2 narration-layer candidate (`architecture.md` §11.3).

## 7. Physics-Verification-as-reward and RL training

> **v1 only.** The SLM layers were dropped in the v2 design (2026-10-14, `architecture.md` §3).
> Kept for the paper's "why not a language model" discussion; section references below point to v1
> `architecture.md` (git `a6e37a7`).

- **Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). "Proximal Policy
  Optimization Algorithms."** arXiv:1707.06347.
  The PPO algorithm CPE's `training/rl_agent/train_ppo.py` entry point is named for and intends to use
  (`architecture.md` §8) — cite directly; this is currently an uncited internal reference.
- **Bai, Y., et al. — Anthropic (2022). "Constitutional AI: Harmlessness from AI Feedback."**
  arXiv:2212.08073.
  Trains a preference/reward model from *AI*-generated judgments against a fixed rule set, then does
  RL against that learned reward — no human-labeled preference data required. **Use:** CPE's Physics
  Verification layer, which computes a deterministic reward from raw kinetics without human
  annotation (`architecture.md` §3, §8), is conceptually the same "non-human, rule-based reward
  signal replacing RLHF's human preference model" paradigm — RLAIF is the natural related-work anchor
  for that design choice, and worth an explicit paragraph in the paper distinguishing CPE's
  *deterministic-physics* reward from RLAIF's *learned-AI-judgment* reward (CPE's reward has no
  learned/black-box component at all, which is a stronger safety property worth stating explicitly).

## 8. Translation

- **Gala, J., et al. — AI4Bharat (2023). "IndicTrans2: Towards High-Quality and Accessible Machine
  Translation Models for all 22 Scheduled Indian Languages."** arXiv:2305.16307.
  Introduces the Bharat Parallel Corpus Collection (230M pairs) and the first NMT model covering all
  22 scheduled Indian languages. **Use:** primary citation for the translation layer
  (`architecture.md` §11.4) — currently referenced only by model name.

## 9. TTS / Audio

- **Ren, Y., et al. (2020). "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech."**
  arXiv:2006.04558.
  Non-autoregressive TTS trained directly against ground-truth targets with explicit pitch/energy/
  duration conditioning. **Use:** citation for the FastSpeech2 fallback option in
  `architecture.md` §11.5.
- **Kim, J., Kong, J., & Son, J. (2021). "Conditional Variational Autoencoder with Adversarial
  Learning for End-to-End Text-to-Speech" (VITS).** ICML 2021. arXiv:2106.06103.
  End-to-end variational TTS with a stochastic duration predictor. **Use:** Piper (the ONNX fast-local
  TTS candidate in `architecture.md` §11.5) is a VITS-architecture implementation with no separate
  paper of its own — cite VITS as the underlying method when citing Piper, rather than citing a
  software project as if it were a paper.
- **AI4Bharat / HuggingFace audio team (2025). IndicF5 and Indic Parler-TTS** — model releases rather
  than single papers; the closest associated technical writeups are the F5-TTS Indic-language
  adaptation study and the "Rasmalai" Indic speech-dataset paper (arXiv:2505.20693,
  arXiv:2505.18609). **Use:** cite these two as the technical basis when discussing the Indic-quality
  TTS candidates in `architecture.md` §11.5 — do not cite IndicF5/Indic Parler-TTS as if they have a
  dedicated peer-reviewed paper; they don't as of this writing, so describe them as model releases.

---

## How to keep this file current

- When a new model or method is adopted (or seriously evaluated) anywhere in the pipeline, add its
  citation here in the matching section — verified via a live search, never from memory.
- When a component in `architecture.md` or `methodology.md` names a model without a citation, that's
  a signal this file is missing an entry — check here first before adding an uncited name elsewhere.
- Section 1 (positioning) is the one to keep most current — new competing assistive-navigation
  systems are the citations most likely to matter to a reviewer, and the field moves fast (the
  Liu & Slade paper above is from 2026).


## 2026-10-25 additions (verified by live search on 2026-10-25)

- **Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L. & Schuster, T. (2024). "Conformal Risk
  Control."** ICLR 2024. arXiv:2208.02814. Extends split conformal prediction to bound the expected
  value of any monotone loss. **Use:** the `crc_budget_*` operating rules of the replay
  (methodology §4.23) bound a new walk's expected false-alert rate.
- **Chan et al. (2016). "Anticipating Accidents in Dashcam Videos."** ACCV 2016 (full author list
  not yet verified). DSA-RNN with an exponential loss on positive frames. **Use:** contrast for the
  earliness-weighted arrival objective (§4.23), which up-weights positives far from the event.
- **Suzuki et al. (2018). "Anticipating Traffic Accidents with Adaptive Loss and Large-scale
  Incident DB."** (full author list not yet verified) CVPR 2018. arXiv:1804.02675. LEA / AdaLEA, earliness
  losses adapted over training. **Use:** same contrast.
- **Kvamme, H. & Borgan, Ø. (2021). "Continuous and Discrete-Time Survival Prediction with Neural
  Networks."** *Lifetime Data Analysis*. arXiv:1910.06724. Discrete-time hazard parameterisations
  edge out PMF ones. **Use:** the arrival-time head (`--time-bins`) is a PMF parameterisation; a
  hazard head is a follow-up.
- **Presti, G., Ahmetovic, D., Ducci, M., Bernareggi, C., Ludovico, L., Baratè, A., Avanzini, F. &
  Mascetti, S. (2019). "WatchOut: Obstacle Sonification for People with Visual Impairment or
  Blindness."** ASSETS '19. Horizontal position mapped to stereo panning. **Use:**
  `src/narration/spatial.py`.
- **HEADS-UP: Head-Mounted Egocentric Dataset for Trajectory Prediction in Blind Assistance
  Systems** (2024). arXiv:2409.20324. The closest dataset to ours (head-mounted, collision risk for
  blind walkers). **Use:** related work; a cross-dataset check of the arrival predictor.
- **BlinkBud** (arXiv:2512.01366): Kalman 3-D tracking with head-motion correction on an earbud,
  for hazards from behind. **Use:** already cited in the paper; it shares our fusion design.
