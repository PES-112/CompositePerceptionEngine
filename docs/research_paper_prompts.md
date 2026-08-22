# Research Paper Drafting Prompts

Use the following prompts with an LLM (like Gemini) to systematically draft the sections of your research paper on the Composite Perception Engine (CPE).

---

## 1. Abstract & Introduction
**Prompt:**
> "I am writing a research paper on an edge-optimized AI system designed to provide real-time, semantically rich hazard alerts for the visually impaired. The system, called the Composite Perception Engine (CPE), uses a hybrid architecture. It combines a lightweight object detector (YOLO26n fine-tuned on street hazards), egocentric depth estimation, and a Knowledge-Distilled Small Language Model (SLM) for semantic scene narration. It uses a deterministic 'Physics Verification' layer to arbitrate between neural network outputs and raw kinetic physics (Time-to-Collision) to prevent hallucinations and guarantee <50ms response times for critical threats. 
> 
> Please write a compelling Academic Abstract (250 words max) and a comprehensive Introduction. The Introduction should highlight the limitations of current assistive technologies (either too computationally heavy for edge devices, or lacking semantic context) and clearly state our contributions: (1) N-frame skip with depth interpolation for 67% compute reduction, (2) Edge-YOLO fine-tuned on domain-specific hazards, and (3) SLM Knowledge Distillation from a massive offline teacher model."

## 2. Related Work
**Prompt:**
> "Based on our edge-AI assistive technology system (CPE) for the visually impaired, write a 'Related Work' section. Compare our approach to three main existing paradigms:
> 1. Traditional sensory substitution devices (e.g., ultrasonic canes) which lack semantic understanding.
> 2. Cloud-based LLM/VLM assistive tech (e.g., Be My Eyes, GPT-4V) which suffer from high latency and require constant network connectivity, making them unsafe for real-time hazard avoidance.
> 3. Embedded CNN-based obstacle detectors (standard YOLO on mobile) which identify objects but fail to prioritize physical threats (like a fast-moving bicycle vs a parked car) or provide natural language context.
> Emphasize how our hybrid approach (deterministic physics + knowledge-distilled SLM) bridges the gap between semantic richness and edge-critical latency."

### 3. Methodology: Perception Stack & N-Frame Skip
**Prompt:**
> "Write the Methodology subsection detailing our 'Hybrid Perception Stack'. Explain how we process RGB and depth data on the edge. Include the following technical details:
> - We fine-tune YOLO26n specifically for street hazards (potholes, bollards, stairs, crosswalks, poles, puddles, dogs, and benches) rather than using standard COCO classes, as COCO is insufficient for navigation.
> - The training workflow keeps the nano architecture fixed, freezes early backbone layers during transfer learning, replaces the broad COCO head with a compact 17-class CPE hazard taxonomy, and exports ONNX/TensorRT FP16 or INT8 edge artifacts with modern quantization flags to avoid increasing runtime latency.
> - We evaluated multiple YOLO26n fine-tuning variants and selected a v3-from-base protocol: start from pretrained `models/yolo/base_yolo26n/yolo26n.pt`, train the compact 17-class CPE head with GB10 high-throughput settings (RAM dataset cache, AutoBatch, CPU-saturated data loading, and deferred validation), then validate all classes and compare retained COCO-class precision/recall against base YOLO26n with explicit class-name remapping. This repaired the v2 retained-class regression while preserving custom hazard improvements.
> - We implement an N-frame detection skip (e.g., every 3rd frame). Between detection frames, we rely on depth-guided tracking and interpolation, reducing compute load by ~67% while maintaining high safety margins.
> - We use the depth map to extract precise distance (d) at the bounding box centroid. Object bearing is computed directly from the YOLO bounding box centroid pixel coordinate: `bearing_deg = ((cx_px − frame_width/2) / (frame_width/2)) × (hfov_deg/2)`, requiring no additional hardware beyond the phone camera.
> - We compute a 'Kinetic Score' K = f(class_severity, velocity, distance) and evaluate six candidate formulas (K0: sev×v²/max(d,ε); K1: TTC-primary; K2: linear velocity; K3: quadratic distance decay; K4: hybrid momentum+TTC; K5: sigmoid-normalised) using a benchmark harness that measures Pearson/Spearman TTC correlation, routing lane precision/recall/F1 against K₊₂ ground truth, monotonicity, and range stability. The winning formula is selected prior to SLM-1 training.
> - We supplement egocentric SANPO pseudo-labels with carefully remapped Roboflow Universe datasets for rare hazard classes, enforcing a fixed 17-class taxonomy and session/source-level validation splits to prevent leakage.
> - We implement an empirical 'YOLO Gap Analysis' pipeline that cross-references depth map structures with YOLO detections using morphological component labeling. The pipeline runs without class whitelist constraints to evaluate general perception failures, compares multi-range depth configurations (from immediate hazards to far-range path planning), compiles comparison metrics, and logs hard examples in structured JSON files for downstream annotation workflows."

## 4. Methodology: Knowledge Distillation & SLM
**Prompt:**
> "Write the Methodology subsection detailing our 'Cognitive Layer and SLM Knowledge Distillation'. Explain the following pipeline:
> - Because running large Vision-Language Models on edge devices is impossible, we use a Teacher-Student distillation approach.
> - Offline, we use a massive Teacher model (e.g., Grounding DINO / GPT-4V) on street-view datasets to generate high-quality pseudo-labels and rich semantic descriptions of hazards (e.g., 'pothole at foot level').
> - We use this generated dataset to fine-tune a Small Language Model (SLM), with Qwen3-1.7B non-thinking mode as the primary cognitive candidate and Qwen2.5-1.5B as fallback.
> - At runtime, the SLM receives a structured 'Symbolic Fact Sheet' — a JSON object containing only fields derivable from a phone camera: track ID, object class, bearing (computed from pixel centroid), distance, closing velocity, TTC, kinetic score, and the routing decision. Intent labels (from HEADSUP dataset) appear only in the SFT training data to help the SLM learn to infer intent from scene context, not as runtime inputs.
> - The SLM-1 response is also structured JSON: primary_threat_id, reason (one sentence), scene_state, confidence, and future_confirmed (K₊₂ ground-truth validation flag). This structured output enables deterministic reward computation in the Physics Verification RL loop."

## 5. Methodology: Physics Verification (The Judge)
**Prompt:**
> "Write the Methodology subsection on the 'Physics Verification Layer'. This is the safety-critical core of the architecture. Explain that:
> - The system converts each YOLO+tracking+depth row into a formal `ThreatEvent` with distance, closing velocity, bearing, Kinetic Score, TTC, route (`ignore`, `cognitive`, or `reflex`), priority, and reason.
> - The SLM runs in soft real-time (~500ms) only for cognitive-route events, but language models can hallucinate.
> - The implemented Reflex Layer consumes reflex-route `ThreatEvent`s, converts them into deterministic `ReflexResult`s, bypasses all SLM/TTS models, and hands immediate override or high-K physics fallback candidates to Physics Verification in hard real-time (<50ms).
> - The Physics Verification acts as an adjudicator: If TTC < 1.0s, it completely bypasses the SLM and triggers an immediate 'Reflex' alarm. If the SLM hallucinates a safe scene but the Kinetic Score is high, the Physics Verification overrides the SLM. This guarantees that critical physical threats are never missed due to neural network latency or hallucination."

## 5b. Methodology: Kinetic Score Formulation and Its Evaluation
**Prompt:**
> "Write the Methodology subsection defending the Kinetic Score `K = severity(c) * v^gamma / max(d, eps)`. Explain that:
> - `gamma = 2` is a deliberate bet that *consequence* matters independently of *arrival time*: the reflex layer's TTC gate already handles arrival time, so a K that merely reproduced TTC ranking would be redundant. State this as a design claim, not a measured result.
> - Class severity is **derived from real-world mass**, not hand-tuned: `severity(c) = behaviour(c) * (mass(c)/70 kg)^lambda`, lambda = 0.5. Justify lambda as *partial* compression, not flattening: the class mass range spans 171x, and lambda=1 (literal kinetic energy) lets a distant bus outrank an imminent pedestrian, while lambda=0.5 - severity proportional to sqrt(mass) - retains a genuine destructive-mass bias (car 4.6x a person, bus 13x, bus 2.8x a car) at the geometric midpoint between discarding mass and passing it through. Report lambda = 0.25 / 0.5 / 1.0 as ablation arms and state that 0.5 is a prior, not a measured optimum. Note that fitting the project's earlier hand-tuned table against log mass recovers lambda ~ 0.18 (R^2 = 0.50), which is evidence the hand table under-weighted mass rather than evidence for 0.18. Note also that with v^2 in the score, kinematics dominate severity in most reorderings, so lambda mainly breaks ties between objects of similar motion - do not oversell it. Report the sparse behaviour multipliers (erratic motion, trip height, head-height mounting) and the four massless trip hazards held outside the law as an explicit, acknowledged exception.
> - Apparent bounding-box size (`A_px/d`) is carried as an optional term with exponent mu, **disabled in the shipped configuration**, because box area is ~99% predictable from class and metric depth by projective geometry. Present its inclusion as an ablation arm and report the measured result either way.
> - Evaluation avoids circularity: no formula is graded against another formula's output. Report (a) the ablation of K0's own terms - gamma=1, severity removed, velocity removed, apparent size added, and plain time-to-hazard as an external reference; (b) six label-free metrics (flicker rate, rank stability under 2/5/10% depth perturbation, temporal smoothness, tie rate, complementarity with SLM-1, future self-consistency); (c) automatic encounter labels derived only from *measured* future distance and bearing, used to eliminate variants rather than to select one.
> - All confidence intervals come from a **session-level** bootstrap, because frames within a walking session are autocorrelated and frame-level intervals would be dishonestly narrow. State the sample: a seeded 30% sample of the SANPO-Real valid streams.
> - The one irreducibly human question - whether gamma and the severity weights are *right* - is settled by a **blinded forced-choice referee** restricted to the frames where two variants pick different top objects (~5% of frames, which is what makes labelling affordable). Three VLM referees from three model families, run locally, estimate labelling noise, not truth; report pairwise Cohen's kappa between them and, critically, kappa against a human-labelled calibration subset. State plainly that if human kappa is poor, the VLM numbers are not evidence."

## 6. Results, Discussion & Conclusion
**Prompt:**
> "Using only the supplied measured results, write the Results, Discussion, and Conclusion sections. Distinguish native NVIDIA GB10 measurements from the analytical Jetson Orin Nano 8GB proxy (4.0x compute-latency scaling plus 3.0 ms overhead). Report the 10-session SANPO result as simulated evidence: average p95 34.31 ms, worst-session p95 41.96 ms, with all simulated sessions below the 50 ms reflex budget. Do not claim physical edge-device FPS, high-kinetic recall, stale-SLM filtering, or power efficiency until those experiments have actually been run. Discuss the expected trade-offs between semantic richness, compute latency, and future battery-powered deployment."

