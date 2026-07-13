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

## 3. Methodology: Perception Stack & N-Frame Skip
**Prompt:**
> "Write the Methodology subsection detailing our 'Hybrid Perception Stack'. Explain how we process RGB and depth data on the edge. Include the following technical details:
> - We fine-tune YOLO26n specifically for street hazards (potholes, bollards, branches) rather than using standard COCO classes, as COCO is insufficient for navigation.
> - The training workflow keeps the nano architecture fixed, freezes early backbone layers during transfer learning, replaces the broad COCO head with a compact 16-class CPE hazard taxonomy, and exports INT8 edge artifacts to avoid increasing runtime latency.
> - We implement an N-frame detection skip (e.g., every 3rd frame). Between detection frames, we rely on depth-guided tracking and interpolation, reducing compute load by ~67% while maintaining high safety margins.
> - We use the depth map to extract precise distance (d) at the bounding box centroid to calculate a 'Kinetic Score' based on velocity and distance.
> - We implement an empirical 'YOLO Gap Analysis' pipeline that cross-references depth map structures with YOLO detections using morphological connected component labeling. This system mines hard examples automatically (saved under `hard_examples/`) and constructs fine-tuning action plans targeting ground clutter or head-height hazards."

## 4. Methodology: Knowledge Distillation & SLM
**Prompt:**
> "Write the Methodology subsection detailing our 'Cognitive Layer and SLM Knowledge Distillation'. Explain the following pipeline:
> - Because running large Vision-Language Models on edge devices is impossible, we use a Teacher-Student distillation approach.
> - Offline, we use a massive Teacher model (e.g., Grounding DINO / GPT-4V) on street-view datasets to generate high-quality pseudo-labels and rich semantic descriptions of hazards (e.g., 'pothole at foot level').
> - We use this generated dataset to fine-tune a Small Language Model (SLM) like Qwen or Phi-3.
> - In real-time inference, the SLM takes a lightweight 'Fact Sheet' (containing the YOLO class, depth, and kinetic score) and outputs a context-aware natural language alert, mimicking the reasoning of the heavy Teacher model."

## 5. Methodology: Physics Verification (The Judge)
**Prompt:**
> "Write the Methodology subsection on the 'Physics Verification Layer'. This is the safety-critical core of the architecture. Explain that:
> - The SLM runs in soft real-time (~500ms) to provide semantic narration, but language models can hallucinate.
> - The Reflex Layer computes raw Time-To-Collision (TTC) deterministically in hard real-time (<50ms).
> - The Physics Verification acts as an adjudicator: If TTC < 1.0s, it completely bypasses the SLM and triggers an immediate 'Reflex' alarm. If the SLM hallucinates a safe scene but the Kinetic Score is high, the Physics Verification overrides the SLM. This guarantees that critical physical threats are never missed due to neural network latency or hallucination."

## 6. Results, Discussion & Conclusion
**Prompt:**
> "Assuming we ran benchmarks showing our system achieves 30 FPS on an edge device (e.g., Jetson Orin Nano) with an N-frame skip of 3, maintains a 95% recall rate on high-kinetic threats, and successfully filters out 99% of SLM hallucinations via the Physics Verification layer, write the Results, Discussion, and Conclusion sections. Discuss the trade-offs between semantic richness and compute latency, and how our Teacher-Student distillation method makes rich narration viable for battery-powered edge devices."
