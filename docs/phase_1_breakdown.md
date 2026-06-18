# CPE Project Implementation Plan

## 1. The 4-Phase Rollout Strategy

A 25% milestone should represent **a complete vertical slice of a subsystem**, not just disparate disconnected parts. 

**Is training SLM-1 a good enough milestone for 25%?**
Yes, but with a caveat: you cannot *train* SLM-1 (via PPO) until the Physics Verification judge exists to reward it. Therefore, Phase 1 (25%) should be: **Data Pipeline + Heuristic Baseline + Supervised SLM-1 Warmup.** 

Here is the 4-phase split:

| Phase | Milestone (25% chunks) | Description |
|---|---|---|
| **Phase 1 (25%)** | **Data & Cognitive Baseline** | Curate SANPO/UASOL/HEADSUP datasets. Build the JSON Fact Sheet generator. Train SLM-1 on a *supervised heuristic baseline* (teaching it basic JSON parsing and threat formatting, no RL yet). |
| **Phase 2 (50%)** | **Physics Engine & Reflexes** | Implement the `Threat Prioritizer` and `Reflex Layer` (TTC math). Build the `Physics Verification` node to act as the PPO Teacher. Connect YOLO to the actual video feeds. |
| **Phase 3 (75%)** | **PPO Reinforcement & Narrator**| Run the PPO loop where the Phase 2 Physics Judge actually trains the Phase 1 SLM-1 on "Future Kinetic Grounding". Integrate SLM-2 (Phi-3-Mini) for basic speech text generation. |
| **Phase 4 (100%)**| **Edge Audio & Polish** | Integrate FastSpeech2 (Audio), Indic Translation, and the System Heartbeat. Export SLMs to GGUF format and test latency metrics. |

---

### 2. Phase 1: Granular Step-by-Step Guide

Phase 1 is entirely about **Data Preparation and Supervised Fine-Tuning (SFT)**. You are building the automated pipeline that turns raw videos into structured JSON files, and then teaching SLM-1 to read those JSON files.

### Step 1: Downloading & Prepping the Datasets
You do not need to download the entire datasets locally. 
- **Locally (Your Laptop):** Download only a tiny subset (~50 short video clips) for writing and debugging the Python scripts.
- **Cloud (Google Colab / Kaggle):** Download the full datasets directly into the cloud volume when you are ready to generate the final JSON files.

**The Split:**
- **SANPO (Main Dataset):** Used for standard depth. Use 70% of this for training.
- **UASOL (Chaos Dataset):** Used for shaky-cam stress-testing. Use 15% for training.
- **HEADSUP (Intent Dataset):** Used to teach SLM-1 pedestrian behavior. Use 15% for training.

---

### Step 2: YOLO26 Offline Processing
We do **not** run YOLO dynamically during SLM training.
1. Write a Python script using `ultralytics` YOLO to process your video frames.
2. Run ByteTrack (or BoT-SORT) to assign a persistent `track_id` to each object.
3. Save the outputs to a CSV file per video:
   `frame_id | track_id | class | bbox_x1 | bbox_y1 | bbox_x2 | bbox_y2`

---

### Step 3: SANPO Depth & Velocity Extraction (The Physics)
Now we add the physics variables ($d$ and $v$) to our CSV.
1. **Depth ($d$):** For each YOLO bounding box in a frame, open the corresponding SANPO **ground-truth depth map**. Extract the median pixel value within that bounding box. Add this as `distance_m`.
2. **Velocity ($v$):** Because you have ByteTrack IDs, you know `track_05` is the same car across frames.
   - Look at `track_05` at Frame 10 ($d_1 = 20m$) and Frame 15 ($d_2 = 18m$).
   - Since the video is 30fps, 5 frames = $0.166$ seconds ($\Delta t$).
   - Velocity = $\frac{d_1 - d_2}{\Delta t} = \frac{20 - 18}{0.166} = 12$ m/s.
   - *If the resulting $v$ is jittery, apply a simple 1D Kalman filter or moving average to smooth it out.*
3. Save $d$ and $v$ to the CSV.

---

### Step 4: HEADSUP Intent Alignment
HEADSUP gives you labels like "Pedestrian looking at phone".
1. Write a script to spatially align the HEADSUP bounding boxes with your YOLO track IDs.
2. If `track_12` matches a HEADSUP box labeled "distracted", add `intent = "distracted"` to your CSV. Otherwise, `intent = "none"`.

---

### Step 5: JSON Target Generation (The Teacher)
Now we convert our master CSV into the `Fact Sheet` prompt that SLM-1 will read.
1. Iterate through your CSV frame-by-frame.
2. **Calculate Present Kinetic Score ($K_0$):** Run your deterministic math formula for every object in the frame to find the most dangerous object *right now*. Save this as `heuristic_target_id`.
3. **Calculate Future Kinetic Score ($K_{+2s}$):** Look 60 frames ahead in the CSV. Which object actually became dangerous 2 seconds later? Save this as `future_target_id`.
4. Generate the final JSONL training file.

**Example Row in `train.jsonl`:**
```json
{
  "system": "You are a navigation AI. Analyze this Fact Sheet...",
  "user": "[SCENARIO FACT SHEET] Object_01: Car, 10m, v=5m/s... Object_02: Pedestrian, 4m, intent=distracted...",
  "assistant": "{\"primary_threat\": \"Object_02\", \"reason\": \"Pedestrian is distracted near path.\"}"
}
```
*Notice how the "assistant" answer uses the `future_target_id` — this is what SLM-1 will learn!*

---

### Step 6: SLM-1 Fine-Tuning (Colab)
Yes, you should use **Google Colab Pro (A100 GPU)** or a cloud provider for this. Your local laptop is for the CSV generated scripts, not LLM fine-tuning.

1. **Framework:** Use HuggingFace `SFTTrainer` (Supervised Fine-Tuning) and `peft` (LoRA).
2. **Model:** Load `Qwen2.5-1.5B-Instruct` in 4-bit quantization (bitsandbytes).
3. **LoRA Adapter:** Train an adapter (Rank=16) on your `train.jsonl` file.
4. **Goal of Phase 1:** We are NOT doing PPO yet. We are simply teaching Qwen to map the `user` string (Fact Sheet) to the exact `assistant` string (JSON output). 
5. **Output:** Once loss converges, export the LoRA adapter. You now have a Baseline SLM-1 that knows how to read your specialized data format.
