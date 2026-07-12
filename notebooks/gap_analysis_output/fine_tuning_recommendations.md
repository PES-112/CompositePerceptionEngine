# Fine-Tuning & Dataset Gap Analysis Report
**Generated on:** 2026-07-12 18:28:43

## 1. Detection Performance
* **Most Common Detections:** `truck` (1 hits)
* **Strongest Class (Highest Mean Confidence):** `truck` (0.55)
* **Weakest Class (Lowest Mean Confidence):** `car` (0.31)

## 2. Depth Gap Analysis
* **Total Uncovered Gaps:** 2
* **Most Common Elevation of Missed Objects:** `foot_level`
* **Missed Gaps by Proximity:**
  * **Immediate hazard zone (<= 2m):** 0 instances (Critical risk)
  * **Mid-range navigation (2m - 4m):** 0 instances
  * **Far-range space (> 4m):** 2 instances

## 3. Dataset & Fine-Tuning Recommendations
1. **Prioritize Ground-Level Hazard Dataset Curation:** A large proportion of gaps are detected at ground/foot level. We need more annotations for steps, potholes, curb changes, and ground trash. Use data augmentation that focuses on low angles/ground planes.
2. **Collect Head-Height Obstacles:** Found substantial head-level depth gaps (e.g. low pipes, tree branches, hanging signs). Introduce a `hanging_hazard` class during the next fine-tuning cycle to capture these high-risk areas.
4. **Target Low-Confidence Classes:** Boost whitelisted classes with low confidence (<50%) by adding more target images: `car`.
5. **Environment Augmentations:** Introduce shadow training patterns (to prevent false depth regions due to contrast transitions) and motion-blur overlays mimicking typical visually impaired head scan behaviors.
6. **Annotation Curation:** Review exported frames in `hard_examples/` to verify boundaries, annotate previously missed items, and update annotations for classes like `person` or `dog` in cluttered contexts.

## 4. Prioritized Action Plan
| Priority | Action Item | Target / Details |
|---|---|---|
| 🔴 **CRITICAL** | Mine Hard Examples | Verify and annotate frames saved in `hard_examples/`. |
| 🟠 **HIGH** | Boost Confidence | Target dataset collection (500+ labels) for weak class `car`. |
| 🟡 **MEDIUM** | Shadow & Blur Augmentation | Implement exposure/shadow contrast variations in PyTorch training pipeline. |