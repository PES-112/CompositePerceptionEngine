# Changelog

All notable changes to the Composite Perception Engine (CPE) project will be documented in this file.

## [2026-07-14] - YOLO26n Hazard Fine-Tuning & Multi-Range Gap Analysis
### Tooling & Evaluation
- **Multi-Range Gap Analysis Script**: Added `tools/gap_analysis_experiments.py` to evaluate and compare 4 distinct depth ranges across 20 valid streams using all YOLO classes (whitelist disabled).
- **JSON Hard Example Logging**: Configured the script to save session ID and frame number mappings of hard examples in a range-specific JSON format (`hard_examples.json`) rather than writing large raw frame images to disk.
- **Experimental Reports**: Configured the script to output range-specific CSV summaries and a master comparison report (`distance_metric_comparison.md`) under the `gap_analysis_output/` folder.
- **Workspace Cleanup**: Cleaned up the workspace by deleting legacy visual output PNGs.

### Training
- Added `training/scripts/train_yolo26n_hazards.py`, a lab-GPU training script that fine-tunes `yolo26n.pt` on CPE navigation hazards while preserving the nano architecture, freezing early layers, applying rare-class weighting, and optionally exporting INT8 edge artifacts with an export-only mode for finished checkpoints.
- Added `training/configs/cpe_hazard_classes.yaml` with a 16-class CPE hazard taxonomy covering retained COCO safety classes plus non-COCO hazards such as `pole`, `bollard`, `stairs`, `crosswalk`, `pothole`, `puddle`, and `overhanging_hazard`.

### Documentation
- Updated `docs/architecture.md` and `docs/research_paper_prompts.md` to document the multi-range gap analysis capabilities, whitelist removal, and JSON logging format.
- Updated the architecture and research-paper prompts to describe the edge-preserving YOLO26n fine-tuning workflow.

## [2026-07-12] — YOLO Gap Analysis Expansion & Automation
### Tooling & Inference
- **Expanded YOLO Gap Analysis Notebook**: Overhauled `notebooks/sanpo_yolo_gap_analysis.ipynb` with centralized configuration, robust morphological object extraction, 5-panel visualizations, and automated fine-tuning advice reports.
- **GCS-Aware Stream Fallback**: Integrated streaming logic in `SANPOLoader` to dynamically fetch images and depth maps from Google Cloud Storage anonymously when local files are missing.
- **Automated Verification**: Created verification test scripts to execute and validate the notebook code cells under headless conditions.

### Architecture & Documentation
- **Architecture Doc Update**: Documented the new error analysis and dataset gap analysis pipeline in `docs/architecture.md`.
- **Research Prompt Curation**: Added prompt guidelines for writing methodology sections on connected component extraction and hard example mining in `docs/research_paper_prompts.md`.

## [2026-07-10] — Implementation Plans for YOLO Fine-Tuning & Gap Analysis
### Documentation
- **`docs/plan_yolo_finetune.md`**: Detailed implementation plan for fine-tuning YOLO26n. Includes class curation rationale (dropping `cat`, `umbrella`, `backpack`, `suitcase`; adding `pole`, `stairs`, `crosswalk`, `overhanging_hazard`), SANPO panoptic-to-YOLO label conversion pipeline, dataset split strategy, training script outline, and benchmark targets.
- **`docs/plan_gap_analysis.md`**: Step-by-step implementation plan for running the gap analysis notebook across SANPO. Documents all config parameters, output folder management (frame sampling to prevent bloat), visualization legend, CSV column definitions, and post-run analysis queries.

## [2026-07-10] — SANPO Analysis & YOLO Class Curation
### Research & Tooling
- **SANPO dataset clarified**: SANPO is a Google Research egocentric navigation dataset (701 real + 1961 synthetic sessions), NOT an Indian dataset. Confirmed 30-class taxonomy covering sidewalks, poles, crosswalks, pedestrians, and cyclists.
- **YOLO class curation**: Removed region-specific classes (autorickshaw); class list finalized to 14 navigation-critical COCO classes plus depth-detected unlabeled blobs.
- **Gap Analysis Notebook**: Created `notebooks/sanpo_yolo_gap_analysis.ipynb` to empirically identify which scene objects are detected by depth but missed by the current YOLO26n COCO whitelist. Outputs per-frame visualizations (3-panel: RGB+YOLO, depth heatmap, gap-only), a frequency table, and a CSV of gap statistics.

## [2026-07-10]
### Architecture & Documentation
- **Knowledge Distillation Strategy**: Added formal documentation to `architecture.md` outlining the Teacher-Student SLM training pipeline for edge constraints.
- **YOLO Domain Fine-tuning**: Added explicit strategy for fine-tuning YOLO26n on street-level hazards rather than generic COCO classes.
- **Documentation Refactoring**: Consolidated `perception_stack.md`, `physics_verification.md`, `phase_1_breakdown.md`, and `scaffolded_components.md` into the master `architecture.md` for a single source of truth.
- **Research Prompts**: Created `research_paper_prompts.md` to aid in generating the academic paper via LLMs.

## [2026-03-30]
### Perception Stack
- **Centre-patch depth sampling**: Added 20% patch to fix bbox edge bleed (`src/perception_stack/depth_loader.py`)
- **Occlusion exclusion masking**: Added `exclude_boxes` param to `median_depth_in_box` (`src/perception_stack/depth_loader.py`)
- **Sparse depth 3x fallback**: Auto-expands patch if no valid pixels found (`src/perception_stack/depth_loader.py`)
- **Class whitelist filter**: `ALLOWED_CLASSES` set to drop non-navigation COCO classes (`src/perception_stack/yolo_tracker.py`)
- **Geometric validation**: Added `is_valid_detection()` for aspect ratio, area, and edge FP checks (`src/perception_stack/yolo_tracker.py`)
- **3-pass depth rescoring**: Added `depth_rescore()` function for size sanity, conf, and depth NMS (`src/perception_stack/yolo_tracker.py`)
- **Grid-based unlabeled obstacle sweep**: Replaces single-ROI scan with 5 columns (`src/perception_stack/pipeline.py`)
- **Front-to-back occlusion ordering**: Added sorting and cumulative occluder list (`src/perception_stack/pipeline.py`)
- **Nth-frame skip**: Added `frame_step` param (e.g., `frame_step=3` for ~67% compute reduction) (`src/perception_stack/pipeline.py`, `tools/run_perception.py`)
- **Streaming generator architecture**: `run_perception_stream` generator yields per-frame, breaking RAM bottleneck (`src/perception_stack/pipeline.py`)
- **tqdm progress bar**: Replaces print every 50 frames (`src/perception_stack/pipeline.py`)
- **StreamingCSVWriter**: Context manager for incremental disk writes paired with generator (`src/perception_stack/csv_writer.py`)
- **Batch tensor physics**: Vectorized functions `batch_compute_bearing` & `batch_kinetic_score` (`src/perception_stack/physics.py`)
- **CLASS_SEVERITY deduplication**: `fact_sheet_builder` now imports from `physics.py` (`src/perception_stack/fact_sheet_builder.py`)

### Tools
- **CLI updates**: Added `--frame_step` arg and streaming API to `run_perception.py`, using `StreamingCSVWriter` (`tools/run_perception.py`)

## [2026-03-27]
### Perception Stack
- **Unlabeled obstacle detection**: Single central-ROI scan for poles/bollards via depth scan (`src/perception_stack/pipeline.py`)

## [2026-03-25]
### Architecture
- **Dual-SLM system design**: Initial CPE architecture finalized with Reflex, Cognitive, and Physics Verification layers (`architecture.md`)

### Perception Stack
- **YOLO26n + ByteTrack integration**: Basic tracking pipeline (`src/perception_stack/yolo_tracker.py`)
- **SANPO + UASOL depth map loading**: Auto-detects `.float16.gz` vs `.png` (`src/perception_stack/depth_loader.py`)
- **Stage 1 pipeline**: RGB+Depth → CSV batch mode with `run_perception()` (`src/perception_stack/pipeline.py`)
- **CSV canonical schema**: `CSV_FIELDS` contract defined and writer implemented (`src/perception_stack/csv_writer.py`)
- **Physics calculations**: Scalar functions for bearing, velocity, and kinetic score (`src/perception_stack/physics.py`)
- **Stage 2 fact sheet builder**: CSV → JSONL with K0 and K+2 lookahead (`src/perception_stack/fact_sheet_builder.py`)

### Physics Verification
- **Adjudication logic**: Added 4-rule judge and RL rewards via `PhysicsVerification` class (`src/physics_verification/physics_verification.py`)

### Tools
- **run_perception.py**: Initial batch API CLI for Stage 1 (`tools/run_perception.py`)
- **run_fact_sheets.py**: Initial implementation for Stage 2 (`tools/run_fact_sheets.py`)
