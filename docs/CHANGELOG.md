# Changelog

All notable changes to the Composite Perception Engine (CPE) project will be documented in this file.

## [2026-07-22] - README Structure Maintenance Rule
### Documentation
- Added a `.agents/AGENTS.md` rule requiring the main `README.md` Project Structure section to be reviewed and updated whenever new files, folders, scripts, datasets, model registry entries, benchmark artifacts, or major docs are added.
- Updated the main `README.md` project map to include current docs, SANPO valid-stream metadata, YOLO training scripts, edge benchmark tools, model registry folders, and benchmark artifact locations.

## [2026-07-22] - SANPO 10-Session Edge Latency Benchmark
### Evaluation
- Ran v3 over 10 bounded SANPO-Real valid streams with the `jetson_orin_nano_8gb` simulation profile, processing 300 RGB+depth frames total.
- Saved individual metrics and aggregate reports under `evaluation/benchmarks/sanpo_edge_realtime/ten_session_v3_jetson_orin_nano_8gb/`.
- Average simulated p95 latency was `34.31 ms`; worst-session simulated p95 latency was `41.96 ms`, with all sessions passing the `<50 ms` detector/reflex budget.

## [2026-07-22] - Edge Device Simulation Profile
### Evaluation
- Added edge-profile support to `tools/benchmark_edge_realtime.py`, including a conservative `jetson_orin_nano_8gb` simulation profile with `4.0x` latency scaling and `3.0 ms` per-frame sensor/memory overhead.
- Ran the SANPO valid-stream smoke benchmark under the Jetson Orin Nano 8GB profile; simulated p95 total latency was `36.96 ms`, passing the `<50 ms` detector/reflex budget.

## [2026-07-22] - SANPO Valid-Stream Edge Smoke Test
### Evaluation
- Documented the public SANPO GCS bucket layout, valid-stream local convention, label metadata, and bounded download strategy in `docs/sanpo_bucket_structure.md`.
- Added `tools/download_sanpo_valid_streams.py` to download selected paired SANPO-Real RGB/depth frames from `valid_streams.json` without copying whole multi-GB sessions.
- Downloaded valid stream `0xCqEk5hjEvrygxu26MZkieSv45D_gaJ` and saved v3 full RGB+depth edge benchmarks under `evaluation/benchmarks/sanpo_edge_realtime/`.
- Fixed SANPO depth handling for native `960 x 960` `.float16.gz` maps by inferring shape and scaling RGB-space boxes into depth-map coordinates.

## [2026-07-22] - Edge Real-Time Benchmark Utility
### Evaluation
- Added `tools/benchmark_edge_realtime.py` to measure YOLO26n + ByteTrack streaming latency, optional depth loading, depth post-processing, effective FPS, and real-time/reflex budget pass-fail status.
- Saved the v3 detector-only GB10 benchmark under `evaluation/benchmarks/yolo26n_version_comparison/` with p95 total latency of `9.65 ms` per processed frame at `frame_step=3`.

## [2026-07-22] - Local Artifact Relocation
### Project Structure
- Moved the pretrained YOLO26n base checkpoint into `models/yolo/base_yolo26n/yolo26n.pt` and updated training, comparison, perception, notebook, and documentation references.
- Moved SANPO valid stream metadata into `simulation/datasets/sanpo/valid_streams.json` and updated gap-analysis consumers.

## [2026-07-22] - Documentation Cleanup and YOLO Guide Consolidation
### Documentation
- Replaced stale YOLO v1/v2 planning docs with `docs/yolo_training.md` as the current source of truth for dataset roots, checkpoints, training commands, evaluation commands, and version-comparison artifacts.
- Updated the docs index, architecture, artifact-structure guide, research-paper prompts, training README, and benchmark README to reflect the v3-from-base checkpoint and cumulative evaluation results.
- Removed obsolete links to documentation files that no longer exist.

## [2026-07-22] - YOLO26n Version Comparison Artifacts
### Evaluation
- Ran all-class held-out test evaluation for the v3-from-base checkpoint.
- Added cumulative v1/v2/v3 comparison artifacts under `evaluation/benchmarks/yolo26n_version_comparison/`, including overall metrics, per-class mAP tables, and retained-class base-YOLO deltas.

## [2026-07-22] - v3 From-Base Training Setup
### Training
- Removed interrupted YOLO training run folders while preserving successful v1/v2 checkpoints and evaluation artifacts.
- Documented the v3 retention-repair path that starts from base `models/yolo/base_yolo26n/yolo26n.pt`, reuses the cleaned v2_full dataset, and recommends extra Roboflow data for `truck`, `bicycle`, `dog`, `stop sign`, and `bus`.

## [2026-07-20] - AutoBatch Startup Fix
### Training
- Updated the GB10 training script to keep cuDNN benchmark disabled during `batch=-1` AutoBatch search so Ultralytics can probe large batch sizes instead of falling back to batch 16.

## [2026-07-20] - v2 Motorcycle Source Fill
### Training Data
- Added a dedicated `motorcycle` Roboflow detection source and moved the active clean rebuild target to `data/yolo_finetune_v2_full` after auditing that the retained COCO-style source had zero motorcycle positives.

## [2026-07-20] - v2 Missing-Class Source Fill
### Training Data
- Added verified Roboflow detection sources for `crosswalk` and `puddle` so the active 17-class detector is not trained with zero positive examples for those classes.
- Added `training/scripts/rebalance_yolo_splits.py` to move deterministic train examples into validation/test splits for train-only Roboflow sources before final all-class evaluation.

## [2026-07-20] - v2 Balanced Dataset Rebuild
### Training Data
- Fixed Roboflow merge accounting so class image caps are not double-counted while merging multi-source v2 data.
- Added explicit `pole` and `bollard` caps/minimums and pointed the active YOLO data config at `data/yolo_finetune_v2_balanced` for a clean v2 rebuild.

## [2026-07-20] - Roboflow Source Version Verification
### Training Data
- Verified the provided Roboflow Universe source URLs with the Roboflow SDK, updated the v2 manifest to the latest usable versions, and removed the invalid stairs source whose requested version was not found.
- Marked `tiger-dataset/dog-ukbxr` as unavailable for download because the project currently has no generated dataset versions; dog examples remain sourced through the COCO-style retained dataset until another dog dataset is provided.

## [2026-07-20] - v2 Roboflow Source Manifest Update
### Training Data
- Updated the active Roboflow Universe manifest with additional stairs, pothole, dog, bench, and retained COCO-class sources while preserving the existing pole and bollard datasets for retention.
- Tightened Roboflow label intake to accept only 5-column YOLO detection rows so segmentation polygon labels are not accidentally treated as bounding boxes.

## [2026-07-20] - Roboflow Class-Balanced Intake Caps
### Training Data
- Added per-class image caps to `training/scripts/download_roboflow_universe.py` so multiple Roboflow Universe URLs can be supplied per class without letting oversized source datasets dominate v2 fine-tuning.
- Documented the v2 class cap policy and SANPO domain-alignment guidance in `docs/plan_yolo_v2_finetune.md`, and added class limit/minimum examples to the Roboflow manifest template.

## [2026-07-20] - Retained COCO Class Expansion
### Training & Perception
- Added `dog` and `bench` back into the active CPE YOLO taxonomy as retained COCO navigation-obstacle classes, expanding the active detector head to 17 classes.
- Updated training, dataset-download, evaluation, retention-comparison, perception-filtering, physics severity, and planning docs to use retained class IDs 0-10 and custom hazard IDs 11-16.

## [2026-07-20] - CPE Taxonomy Simplification
### Training & Perception
- Reduced the active CPE YOLO hazard taxonomy from 16 to 15 classes by removing the broad head-height obstacle catch-all from training configs, scripts, perception filters, and docs.
- Kept `puddle` separate from `pothole` because it represents a different navigation risk, while noting that v2 data quality should decide whether it remains active long term.

## [2026-07-20] - GB10 YOLO26n Training Throughput Optimization
### Training
- Retuned `training/scripts/train_yolo26n_hazards.py` for the Grace-Blackwell GB10 lab server with RAM dataset caching, AutoBatch defaults, CPU-saturating worker selection, disabled training-loop validation/plots, skipped default export, and startup throughput reporting.
- Updated `docs/plan_yolo_v2_finetune.md` with the GB10 high-throughput profile, the 6.7GB dataset-cache estimate, and the fast v2 command/checklist.

## [2026-07-20] - Project Branding Cleanup
### Documentation & UI Text
- Removed visible legacy project-branding references from repository documentation and the Stage 1 visualization overlay, standardizing project-facing text on Composite Perception Engine (CPE).

## [2026-07-20] - Repository Artifact Cleanup & Model Registry
### Repository Hygiene
- Removed generated Ultralytics plot/preview artifacts from tracked training runs while preserving compact metrics and checkpoint metadata.
- Added `docs/artifact_structure.md`, `training/README.md`, `evaluation/benchmarks/README.md`, and `models/yolo/README.md` to clarify artifact locations, naming conventions, and cleanup rules.
- Added a local model registry entry under `models/yolo/cpe_yolo26n_hazards_v1/` and compact v1 benchmark summaries under `evaluation/benchmarks/yolo26n_hazards_v1/`.
- Updated `.gitignore` to keep datasets, secrets, binary weights, training plots, and local Roboflow manifests out of version control.

## [2026-07-20] - YOLO26n v2 Fine-Tuning Plan & Retention Comparison
### Training & Evaluation
- Added `docs/plan_yolo_v2_finetune.md`, a checklist-driven plan for continuing from the v1 `best.pt` checkpoint, repairing weak `stairs`/`pothole` performance, and adding retained-class validation data.
- Added `training/scripts/compare_yolo26n_retention.py` to compare a fine-tuned CPE checkpoint against base `models/yolo/base_yolo26n/yolo26n.pt` for retained COCO classes using class-name remapping, while validating the candidate on all active CPE classes.
- Updated `docs/architecture.md`, `docs/plan_yolo_finetune.md`, and `docs/research_paper_prompts.md` to document the v2 staged fine-tuning and retained-class validation workflow.

## [2026-07-20] - YOLO26n Hazard Checkpoint Evaluation
### Training & Evaluation
- Added `training/scripts/evaluate_yolo26n_hazards.py` to validate a fine-tuned checkpoint on `val` or `test`, export per-class metrics, generate a Markdown report, and save prediction previews for the new navigation-hazard classes.
- Documented the held-out test evaluation command in `docs/plan_yolo_finetune.md`.

## [2026-07-15] - Roboflow Universe Dataset Intake for YOLO26n
### Training Data
- Optimized `training/scripts/train_yolo26n_hazards.py` for GPU fine-tuning with CUDA runtime reporting, TF32/cuDNN benchmark flags, multi-GPU device parsing, integer batch handling, AMP controls, cache mode, and optional torch compile.
- Fixed the YOLO26n training preflight by using integer batch sizes and normalizing Roboflow labels to 5-column detection format before training.
- Added `.env` loading to `training/scripts/download_roboflow_universe.py` so SSH users can keep `ROBOFLOW_API_KEY` outside source code while running Roboflow Universe downloads.
- Added `training/scripts/download_roboflow_universe.py` to download selected Roboflow Universe datasets, remap source labels into the CPE hazard taxonomy, and merge YOLO images/labels into `data/yolo_finetune`.
- Added `training/configs/roboflow_universe_sources.example.json` as a manifest template for Universe workspace/project/version sources and class mappings.

### Perception Stack
- Aligned `src/perception_stack/yolo_tracker.py` and `src/perception_stack/physics.py` with the current hazard fine-tuning taxonomy so newly trained hazard classes are not filtered out at inference.

### Documentation
- Updated `docs/plan_yolo_finetune.md` with the SSH Roboflow workflow and minimum/preferred image-count targets for each new hazard class.

## [2026-07-14] - YOLO26n Hazard Fine-Tuning & Multi-Range Gap Analysis
### Tooling & Evaluation
- **Multi-Range Gap Analysis Script**: Added `tools/gap_analysis_experiments.py` to evaluate and compare 4 distinct depth ranges across 20 valid streams using all YOLO classes (whitelist disabled).
- **JSON Hard Example Logging**: Configured the script to save session ID and frame number mappings of hard examples in a range-specific JSON format (`hard_examples.json`) rather than writing large raw frame images to disk.
- **Experimental Reports**: Configured the script to output range-specific CSV summaries and a master comparison report (`distance_metric_comparison.md`) under the `gap_analysis_output/` folder.
- **Workspace Cleanup**: Cleaned up the workspace by deleting legacy visual output PNGs.

### Training
- Added `training/scripts/train_yolo26n_hazards.py`, a lab-GPU training script that fine-tunes `models/yolo/base_yolo26n/yolo26n.pt` on CPE navigation hazards while preserving the nano architecture, freezing early layers, applying rare-class weighting, and optionally exporting INT8 edge artifacts with an export-only mode for finished checkpoints.
- Added `training/configs/cpe_hazard_classes.yaml` with a CPE hazard taxonomy covering retained COCO safety classes plus non-COCO hazards such as `pole`, `bollard`, `stairs`, `crosswalk`, `pothole`, and `puddle`.

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
- **`docs/plan_yolo_finetune.md`**: Detailed implementation plan for fine-tuning YOLO26n. Includes class curation rationale (dropping non-navigation COCO classes while adding `pole`, `stairs`, `crosswalk`, and surface hazards), SANPO panoptic-to-YOLO label conversion pipeline, dataset split strategy, training script outline, and benchmark targets.
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
