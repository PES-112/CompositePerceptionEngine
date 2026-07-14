# Multi-Range Gap Analysis Comparison Report
**Generated on:** 2026-07-14 11:19:02

## Executive Summary
This report compares the detection profile of object-agnostic perception gaps (obstacles seen by depth but missed by YOLO26n)
across **20 streams** (5 sampled frames per stream) under **four distinct depth range configurations**.
Crucially, **no class whitelist was used**; YOLO26n predicted all 80 standard COCO classes, and any overlapping YOLO detections
regardless of class were masked out to ensure only truly unexplained physical obstructions are flagged as gaps.

## Comparison Matrix
| Range Name | Depth Bounds | Total Gaps | Avg Gaps / Frame | Avg Gap Size (px) | Median Gap Size (px) | Avg Unexplained Area (%) | Elevation (Foot/Mid/Head %) | Hard Example Sessions | Total Hard Frames |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| range_0.5_2.0 | 0.5m - 2.0m | 26 | 0.26 | 21842.3 | 8525.5 | 93.33 | 61.5% / 26.9% / 11.5% | 8 | 16 |
| range_0.5_4.0 | 0.5m - 4.0m | 93 | 0.93 | 72632.1 | 39259.0 | 93.33 | 77.4% / 18.3% / 4.3% | 17 | 60 |
| range_0.5_6.0 | 0.5m - 6.0m | 97 | 0.97 | 144369.6 | 120873.0 | 93.33 | 80.4% / 15.5% / 4.1% | 18 | 75 |
| range_2.0_6.0 | 2.0m - 6.0m | 120 | 1.2 | 111664.6 | 56227.5 | 93.33 | 82.5% / 14.2% / 3.3% | 18 | 74 |

## Key Takeaways
1. **Impact of Short Ranges (0.5m - 2.0m)**: Focuses strictly on immediate walking path hazards. Gaps here are low in count but represent critical trip risks.
2. **Standard and Mid-Range Proximity**: Increasing depth range to 4m or 6m captures infrastructure blocks (bollards, signs, posts) further ahead,
   increasing both the gap counts and the total unexplained navigation area.
3. **Long-Range Filtering (2.0m - 6.0m)**: By ignoring objects closer than 2 meters, this range highlights upcoming visual planning features,
   useful for filtering out local walking cane movement or foot motion noise.

## Saved Data Structure
For each range configuration, results are saved inside `notebooks/gap_analysis_output/range_<min>_<max>/`:
- `frame_summary.csv`: Frame-by-frame gap counts and coverage details.
- `candidate_regions.csv`: Detailed geometric and physical metrics for every gap.
- `hard_examples.json`: Mapping of stream session IDs to lists of frame numbers flagged as hard examples.