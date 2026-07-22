# YOLO26n Version Comparison

Generated: 2026-07-22

## Overall Test Metrics

| Version | Checkpoint | mAP50 | mAP50-95 | Change mAP50 vs Previous | Change mAP50-95 vs Previous |
|---|---|---:|---:|---:|---:|
| v1 | `training/runs/cpe_yolo26n_hazards/weights/best.pt` | 0.651 | 0.432 |  |  |
| v2 | `training/runs/cpe_yolo26n_hazards_v2_full_autobatch-2/weights/best.pt` | 0.599 | 0.424 | -0.052 | -0.008 |
| v3 | `training/runs/cpe_yolo26n_hazards_v3_from_base/weights/best.pt` | 0.641 | 0.459 | 0.042 | 0.035 |

Notes:
- v1 benchmark only contains the original compact hazard benchmark; many retained COCO classes were not evaluated there.
- v2 and v3 use the same cleaned `data/yolo_finetune_v2_full` held-out test split.
- v3 starts from base `models/yolo/base_yolo26n/yolo26n.pt`, which repaired most retained-class degradation seen in v2.

## Per-Class mAP50

| Class | v1 | v2 | v3 | v3 - v2 | v3 - v1 |
|---|---:|---:|---:|---:|---:|
| person |  | 0.514 | 0.570 | 0.056 |  |
| bicycle |  | 0.304 | 0.449 | 0.145 |  |
| car |  | 0.440 | 0.482 | 0.043 |  |
| motorcycle |  | 0.954 | 0.985 | 0.031 |  |
| bus |  | 0.490 | 0.597 | 0.107 |  |
| truck |  | 0.280 | 0.370 | 0.091 |  |
| traffic light |  | 0.344 | 0.370 | 0.026 |  |
| stop sign |  | 0.460 | 0.509 | 0.049 |  |
| fire hydrant |  | 0.608 | 0.654 | 0.046 |  |
| pole | 0.937 | 0.954 | 0.938 | -0.016 | 0.001 |
| bollard | 0.934 | 0.911 | 0.903 | -0.008 | -0.031 |
| stairs | 0.086 | 0.471 | 0.474 | 0.003 | 0.387 |
| crosswalk |  | 0.976 | 0.990 | 0.013 |  |
| pothole | 0.647 | 0.646 | 0.673 | 0.027 | 0.026 |
| puddle |  | 0.699 | 0.732 | 0.033 |  |
| dog |  | 0.498 | 0.529 | 0.031 |  |
| bench |  | 0.633 | 0.666 | 0.033 |  |

## Retained COCO F1 vs Base YOLO26n

| Class | v2 F1 Delta vs Base | v3 F1 Delta vs Base | Verdict |
|---|---:|---:|---|
| person | -0.055 | 0.005 | improved/retained |
| bicycle | -0.120 | 0.001 | repaired vs v2 |
| car | 0.123 | 0.139 | improved/retained |
| motorcycle | 0.216 | 0.279 | improved/retained |
| bus | 0.011 | 0.109 | improved/retained |
| truck | -0.267 | 0.080 | repaired vs v2 |
| traffic light | -0.023 | 0.004 | improved/retained |
| stop sign | -0.110 | -0.041 | repaired vs v2 |
| fire hydrant | -0.027 | 0.078 | improved/retained |
| dog | -0.156 | -0.136 | remaining degradation |
| bench | 0.041 | 0.112 | improved/retained |

## Files

- `overall_version_metrics.csv`
- `per_class_version_metrics.csv`
- `retention_vs_base_metrics.csv`
