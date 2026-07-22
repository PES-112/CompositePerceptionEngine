# Training Workflows

## YOLO26n Hazard Fine-Tuning

Canonical config:

```text
training/configs/cpe_hazard_classes.yaml
```

Current workflow guide:

```text
docs/yolo_training.md
```

Preferred checkpoint:

```text
training/runs/cpe_yolo26n_hazards_v3_from_base/weights/best.pt
```

Training runs should be launched by the user in the terminal when progress visibility matters:

```bash
.venv/bin/python training/scripts/train_yolo26n_hazards.py \
  --weights models/yolo/base_yolo26n/yolo26n.pt \
  --device 0 \
  --epochs 75 \
  --name cpe_yolo26n_hazards_v3_from_base
```

All-class validation:

```bash
.venv/bin/python training/scripts/evaluate_yolo26n_hazards.py \
  --weights training/runs/cpe_yolo26n_hazards_v3_from_base/weights/best.pt \
  --split test \
  --device 0 \
  --skip-previews
```

Retained COCO-class comparison against base YOLO:

```bash
.venv/bin/python training/scripts/compare_yolo26n_retention.py \
  --candidate training/runs/cpe_yolo26n_hazards_v3_from_base/weights/best.pt \
  --base models/yolo/base_yolo26n/yolo26n.pt \
  --split test \
  --device 0
```


Edge export:

```bash
.venv/bin/python training/scripts/export_yolo26n_edge.py \
  --weights training/runs/cpe_yolo26n_hazards_v3_from_base/weights/best.pt \
  --formats onnx \
  --quantize 16 \
  --device 0
```
