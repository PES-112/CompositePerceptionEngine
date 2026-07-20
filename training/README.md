# Training Workflows

## YOLO26n Hazard Fine-Tuning

Canonical config:

```text
training/configs/cpe_hazard_classes.yaml
```

Current v1 checkpoint:

```text
training/runs/cpe_yolo26n_hazards/weights/best.pt
models/yolo/cpe_yolo26n_hazards_v1/best.pt
```

The `models/yolo/...` copy is a local model registry entry. Binary weights are ignored by Git.

## V2 Fine-Tuning

Follow:

```text
docs/plan_yolo_v2_finetune.md
```

Start v2 from v1 `best.pt` after adding better stairs/pothole data and retained-class validation data:

```bash
.venv/bin/python training/scripts/train_yolo26n_hazards.py   --weights training/runs/cpe_yolo26n_hazards/weights/best.pt   --device 0   --batch -1   --epochs 50   --cache disk   --name cpe_yolo26n_hazards_v2   --export-formats
```

## Evaluation

All-class candidate validation:

```bash
.venv/bin/python training/scripts/evaluate_yolo26n_hazards.py   --weights training/runs/cpe_yolo26n_hazards_v2/weights/best.pt   --split test   --device 0
```

Retained COCO-class comparison against base YOLO:

```bash
.venv/bin/python training/scripts/compare_yolo26n_retention.py   --candidate training/runs/cpe_yolo26n_hazards_v2/weights/best.pt   --base yolo26n.pt   --split test   --device 0
```
