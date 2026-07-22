# YOLO Model Registry

Local-only checkpoint registry for CPE YOLO models.

Each model folder should be named:

```text
cpe_yolo26n_<task>_v<number>
```

Recommended contents:

```text
README.md          # tracked, explains source run and intended use
args.yaml          # tracked if small
results.csv        # tracked if small
exports/*.json     # tracked export manifests when compact
best.pt            # local only, ignored by Git
exports/*.onnx     # local only, ignored by Git
exports/*.engine   # local only, ignored by Git
```

Current entries:

| Entry | Purpose |
|---|---|
| `base_yolo26n/` | Local base YOLO26n checkpoint registry entry. |
| `cpe_yolo26n_hazards_v3_from_base/` | Preferred v3 detector registry entry and edge export manifests. |

Do not commit binary weights. Store release checkpoints in external artifact storage when needed.
