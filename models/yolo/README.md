# YOLO Model Registry

Version-controlled checkpoint registry for CPE YOLO models.

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
best.pt            # tracked — commit checkpoints crucial to the CPE end product
exports/*.onnx     # tracked — commit export artifacts crucial to the CPE end product
exports/*.engine   # tracked — commit export artifacts crucial to the CPE end product
```

Current entries:

| Entry | Purpose |
|---|---|
| `base_yolo26n/` | Base YOLO26n checkpoint registry entry. |
| `cpe_yolo26n_hazards_v3_from_base/` | Preferred v3 detector registry entry and edge export manifests. |

Commit binary weights — checkpoints that are part of the CPE end product must be tracked in Git regardless of size.
