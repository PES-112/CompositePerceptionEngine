# YOLO Model Registry

Local-only checkpoint registry for CPE YOLO models.

Each model folder should be named:

```text
cpe_yolo26n_<task>_v<number>
```

Recommended contents:

```text
README.md      # tracked, explains source run and intended use
args.yaml      # tracked if small
results.csv    # tracked if small
best.pt        # local only, ignored by Git
```

Do not commit binary weights. Store release checkpoints in external artifact storage when needed.
