"""
Train the walkable-surface segmentation model (docs/architecture.md §6.3):
YOLO26s-sem (Cityscapes-pretrained) fine-tuned on SANPO-Real masks remapped to
CPE's 8-class scheme (src/segmentation/labels.py).

The 17-class hazard detector (cpe_yolo26n_hazards_v3_from_base) stays the
object source; this model only supplies the ground: surface under each
object, the walkable edge ahead, and the flat-ground mask for the depth sweep.

Build the data first (lab server; ~3.5 MB per source frame downloaded):
    python tools/build_sanpo_segmentation.py --out data/sanpo_walkable_seg --workers 32

Then:
    python training/scripts/train_yolo26_sem_walkable.py \
        --data data/sanpo_walkable_seg/cpe_walkable.yaml --device 0 --name cpe_yolo26s_sem_walkable
    python training/scripts/evaluate_yolo26_sem_walkable.py \
        --weights training/runs/cpe_yolo26s_sem_walkable/weights/best.pt \
        --data data/sanpo_walkable_seg --split test

Choices (declared before training, from research_notes/CPE v2 next steps/segmentation.md):
  - s over n: 80.8 vs 78.3 Cityscapes mIoU; on-device cost is a few ms either way at 2-5 Hz.
  - imgsz 1024: curbs and stair edges are thin; Ultralytics warns that thin structures degrade at
    low imgsz. Export at the size that is trained.
  - Model selection on val (curated replay-val sessions + 15% of the other train sessions);
    test is scored once, on HUMAN_ANNOTATED frames only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data" / "sanpo_walkable_seg" / "cpe_walkable.yaml"
DEFAULT_PROJECT = PROJECT_ROOT / "training" / "runs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default="yolo26s-sem.pt", help="Cityscapes-pretrained YOLO26 semantic checkpoint.")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=-1, help="-1 = Ultralytics AutoBatch.")
    p.add_argument("--device", default="0", help="0 = first CUDA GPU, mps = Apple silicon GPU, cpu.")
    p.add_argument("--workers", type=int, default=16, help="Data loader workers (4-6 on a laptop).")
    p.add_argument("--cache", choices=["false", "ram", "disk"], default="disk")
    p.add_argument("--lr0", type=float, default=0.002)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default="cpe_yolo26s_sem_walkable")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--export-coreml", action="store_true",
                   help="After training, export FP16 Core ML (argmax in graph) at --imgsz.")
    return p.parse_args()


def main() -> None:
    from ultralytics import YOLO

    args = parse_args()
    model = YOLO(args.weights)
    model.train(
        data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
        workers=args.workers, cache=False if args.cache == "false" else args.cache, lr0=args.lr0,
        patience=args.patience, seed=args.seed, project=str(DEFAULT_PROJECT), name=args.name,
        resume=args.resume, fliplr=0.5, plots=True,
    )
    best = DEFAULT_PROJECT / args.name / "weights" / "best.pt"
    print(f"best checkpoint: {best}")
    if args.export_coreml:
        # FP16, not INT8: no speed gain at this size on the A19 Pro, and the risk falls on curb edges.
        YOLO(str(best)).export(format="coreml", imgsz=args.imgsz, half=True)


if __name__ == "__main__":
    main()
