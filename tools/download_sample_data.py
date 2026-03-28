"""
download_sample_data.py
=======================
CPE Phase 1 — Downloads a realistic SANPO-Synthetic sample (≤3GB).

Correct bucket path: gs://gresearch/sanpo_dataset/v0/sanpo-synthetic/

Requirements:
    pip install google-cloud-storage

What gets downloaded:
  - A configurable number of SANPO-Synthetic sessions (short video clips)
  - Each session: left-eye RGB video + paired dense depth maps
  - Target: ≤3GB total

Output folder structure:
  data/sanpo/raw/<session_id>/   ← downloaded files per session
  data/sanpo/sample/rgb/         ← extracted frames (run prepare_sample_data.py after)
  data/sanpo/sample/depth/       ← extracted depth maps
  data/processed/                ← Stage 1 CSV outputs
  data/training/                 ← Stage 2 JSONL outputs
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
BUCKET_NAME          = "gresearch"
SANPO_BASE_PREFIX    = "sanpo_dataset/v0/sanpo-synthetic/"

TARGET_SESSIONS      = 3           # exactly 3 scenes
MAX_DOWNLOAD_GB      = 3.0         # hard cap — stop if total exceeds this
SCAN_SESSIONS        = 60          # how many sessions to scan when sorting by size

DIRS = [
    "data/sanpo/raw",
    "data/sanpo/sample/rgb",
    "data/sanpo/sample/depth",
    "data/processed",
    "data/training",
]


# ── Folder setup ──────────────────────────────────────────────────────────────

def create_dirs():
    for d in DIRS:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✅  ./data/ folder structure ready\n")


# ── YOLO26n check ─────────────────────────────────────────────────────────────

def check_yolo26():
    print("=== Checking YOLO26n ===")
    try:
        from ultralytics import YOLO
        model = YOLO("yolo26n.pt")
        print(f"✅  yolo26n.pt — task={model.task}, {len(model.names)} COCO classes\n")
    except ModuleNotFoundError:
        print("⚠️  ultralytics not found in this env.")
        print("    Make sure you activate the venv first:")
        print("    e:\\capstone\\CompositePerceptionEngine\\venv\\Scripts\\activate\n")
    except Exception as e:
        print(f"⚠️  {e}\n")


# ── GCS helpers ───────────────────────────────────────────────────────────────

def get_client():
    """Return an anonymous GCS client (SANPO-Synthetic is public)."""
    try:
        from google.cloud import storage
        from google.auth.credentials import AnonymousCredentials
        return storage.Client(credentials=AnonymousCredentials(), project="none")
    except ImportError:
        print("❌  google-cloud-storage not installed.")
        print("    Run: pip install google-cloud-storage")
        sys.exit(1)


def list_sessions(client, bucket_name: str, prefix: str) -> list[str]:
    """List top-level session directories under a prefix."""
    blobs    = client.list_blobs(bucket_name, prefix=prefix, delimiter="/")
    _ = list(blobs)             # must consume iterator to populate prefixes
    sessions = [p.rstrip("/").split("/")[-1] for p in blobs.prefixes]
    return sorted(sessions)


def session_size_bytes(client, bucket_name: str, prefix: str) -> int:
    """Return total byte size of all blobs under a session prefix."""
    return sum(b.size for b in client.list_blobs(bucket_name, prefix=prefix))


def download_session(client, bucket_name: str, gcs_prefix: str, local_dir: Path):
    """Download all blobs in a session to local_dir, skip already-existing files."""
    blobs = list(client.list_blobs(bucket_name, prefix=gcs_prefix))
    for blob in blobs:
        rel   = blob.name[len(gcs_prefix):]   # path relative to session root
        dest  = local_dir / rel
        if dest.exists() and dest.stat().st_size == blob.size:
            print(f"    ⏭  {dest.name} (cached)")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        mb = blob.size / 1_048_576
        print(f"    ↓ {dest.name}  ({mb:.1f} MB)")
        blob.download_to_filename(str(dest))


# ── Main download ─────────────────────────────────────────────────────────────

def download_sanpo():
    print("=== Downloading 3 largest SANPO-Synthetic sessions (≤ 3 GB) ===")
    print(f"    Bucket : gs://{BUCKET_NAME}/{SANPO_BASE_PREFIX}")
    print(f"    Auth   : Anonymous (public bucket)\n")

    client = get_client()

    # 1. List available sessions
    print("  Listing available sessions (scanning up to first 60)...")
    sessions = list_sessions(client, BUCKET_NAME, SANPO_BASE_PREFIX)

    if not sessions:
        print("  ❌  No sessions found. Bucket path may have changed.")
        print(f"      Browse: https://console.cloud.google.com/storage/browser/{BUCKET_NAME}/{SANPO_BASE_PREFIX}")
        _print_manual_fallback()
        return

    # 2. Measure sizes for the first SCAN_SESSIONS sessions, pick 3 largest
    print(f"  Measuring sizes for up to {SCAN_SESSIONS} sessions (this takes ~10s)...\n")
    sized: list[tuple[int, str, str]] = []   # (bytes, session_id, prefix)
    for session_id in sessions[:SCAN_SESSIONS]:
        prefix = f"{SANPO_BASE_PREFIX}{session_id}/"
        size   = session_size_bytes(client, BUCKET_NAME, prefix)
        sized.append((size, session_id, prefix))

    # Sort descending by size → pick the 3 largest
    sized.sort(reverse=True)
    selected = sized[:TARGET_SESSIONS]

    total_bytes = sum(s[0] for s in selected)
    total_gb    = total_bytes / 1_073_741_824

    print(f"  Top {TARGET_SESSIONS} largest sessions selected:")
    for size, sid, _ in selected:
        print(f"    {sid[:16]}...  {size/1_048_576:>7.1f} MB")
    print(f"  Total: {total_gb:.2f} GB\n")

    if total_gb > MAX_DOWNLOAD_GB:
        print(f"  ⚠️  Total ({total_gb:.2f} GB) exceeds {MAX_DOWNLOAD_GB} GB cap.")
        print(f"      Trimming to first 2 sessions.")
        selected = selected[:2]
        total_bytes = sum(s[0] for s in selected)
        print(f"      New total: {total_bytes/1_048_576:.0f} MB\n")

    # 3. Download — skips files already on disk (safe to resume)
    for size, session_id, prefix in selected:
        local_dir = Path("data/sanpo/raw") / session_id
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [{session_id[:20]}]  →  {local_dir}")
        download_session(client, BUCKET_NAME, prefix, local_dir)

    print(f"""
✅  SANPO download complete — {len(selected)} sessions under data/sanpo/raw/

Next step — extract RGB frames + depth maps from each session:
  python tools/prepare_sample_data.py \\
      --video    data/sanpo/raw/<session_id>/<rgb_file>.mp4 \\
      --depth    data/sanpo/raw/<session_id>/<depth_file>.zip \\
      --out_rgb  data/sanpo/sample/rgb \\
      --out_dep  data/sanpo/sample/depth \\
      --source   sanpo --stride 3

Then run Stage 1 (perception → CSV):
  python tools/run_perception.py \\
      --rgb_dir   data/sanpo/sample/rgb \\
      --depth_dir data/sanpo/sample/depth \\
      --out       data/processed/sanpo_test.csv \\
      --source    sanpo --fps 30
""")


def _print_manual_fallback():
    print("""
  Manual fallback — use gcloud SDK:
    1. Install: https://cloud.google.com/sdk/docs/install
    2. Run (no login needed for public bucket):
         gcloud storage ls gs://gresearch/sanpo_dataset/v0/sanpo-synthetic/
         gcloud storage cp -r gs://gresearch/sanpo_dataset/v0/sanpo-synthetic/<id>/ data/sanpo/raw/
""")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    create_dirs()
    check_yolo26()
    download_sanpo()
