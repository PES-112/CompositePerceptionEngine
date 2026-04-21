# SANPO Kinetic Replay (Colab Pack)

This folder is ready to upload to Google Colab. It will:
- Parse one processed JSONL session (fact-sheet format)
- Stream SANPO session frames directly from the public bucket
- Recompute per-object kinetic score with your custom formula
- Export an annotated MP4

## Files
- `sanpo_kinetic_replay_colab.ipynb`: main notebook to run in Colab
- `sanpo_bucket_replay.py`: helper module (bucket streaming + rendering)
- `requirements_colab.txt`: optional dependency list

## Quick Colab Usage
1. In Colab, upload this whole folder (or upload the zip and unzip it).
2. Open `sanpo_kinetic_replay_colab.ipynb`.
3. Run cells in order.
4. Upload your `.jsonl` session file when prompted.
5. Set `SESSION_ID` to the SANPO session id to stream from `gs://gresearch/sanpo_dataset/v0/sanpo-synthetic/`.

## Notes
- SANPO bucket access here uses an anonymous client (public data, no key required).
- The JSONL includes object facts in text, not bounding boxes; this notebook overlays a ranked kinetic panel per frame.
- If some frame ids are missing, verify the JSONL and SANPO `SESSION_ID` correspond to the same source session.
