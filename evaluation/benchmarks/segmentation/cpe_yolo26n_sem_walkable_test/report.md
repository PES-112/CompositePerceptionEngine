# Walkable-surface segmentation - cpe_yolo26n_sem_walkable, test

3850 human-annotated frames, 48 sessions. 95% CIs: session bootstrap.

- mIoU **0.615** [0.540, 0.641]
- false-safe rate **0.044** [0.029, 0.062] (unsafe ground called walkable)
- road as walkable 0.173; walkable as unsafe 0.056
- boundary F (±3 px): curb 0.295, stairs 0.062

| Class | IoU | chest | head |
|---|---:|---:|---:|
| walkable | 0.828 | 0.827 | 0.832 |
| crosswalk | 0.590 | 0.593 | 0.555 |
| road | 0.494 | 0.492 | 0.503 |
| curb | 0.239 | 0.251 | 0.215 |
| stairs | 0.474 | 0.492 | 0.000 |
| non-walkable ground | 0.686 | 0.654 | 0.717 |
| obstacle | 0.741 | 0.759 | 0.701 |
| sky | 0.868 | 0.760 | 0.905 |
