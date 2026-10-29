# Walkable-surface segmentation - idd_finetune_sanpo, test

3850 human-annotated frames, 48 sessions. 95% CIs: session bootstrap.

- mIoU **0.635** [0.552, 0.670]
- false-safe rate **0.032** [0.021, 0.044] (unsafe ground called walkable)
- road as walkable 0.090; walkable as unsafe 0.067
- boundary F (±3 px): curb 0.351, stairs 0.104

| Class | IoU | chest | head |
|---|---:|---:|---:|
| walkable | 0.850 | 0.854 | 0.839 |
| crosswalk | 0.547 | 0.537 | 0.669 |
| road | 0.492 | 0.486 | 0.523 |
| curb | 0.255 | 0.258 | 0.250 |
| stairs | 0.606 | 0.609 | 0.000 |
| non-walkable ground | 0.698 | 0.676 | 0.720 |
| obstacle | 0.755 | 0.775 | 0.709 |
| sky | 0.876 | 0.773 | 0.910 |
