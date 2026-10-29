# Walker-surface rules, SANPO test

981 frames; walker patch = context.walker_surface. Chosen: road bias **2.5**, stripe threshold **0.0**.

| | road_called_safe | road_called_crosswalk | walkable_called_road | crosswalk_called_road | agreement |
|---|---:|---:|---:|---:|---:|
| model as shipped | 0.834 | 0.008 | 0.000 | 0.000 | 0.157 |
| road bias 2.5 | 0.639 | 0.005 | 0.000 | 0.000 | 0.353 |
| + stripe check 0.0 | 0.639 | 0.005 | 0.000 | 0.000 | 0.353 |
