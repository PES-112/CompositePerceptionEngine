# Walker-surface rules, SANPO test

3850 frames; walker patch = context.walker_surface. Chosen: road bias **2.5**, stripe threshold **45.0**.

| | road_called_safe | road_called_crosswalk | walkable_called_road | crosswalk_called_road | agreement |
|---|---:|---:|---:|---:|---:|
| model as shipped | 0.433 | 0.100 | 0.011 | 0.037 | 0.929 |
| road bias 2.5 | 0.300 | 0.033 | 0.039 | 0.093 | 0.907 |
| + stripe check 45.0 | 0.300 | 0.033 | 0.047 | 0.279 | 0.897 |
