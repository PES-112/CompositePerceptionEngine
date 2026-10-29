# Walker-surface rules, SANPO test

3850 frames; walker patch = context.walker_surface. Chosen: road bias **3.25**, stripe threshold **0.0**.

| | road_called_safe | road_called_crosswalk | walkable_called_road | crosswalk_called_road | agreement |
|---|---:|---:|---:|---:|---:|
| model as shipped | 0.133 | 0.033 | 0.029 | 0.158 | 0.926 |
| road bias 3.25 | 0.067 | 0.033 | 0.066 | 0.284 | 0.897 |
| + stripe check 0.0 | 0.067 | 0.033 | 0.066 | 0.284 | 0.897 |
