# IDD fine-tune vs original detector

IDD test mAP50 (IDD-labelled classes); retention on 1000 SANPO val frames at conf 0.3.

| class | IDD AP50 old | new | SANPO boxes old | new | retention | phone uses |
|---|---:|---:|---:|---:|---:|---|
| person | 0.287 | 0.405 | 820 | 1491 | 0.909 | new |
| bicycle | 0.071 | 0.127 | 12 | 21 | 0.417 | new |
| car | 0.391 | 0.526 | 145 | 838 | 0.669 | new |
| motorcycle | 0.024 | 0.464 | 0 | 72 | — | new |
| bus | 0.325 | 0.507 | 15 | 80 | 0.400 | new |
| truck | 0.093 | 0.220 | 16 | 144 | 0.562 | new |
| traffic light | 0.007 | 0.052 | 35 | 2 | 0.057 | new |
| stop sign | 0.995 | 0.995 | 1 | 3 | 0.000 | legacy |
| fire hydrant | — | — | 6 | 0 | 0.000 | legacy |
| pole | 0.774 | 0.667 | 24 | 27 | 0.583 | legacy |
| bollard | 0.522 | 0.153 | 83 | 1 | 0.012 | legacy |
| stairs | 0.652 | 0.017 | 32 | 0 | 0.000 | legacy |
| crosswalk | 0.995 | 0.059 | 5 | 3 | 0.000 | legacy |
| pothole | — | — | 0 | 0 | — | legacy |
| puddle | 0.040 | 0.000 | 0 | 0 | — | legacy |
| dog | 0.004 | 0.104 | 3 | 10 | 0.333 | new |
| bench | 0.995 | 0.000 | 42 | 4 | 0.095 | legacy |

LEGACY_CLASSES = ['stop sign', 'fire hydrant', 'pole', 'bollard', 'stairs', 'crosswalk', 'pothole', 'puddle', 'bench']
