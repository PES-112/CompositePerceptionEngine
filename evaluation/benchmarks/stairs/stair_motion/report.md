# Stairs from walker motion (stairs.StairMotion), SANPO

| split | sessions with labels | labelled min | stair episodes | found | false starts | false / labelled min |
|---|---:|---:|---:|---:|---:|---:|
| train | 107 | 56.5 | 5 | 1 | 7 | 0.124 |
| val | 32 | 17.5 | 2 | 1 | 1 | 0.057 |
| test | 38 | 21.0 | 2 | 0 | 4 | 0.190 |

All 422 pose sessions (210 min): 26 stair starts, 0.124/min (an upper bound on false starts: most walks have no stairs).

Episodes and events:

- train `NEljbgsZaaSaMyQwCHvre5n0RnVexfzE`: episodes [(0.0, 2.7), (10.4, 13.7)] found [False, False]; events []; false []
- train `S7U2VKRn0NwTuhp1Qk67R7l8k5iYZFI-`: episodes [] found []; events [(2.4, 'stairs_up', 0.49)]; false [(2.4, 'stairs_up')]
- train `c4Mh8piOiKI4EEHHQOLTbFUfCDLlq29P`: episodes [] found []; events [(2.4, 'stairs_up', 0.96), (7.6, 'stairs_end', 0.04), (37.2, 'stairs_up', 0.36)]; false [(2.4, 'stairs_up'), (37.2, 'stairs_up')]
- test `eHxtA669WpN381O4ZjVAmG3-3ZUewuXr`: episodes [] found []; events [(19.2, 'stairs_down', -0.82), (25.2, 'stairs_end', 0.02)]; false [(19.2, 'stairs_down')]
- train `huRVT5BlKVdTYbUhQWrr2feodI612PZH`: episodes [] found []; events [(15.2, 'stairs_up', 0.19)]; false [(15.2, 'stairs_up')]
- test `i2jglnBfoIqIIA7ojQGe-4vK07hUm4T3`: episodes [(29.2, 29.2), (32.8, 37.6)] found [False, False]; events []; false []
- train `ixZaSR2bFF5XplHrvc8r8vEMpkc8amOV`: episodes [(9.4, 16.0)] found [False]; events []; false []
- train `j5QOrc58fUSYWM7vbUAn0g4NIRwfOhBg`: episodes [] found []; events [(7.0, 'stairs_up', 0.26), (14.4, 'stairs_end', 0.05)]; false [(7.0, 'stairs_up')]
- val `lmkIchCJ1RIKsZvbb4HjCDl85B2nOicv`: episodes [(2.4, 26.8)] found [True]; events [(5.4, 'stairs_up', 0.39), (11.6, 'stairs_end', 0.05), (13.2, 'stairs_up', 0.32), (22.2, 'stairs_end', 0.03), (25.2, 'stairs_up', 0.33), (29.6, 'stairs_end', -0.03), (56.2, 'stairs_up', 0.22)]; false [(56.2, 'stairs_up')]
- train `mE6bg3DCm-zMCm7_Nt8zOhETHz1XHoTE`: episodes [(0.7, 4.0)] found [True]; events [(2.4, 'stairs_up', 0.57), (21.2, 'stairs_end', -0.03), (27.6, 'stairs_down', -0.43), (42.0, 'stairs_end', 0.05)]; false [(27.6, 'stairs_down')]
- test `nLMkaU45iav7Ix1Iymm2LcA5PWJs0InI`: episodes [] found []; events [(23.4, 'stairs_up', 0.18), (24.4, 'stairs_end', 0.05)]; false [(23.4, 'stairs_up')]
- test `pLx7nHA3heIzjrIMBy4AVjuT6GnwneSR`: episodes [] found []; events [(35.0, 'stairs_up', 0.43), (38.8, 'stairs_end', 0.03)]; false [(35.0, 'stairs_up')]
- train `sW4MEG73P0tPDuuXSUu6NAvx1KJSb10W`: episodes [] found []; events [(2.6, 'stairs_up', 0.62), (15.4, 'stairs_end', -0.02), (54.2, 'stairs_up', 0.46)]; false [(2.6, 'stairs_up')]
- train `uNjwwJ2t4YyphdEcXW6aOstBFJ5m-2M2`: episodes [(3.1, 6.8)] found [False]; events []; false []
- val `vczXAwthxnadTYUS_TiR7IHiqEQrdSJx`: episodes [(45.6, 45.6)] found [False]; events []; false []
- test `wta4PcYCd4Kfjw_DuB8Bd-dO2Du3ZcnN`: episodes [] found []; events [(11.4, 'stairs_up', 0.48), (16.4, 'stairs_end', -0.01)]; false [(11.4, 'stairs_up')]
