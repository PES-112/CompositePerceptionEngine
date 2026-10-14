# Ensemble threat ranking

Combines the existing arrival predictors into one ranking of **which object reaches the walker
first**. A later step would blend that with K0's consequence score to decide what gets announced.

| Level | What | Status |
|---|---|---|
| 1 | Arrival ensemble (`ensemble.py`; held-out `heldout.py`) | **Implemented and run**: 5-fold CV on 387 and 462 sessions; held-out on SANPO's official test split |
| 2 | Announcement priority (arrival × severity^γ × context) | **Implemented** in `src/threat_engine/priority.py`, replay-tested (`tools/replay_threat_engine.py`) |

## Why

On 387 SANPO sessions (`evaluation/benchmarks/gru_encounter_eval/run_2026_09_29_arrival/`), the
arrival-time GRU and the "nearest object in the forward cone" rule **tie**. They get there
differently:
- The GRU reads 1.6 s of track history and has a steadier top pick.
- The rule reacts sharply to the current distance and bearing.

Two equally good predictors with different error patterns are the classic case where combining
them can beat both.

## Level 1 — arrival ensemble

**Components.** Per-object scores on the same pooled cross-validation test frames:
- the saved out-of-fold predictions of a `models/GRU` run set trained with `--time-bins`, for both
  `gru` and `gru_attention`;
- the `nearest_in_cone` and `TTC` baselines.

**Combiners.**
- **`ensemble_rank`** — the average percentile rank across components. Nothing is fitted and labels
  are never used.
- **`ensemble_stack`** — a softmax regression over the components plus distance, bearing and TTC
  features, predicting the arrival slice. It gives both "will it arrive" and "how soon".
  - It is fitted **out-of-fold**: fold F is scored by a stacker trained only on the other folds.
    A unit test checks it never sees fold F's labels.
  - Fitted weights are saved to `stacker.json`.

**Run it:**

```bash
python -m models.Ensemble.ensemble --run-name arrival387_2026_09_29
# -> evaluation/benchmarks/ensemble_eval/arrival387_2026_09_29/{report.md, paired_differences.json, stacker.json}
```

It takes about 1 minute on a CPU. No training is needed beyond the GRU runs it reads. Tests:
`python -m unittest tests.test_ensemble`.

### Held-out results on SANPO's official test split (quote these)

```bash
python -m models.Ensemble.heldout --run-name official462_2026_10_14
# -> evaluation/benchmarks/heldout_official_split/official462_2026_10_14/report.md
```

GRUs trained on the official train sessions; the stacker is fitted on the **validation** sessions'
predictions only, so the 71 test sessions are touched once. Neither ensemble significantly beats the
plain GRU or nearest-in-cone (stack vs GRU: AUROC +0.005 [−0.015, +0.016], arrival_order +0.011
[−0.051, +0.043]). The cross-validation gains below were small and don't survive the held-out test.

### Results on all 462 sessions (5-fold CV)

Report: `evaluation/benchmarks/ensemble_eval/arrival462_2026_09_29/report.md`. The verdict is the
same as at 387 sessions below.

| ensemble_stack minus | AUROC | arrival_order |
|---|---|---|
| nearest_in_cone | **+0.023** [+0.013, +0.034] | +0.007 [−0.008, +0.021] |
| gru | **+0.007** [+0.001, +0.014] | +0.012 [−0.002, +0.028] |
| gru_attention | +0.001 [−0.003, +0.009] | **+0.022** [+0.009, +0.037] |
| ensemble_rank | **+0.015** [+0.009, +0.021] | **+0.015** [+0.003, +0.025] |

ensemble_stack reaches AUROC 0.894 and arrival_order 0.688; nearest_in_cone scores 0.871 and
0.681.

### Results (387 sessions, 3 seeds, 95% paired session-bootstrap CIs)

| Method | AUROC | arrival_order | soonest_top1 | flicker_rate ↓ |
|---|---:|---:|---:|---:|
| **ensemble_stack** | **0.889** | **0.681** | 0.622 | 0.652 |
| ensemble_rank | 0.874 | 0.675 | 0.619 | 0.653 |
| gru | 0.877 | 0.669 | 0.608 | **0.593** |
| gru_attention | 0.886 | 0.656 | 0.597 | 0.664 |
| nearest_in_cone | 0.870 | 0.680 | **0.652** | 0.731 |
| K0 | 0.569 | 0.586 | 0.460 | 0.666 |

| ensemble_stack minus | AUROC | arrival_order |
|---|---|---|
| nearest_in_cone | **+0.019** [+0.008, +0.032] | +0.001 [−0.010, +0.012] |
| gru | **+0.012** [+0.005, +0.021] | +0.012 [−0.006, +0.030] |
| gru_attention | +0.003 [−0.003, +0.009] | **+0.026** [+0.007, +0.044] |
| K0 | **+0.320** [+0.259, +0.359] | **+0.096** [+0.058, +0.129] |

**Verdict.** The success criterion (beat nearest-in-cone *and* the GRU on `arrival_order`) is
**not met**.
- **Best "will it come close" predictor.** The stacked ensemble is the best one tested. It
  significantly beats every single component on AUROC except the attention GRU, which it ties.
- **No better at ordering by arrival.** Every learned or combined ranker ties the nearest-in-cone
  rule. That points to a ceiling in what the current perception features (depth, closing speed,
  bearing) can say about arrival order, not to a missing model.
- **Rank averaging adds nothing** over the GRU alone.
- **Full report:** `evaluation/benchmarks/ensemble_eval/arrival387_2026_09_29/report.md`.

**Caveat.** Each fold's stacker is trained on the other folds' out-of-fold predictions. Those came
from GRUs whose training data included the scored fold. This is standard cross-validation stacking:
the stacker never sees the scored fold's labels. It is not a fully nested design.

## Level 2 — announcement priority (implemented)

Lives in the runtime, `src/threat_engine/` (`docs/architecture.md` §6.4):

1. **Safety envelope first.** TTC ≤ 1 s inside the forward walking cone, on Kalman-fused velocity,
   always wins (`envelope.py`).
2. **Then by priority** = P(arrives within 4 s) × severity(class)^γ × context (`priority.py`),
   expected arrival time breaking ties. γ = 0.25 is declared (a bus counts 1.9× a person, not 13×).
   It uses the plain GRU, not the stack: the stack adds nothing held-out and needs both GRUs.
3. **The announcement manager** decides whether to speak at all (`announcer.py`).

System-level replay (`evaluation/benchmarks/threat_engine_replay/official_2026_10_14/report.md`):
on the official test sessions, v2 announces 17/min with precision 0.59, against v1's 119/min at 0.24,
but warns about fewer arrivals (26% vs 65%). There is still no ground truth for "what should be
announced first"; the blinded VLM referee (`evaluation/vlm_referee.py`) on frames where v2's order
differs from K0's is the planned check (`docs/pending_work.md` P2.4).
