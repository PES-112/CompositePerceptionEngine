# Kinetic Score Evaluation Benchmark

Results from evaluating the production kinetic score **K0** (`sev × v² / max(d, ε)`).

> The former K0–K5 comparison harness (`evaluation/kinetic_score_comparison.py`,
> `evaluation/threat_score_eval.py`) has been **deleted**. It was circular — it graded each formula
> against a "ground truth" computed by re-running that same formula on a future frame — and its
> discriminating metric was saturated, passing five of six candidates. K1–K5 were dummies and are
> gone from `src/perception_stack/physics.py`.
>
> See **`docs/kinetic_score_opinion.md`** for the replacement strategy.

## What replaces it

K0 is defended by **ablation of its own terms**, not by a contest against strawmen:

| Variant | Claim under test |
|---|---|
| `sev · v²/d` | K0 as-is (baseline) |
| `sev · v/d` | is the `v²` exponent doing work? |
| `v²/d` | is class severity doing work? |
| `sev/d` | is velocity doing work? |
| `-(d − D_haz)/v` | is K beaten by plain time-to-hazard? |

## Metrics requiring no ground truth

Run these first — they may settle the question before any labelling.

1. **Flicker rate** — how often `argmax K` changes identity between consecutive frames.
2. **Rank stability** — Kendall τ between rankings on clean vs. depth-perturbed input.
3. **Temporal smoothness** — mean `|K(t) − K(t−1)| / mean K` per track.
4. **Tie rate** — frames where the top two objects fall within 5%.
5. **Complementarity with SLM-1** — disagreement rate between `argmax K` and SLM-1's pick.
6. **Future self-consistency** — does `argmax K` at T match `argmax K` at T+H?

## Automatic ground truth (eliminates, never selects)

An object is a **true encounter** at frame T if, within horizon H, its *measured* `distance_m` drops
below `D_haz` while `|bearing_deg| < θ`. Uses only `distance_m` and `bearing_deg` — never velocity,
severity, or any K. Report as a sensitivity grid over `(H, D_haz, θ)`, not a single setting.

## The one question needing a human

Whether K0's `v²` and its class severity weights are *right* is a value judgment, settled only by a
**blinded referee on disagreement frames** (§3 of the opinion doc). Budget ~100–300 frames.

## Prerequisites (blocking)

Stage-1 perception CSVs do not exist yet. Generate them with `tools/run_perception.py` over the ten
sessions in `../sanpo_edge_realtime/selected_10_sessions.json`, on the machine holding SANPO.
Bootstrap at the **session** level — frames within a session are autocorrelated.
