# Narration Layer Latency Benchmark

No SANPO data or GPU required. Measures src/narration/ in isolation.

**Budget:** v2 plays pre-rendered clips (docs/architecture.md §7), so the narration budget is ~0 ms. The Piper rows measure live synthesis, which v2 does not use on the phone.

| Stage | Case | Latency (ms) | Output |
|---|---|---:|---|
| template | override | 0.00 | Stop! Car very close ahead. |
| template | fast_closing | 0.00 | Motorcycle fast from your left. |
| template | near_static | 0.00 | Person close, ahead. |
| template | context | 0.00 | Bus far right, 25 meters. |
| critical_tts_cache_miss | override | 0.00 | (fallback beep — no clip cached) |
| piper_load | n/a | 443.60 | None |
| piper_synthesize | override | 68.06 | 67116 bytes |
| piper_synthesize | fast_closing | 39.16 | 62508 bytes |
| piper_synthesize | near_static | 30.35 | 48172 bytes |
| piper_synthesize | context | 42.87 | 74796 bytes |

## Reading this

- `template` rows: the critical lane's actual dependency — must stay near-zero, and does (pure Python, no I/O).
- `critical_tts_cache_miss` row: confirms a missing cached clip still returns instantly (a fallback beep), never blocks waiting on a model.
- `piper_load` row: one-time cost at process startup, not per-alert — acceptable even if large, as long as it happens before the first real event.
- `piper_synthesize` rows: real per-utterance cost for the narration (non-critical) lane only. Compare against whatever budget gets documented for this lane.
