# Listening study with blind and low-vision participants (plan)

Status: **plan only, not approved or run** (2026-10-17). It is move 9 of `reports/CPE v2 next steps.md`.
It answers the question the replay can't: how many alerts per minute, and which alerts, blind and
low-vision (BLV) walkers will actually accept. The literature gives no tolerable alert rate for BLV
pedestrians (`research_notes/CPE v2 next steps/evaluation_methodology.md`, Q1). This is what an
ASSETS / IMWUT reviewer will ask for. A CV-workshop paper doesn't need it.

It is a **seated, remote listening study with no walking**. It measures preference, perceived
usefulness, annoyance and tolerable alert rate for recorded alert streams. It does **not** measure
safety, collision avoidance or real-world use, and the paper must not claim those from it.

## 1. Research questions

1. **RQ1, engine preference:** at a matched alert budget, do BLV listeners prefer v2's announcements
   (GRU arrival × severity, announcement manager) over v1's (K0 routing) and over nearest-in-corridor?
2. **RQ2, tolerable rate:** which of the frozen budgets (2, 5 and 10 false alerts/min; the replay's
   `budget_*` points) do listeners accept, and where does annoyance overtake usefulness?
3. **RQ3, order:** when two objects compete, does the announced one match what the listener would
   want to hear first? This gives a small human check on the VLM referee's question (§4.7).

## 2. Stimuli

- **Clips.** 12 SANPO official-**test** sessions, 30–60 s each, chosen before any rating:
  - 4 with a crossing mover (person or bicycle);
  - 4 approaching static hazards;
  - 4 busy scenes with several candidates.

  Selection is stratified by the replay's arrival count and is fixed in a pre-registered list.
- **Audio per clip and condition.** Each clip is a soundtrack of the engine's announcements, placed
  at the replay's announcement times (`tools/replay_threat_engine.py` logs). They use the
  pre-rendered phrase clips (`templates.all_phrases`, pending_work P4.1; same voice and loudness
  for every condition) over a neutral street-ambience bed. SANPO has no audio, so the bed is the same
  generic recording for every clip.
- **Scene preface.** A sighted researcher writes a 2–3 sentence description of each clip from the
  video and labels, for example "You are walking along a sidewalk. A cyclist approaches from your
  left after about ten seconds." It is read in a different voice before the clip. The text never
  mentions the engines.
- **Conditions per clip** (blinded as A–E, order counterbalanced):

  | Condition | Engine | Operating point (chosen on validation, frozen) |
  |---|---|---|
  | A | v2 + GRU | `budget_2` |
  | B | v2 + GRU | `budget_5` |
  | C | v2 + GRU | `budget_10` |
  | D | nearest-in-corridor | `budget_5` |
  | E | v1 K0 as built | its replay setting (≈ 100 alerts/min) |

  If 5 conditions × 12 clips is too long (a pilot decides), drop C and keep A, B, D and E.

**Dependencies:**
- P4.1 (render the clip library).
- A small exporter that writes per-arm announcement logs from the replay. Not written yet; it reads
  the same `run_v1` / `run_v2` logs the replay builds.

## 3. Procedure (≈ 60–75 min, remote, video call or phone)

1. Consent (§6). Short practice clip.
2. For each clip: the preface, then each condition in counterbalanced order (Williams-design Latin
   square over participants). Ratings follow each condition.
3. Per condition, 7-point scales, read aloud and answered verbally or by keypad:
   - "The alerts told me what I needed to know." (usefulness)
   - "The alerts were annoying." (annoyance)
   - "I would trust these alerts while walking." (trust)
   - "How mentally demanding was it to follow these alerts?" (raw NASA-TLX mental demand)
4. After each clip, a forced choice between the two most-preferred conditions: "Which would you
   rather walk with?"
5. **RQ3.** On 10 short excerpts where engines disagree, the listener hears the scene preface plus a
   list of the 2–3 objects and says which should be announced first. These frames come from the
   referee sample, so they also add to the human-ceiling labels.
6. **Tolerable rate.** At the end, the listener rates a synthetic stream played at 1, 2, 5, 10 and 20
   alerts/min (same phrases, randomised order) as acceptable or not for an hour's walk.
7. A semi-structured interview: when an alert was wrong, what was wrong; what they would change.

## 4. Participants

- **Who:** 12–16 adults who are blind or have low vision, travel independently with a cane or a dog,
  and have no hearing impairment that affects speech in noise. Recruit through organisations of and
  for blind people near PES University. Pay participants at the local research rate for their time.
- **Why 12–16:** it matches comparable ASSETS / CHI evaluations (BBeep n = 6; Mobilio n = 14). With 16
  participants × 12 clips the within-subject paired comparisons have enough trials for the planned
  mixed models. This is not a powered test of small effects, and the paper should say so.
- **Piloting:** two sighted, blindfolded pilots check timing and audio levels only. Their data is
  excluded.

## 5. Analysis (pre-registered before the first participant)

- **Ratings:** cumulative-link mixed models (ordinal), condition as fixed effect, random intercepts for
  participant and clip. Report contrasts A–E vs. B with 95% CIs.
- **Forced choices:** Bradley–Terry model with participant random effects; win probabilities with CIs.
- **Tolerable rate:** per participant, the highest acceptable rate. Report the median and IQR, plus a
  logistic mixed model of "acceptable" against log(rate).
- **RQ3:** agreement with the engines' picks against chance (1/k), Gwet's AC1 with the referee's
  humans. Same coding as `evaluation/vlm_referee.py`.
- Qualitative: thematic analysis of the interviews by two coders.

**What the results can support:** preference and perceived usefulness of alert streams; a measured
tolerable-rate range for this group, which fills the gap in the literature; and whether the announced
order matches listeners' choices. **What they cannot support:** safety, fewer collisions, or
real-world usability.

## 6. Ethics and data

- **Approval:** PES University's institutional ethics committee. Plan 4–8 weeks, so submit as soon as
  the protocol is frozen. No physical risk; participants are seated.
- **Accessible consent:** the information sheet and consent form come in screen-reader-accessible
  digital form, read aloud on request, with verbal consent recorded. Participants can stop at any
  time without giving a reason, and their data is then deleted.
- **Data:**
  - Ratings and choices are stored under a participant code.
  - Interview audio is transcribed and then deleted.
  - No video of participants.
  - Demographics are limited to age band, vision category, mobility aid and years of independent
    travel.
- **Stimuli:** SANPO (CC BY 4.0) frames are never shown to participants and are only used to write
  the prefaces.

## 7. Timeline

| Week | Step |
|---|---|
| 0 | Freeze protocol and clip list; submit ethics |
| 0–2 | Render clips (P4.1), write the log exporter, build the accessible playback page, pilot |
| 4–8 | Ethics approval (estimate) |
| 8–11 | Sessions (3–4 per week) |
| 11–13 | Analysis and write-up |
