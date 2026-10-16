# Binding by timing: the gate PASSED, G1 to G4 at 3 of 3 seeds

Date: 2026-10-16 (runs of 2026-10-15). Rules: `docs/decisions/2026_10_15_rhythm_binding.md`, pushed
before the runs (commit 31bc7bf).

## Verdict

PASS. All four gates of the oscillator design hold at seeds 211, 213 and 214. Oscillators driven by
the rhythm of each place on the tectum map and of each cochlear band synchronise when a seen object
changes in time with the louder tone, and not otherwise. The agent uses this binding: it goes to the
object with the louder tone in 115 of 120 episodes, and in 67 of 120 when the sound's rhythm is
destroyed. Two measures show no effect, and the section "Reported only" gives them.

## Gates

| Gate | Rule | 211 | 213 | 214 |
|---|---|---|---|---|
| G1 selective | coherence on the louder object minus the other object, per episode, Wilcoxon p < 0.05 | +0.607, p 2.6e-08 | +0.580, p 2.8e-08 | +0.616, p 9.1e-13 |
| G2 more than the bids | intercept of that difference on the luminance difference > 0, p < 0.05 | 0.576, p 1.1e-21 | 0.461, p 4.4e-17 | 0.567, p 5.1e-24 |
| G3 used | first contact with the louder object, rhythm above scrambled and above active | see below | | |
| G4 not frozen | group on the light at level 0.5 and on the decoy at level 3.0, each > 0.7 | 0.992 / 0.975 | 0.984 / 0.944 | 0.943 / 0.870 |

Counted steps (step 12 and later, both objects in view): 245, 231 and 263, in all 40 episodes of each
run. The group was on the louder object at 0.984, 0.965 and 0.909 of them.

## G3: the binding is used

First object touched within the first 60 steps, the one with the louder tone:

| Arm | 211 | 213 | 214 | Pooled |
|---|---|---|---|---|
| rhythm (oscillator binding) | 39 of 40 | 38 of 40 | 38 of 40 | 115 of 120 (0.958) |
| scrambled (the sound's rhythm destroyed) | 25 of 40 | 22 of 40 | 20 of 40 | 67 of 120 (0.558) |
| active (no selection) | 21 of 40 | 19 of 40 | 23 of 40 | 63 of 120 (0.525) |

Rhythm against scrambled: per seed +0.350, +0.400, +0.450; pooled one-sided Fisher p 3.3e-14.
Rhythm against active: per seed +0.450, +0.475, +0.375; pooled one-sided Fisher p 8.0e-16.

A second count with separate code, read directly from the step records, gave the same numbers for
all nine runs. The scrambled arm keeps the same sound values, the same image and the same wait; only
the timing is destroyed. So the effect comes from the match in timing.

## Reported only

- **The workspace competition did not change.** With the binding, the leader of the workspace
  changed at 0.0000 of the steps at all three seeds (`probe_binding_decisive.py`). Vision and audio
  both had the factor 1.469 to 1.470, and memory and body 1.480 to 1.499 from the old workspace layer. The
  binding acts on behaviour through the motor map of the tectum, not through the workspace. The
  probe's own label "UNDECIDED" refers to its earlier ten-seed hypothesis, not to this gate.
- **The share of steps in the light was not higher with binding.** Rhythm 0.374, 0.391, 0.374;
  scrambled 0.370, 0.271, 0.420; active 0.338, 0.456, 0.517. The binding takes the agent to the
  object with the louder tone, and that object is the light in only half of the episodes (level
  0.5). The reward is not the target of this test.

## Runs

Revision 31bc7bf, Python 3.12.10. The control with all new flags off (seed 73) gave the md5
`7825766f42d3d85b032f8028c94dc1e0` for `metrics.csv`. Each of the nine runs has 41 lines in
`episodes.csv`, 8001 lines in `metrics.csv`, and 8000 entries in `map_binding.json` and in
`tectal_orienting.json`. Source of the numbers: `scripts/analysis/probe_rhythm_binding.py`, output in
`runs/_audit_followup/oct13/run_rhythm_gate.out`.

## Limits

- The orienting is a fixed reflex, and the gate of the basal ganglia is fixed open. The test shows
  that the binding steers behaviour through the tectum, not that the agent learned to use it.
- The rhythms are two known frequencies (1/10 and 1/6 cycles per step), and the sound has no
  direction by design. Irregular rhythms and a sound with weak direction are open tests.
- The counted steps are few in each episode (about 6), because the agent reaches an object soon
  after it starts to move.
- The threshold (0.55) and the wait (12 steps) were set on separate room layouts before the gate.
- No score of the indicator rubric changes with this document. RPT-2 and GWT-2 stay PARTIAL until
  the owner decides.
