# Binding stage 3, option B: the tectum map grouped into connected areas

Date: 2026-10-14. Owner decision of the same day: design A of
`docs/decisions/2026_10_14_binding_stage3_map_oscillators.md`, tested first as option B
(the end result that Terman and Wang prove for LEGION) and then as option A (the full
oscillator equations).

## The change

`--map-binding limit`, default `off`. In each step the tectum computes a visual drive
and an audio drive on its 16 by 16 map (`models/core/map_binding.py`). The audio drive
is a Gaussian area of sigma 2 cells around the sound. The cells with a combined drive of
at least 0.05 are grouped into connected areas with 4 nearest neighbours. The area with
the most drive is selected, and vision and audio get the binding factor 1 + 0.5 times the
share of their drive inside it. Memory, body and semantic keep the workspace layer. The
phase version in the same file (`MapOscillators`) is not used by this option.

## Pre-stated gate. Written before its runs

Runs: the Gate B7 command with `--episodes 20 --dark-room-decoy-sound
--dark-room-decoy-level 0.5,3.0 --map-binding limit`, seeds 132, 133 and 134, which no run
folder names. Python 3.12.10, each run alone.

| Check | Statement |
|---|---|
| C0 | The Gate B7 command at seed 73 without the new flags gives the `metrics.csv` of `runs/f2_py312_s73` (md5 `7825766f42d3d85b032f8028c94dc1e0`). If not, VOID |
| S0 | `map_binding.json` has 4000 entries in each run, and at least one step has two or more groups. If not, VOID |
| B1 | Each run has 100 or more related and 100 or more unrelated steps (`probe_binding_selectivity.py`). If not, VOID |
| P | The selected group is on the light at more than half of the related steps (`probe_map_selection.py`), in each run |
| G1 and G2 | The residual joint alignment is higher at related steps than at unrelated steps, difference above 0 and one-sided p below 0.05, at 3 of 3 seeds |
| G3 | The binding changes the leader at 0.05 or more of the steps (`probe_binding_decisive.py`), at 3 of 3 seeds |

Option B PASSES when C0, S0 and B1 hold and P, G1 and G3 hold at 3 of 3 seeds. It FAILS
when one of P, G1 and G3 fails at 2 or more seeds. Otherwise it is UNDECIDED. The G4 of
stage 2 is left out: it reads `sync_r` of the workspace layer, which option B does not
change.

My expectation is not firm. With the light in view, the light covers about a quarter of
the map, and the sound area of a louder light lies on it at about half of the steps
(stage 2 diagnostic), so G1 can pass. The selected area may often be the light alone, and
then the audio factor stays low at related steps where the sound area does not touch it.

This is a gate on the binding of two senses. It is not evidence about consciousness, and
it moves no indicator.

## Result. VOID

B1 failed at seed 132 (54 related steps), so the gate is VOID by its own rule. The other
checks are reported for the record and decide nothing.

| Check | Seed 132 | Seed 133 | Seed 134 | Source |
|---|---|---|---|---|
| C0 | md5 `7825766f42d3d85b032f8028c94dc1e0`, identical | | | `runs/s3b_c0_s73/metrics.csv` |
| S0 | 4000 entries | 4000 entries | 4000 entries | `map_binding.json` |
| B1, related and unrelated steps | 54 and 416, FAILED | 1163 and 394 | 723 and 233 | `probe_binding_selectivity.py` |
| P, share on the light | 0.389 | 0.510 | 0.372 | `probe_map_selection.py` |
| G1, difference and p | not computed | +0.053, p 0.001 | -0.004, p 1.0 | `probe_binding_selectivity.py` |
| G3, changed leader | 0.0018 | 0.0010 | 0.0010 | `probe_binding_decisive.py` |

Each run has 4001 lines in `metrics.csv` and 21 in `episodes.csv`. The R squared of
`sync_r` on the bids stayed at 0.999, 0.9987 and 0.9996, as expected, because option B
does not change the workspace layer.

Run history. The first launch of the three runs crashed with a device error (a GPU tensor
met a CPU tensor in `combine`) and left empty folders. The fix computes the visual drive on the CPU, with a
regression test. A power cut stopped seed 134 at episode 11 of 20. It was run again from
the start. The stopped and crashed folders are kept aside and are not used.

## Diagnosis

The factors of the map reached the bids. In seed 133 the recorded ratio of bound to
modulated bid equals the map factor of the same step at 4000 of 4000 steps for vision and
at 3992 of 3992 steps with an audio bid.

The grouping was degenerate. The map held one group at 0.992, 0.951 and 0.994 of the
steps. The median group had 115, 115 and 111 of the 256 cells and held a median share of
0.984, 0.986 and 0.987 of the visual drive and 0.980, 0.980 and 0.982 of the audio drive.
So both senses got a factor near 1.5 at almost every step. The causes, and the
candidates for a new drive, are in `docs/decisions/2026_10_14_map_binding_drive.md`.

## Option B with the salience drive. Pre-stated gate, written before its runs

Owner decision of 2026-10-14. The offline comparison of the drives was VOID
(`docs/decisions/2026_10_14_map_binding_drive.md`), so the drive is chosen by the
sources. It is candidate K2 of that document. The visual drive is the contrast of a centre
Gaussian (sigma 1 cell) against a surround Gaussian 6.7 times wider, scaled by its map
maximum. A cell is stimulated when its combined drive is 0.5 or more. Grouping, selection
and factors are as in the first form.

Runs: the same command with `--map-binding salience` in place of `--map-binding limit`,
seeds 135, 136 and 137, which no run folder names. Python 3.12.10, each run alone, 20
episodes, decoy levels 0.5 and 3.0. The C0 control runs again, because the tectum now
calls the map through a new method.

The checks C0, S0, B1, P, G1 and G3 and the PASS, FAIL and UNDECIDED rule are those of
the first form, unchanged.

My expectation is not firm. In the offline comparison, which decides nothing, this drive
gave two or more groups at most steps and a group on the light at most related steps of
one seed. G1 needs the audio factor to be higher when the sound lies on the light, and
nothing measured so far shows that it will be.

## Result of the salience form. VOID

B1 failed at seed 135 (53 related steps), so the gate is VOID by its rule. This is the
second VOID of option B from B1, after seed 132 (54 related steps). The other checks
decide nothing and are reported for the record.

| Check | Seed 135 | Seed 136 | Seed 137 | Source |
|---|---|---|---|---|
| C0 | md5 `7825766f42d3d85b032f8028c94dc1e0`, identical | | | `runs/s3k2_c0_s73/metrics.csv` |
| S0, entries, and share of steps with two or more groups | 4000, 0.996 | 4000, 0.438 | 4000, 0.854 | `map_binding.json` |
| B1, related and unrelated steps | 53 and 397, FAILED | 158 and 346 | 345 and 594 | `probe_binding_selectivity.py` |
| P, share on the light | 0.585 | 0.418 | 0.104 | `probe_map_selection.py` |
| G1, difference and p | not computed | -0.010, p 0.81 | +0.084, p 0.001 | `probe_binding_selectivity.py` |
| G3, changed leader | 0.0542 | 0.0512 | 0.0571 | `probe_binding_decisive.py` |

Each run has 4001 lines in `metrics.csv` and 21 in `episodes.csv`. The mean factors were
1.199 to 1.215 for vision and 1.319 to 1.410 for audio.

Two facts for the next step, from the runs where B1 held.

1. P failed at seeds 136 and 137, the two seeds with enough steps. If B1 had held at all
   seeds, the gate would have FAILED on P.
2. G3 held at 3 of 3 seeds. This is the first form of the binding in this project that
   changed the leader at 0.05 or more of the steps.

What the selected group covered at related steps. Read-only, each cell classed by the
colour that covers more than half of its pixels.

| Seed | Group size, median cells | Floor | Light | Wall | Group holds one of the 4 centre cells |
|---|---|---|---|---|---|
| 135 | 63 | 0.403 | 0.513 | 0.071 | 0.830 |
| 136 | 64 | 0.424 | 0.448 | 0.124 | 0.696 |
| 137 | 77 | 0.603 | 0.151 | 0.238 | 0.922 |

The agent's ring is drawn on the 4 centre cells in every frame. Its lines are thin, so no
cell is mostly ring colour. At seed 137 the group held a centre cell at 0.922 of the
related steps and was mostly floor. So the group around the agent's own ring is a likely
winner. That is a reading of these counts and was not tested. The review of the drive
stated this risk before the runs, because no candidate has adaptation to a stimulus that
does not change (Knudsen 2018, as read in `docs/sensory_selection_evidence.md`).

Open for the owner. The B1 rule voided both runs of option B at one seed of three, so a
valid gate needs a test bed decision. Adaptation of the drive is the next design
question, and it needs its own review of the sources before a build.
