# Tectal motor map: the path gate PASSED; G3 FAILED, the binding's selection is not better than no selection

Date: 2026-10-15. Rules: `docs/decisions/2026_10_15_tectal_motor_map.md`, pushed before the runs
(commit da3befc).

## Verdict

G3 FAILED. With the binding's selected group as the target, the first object touched was the
louder one at 95 of 120 episodes. With all active cells and no selection it was 102 of 120. The
difference was below 0 at 3 of 3 seeds. The path gate PASSED at 3 of 3 seeds: the agent now moves
toward what its tectum selects.

## Path gate (seeds 197 to 199, 10 episodes)

| Check (rule) | 197 | 198 | 199 |
|---|---|---|---|
| Move toward the target (> 0.8 of steps with a target) | 1.0 | 1.0 | 1.0 |
| Steps at a wall (< 0.5) | 0.011 | 0.0005 | 0.0 |
| Contact with an object within 40 steps (> 0.8 of episodes) | 1.0 | 1.0 | 1.0 |
| First contact with the louder object (reported only) | 9 of 10 | 8 of 10 | 8 of 10 |

## G3 test (seeds 201 to 203, 40 episodes)

First contact with the louder object within 40 steps.

| Arm | 201 | 202 | 203 | Pooled |
|---|---|---|---|---|
| `group` (binding's selection) | 33 of 40 | 34 of 40 | 28 of 40 | 95 of 120 (0.792) |
| `active` (no selection) | 35 of 40 | 36 of 40 | 31 of 40 | 102 of 120 (0.850) |
| `audio` (sound place) | 33 of 40 | 39 of 40 | 32 of 40 | 104 of 120 (0.867) |

`group` against `active`: per seed -0.050, -0.050, -0.075; one-sided Fisher p 0.911.
`group` against `audio` (reported only): per seed 0.0, -0.125, -0.100; one-sided Fisher p 0.957.

| Arm | Share of steps in the light, 201 / 202 / 203 | Steps at a wall, 201 / 202 / 203 |
|---|---|---|
| `group` | 0.3750 / 0.3684 / 0.3954 | 0.024 / 0.019 / 0.026 |
| `active` | 0.3976 / 0.3986 / 0.3955 | 0.001 / 0.009 / 0.003 |
| `audio` | 0.3969 / 0.3950 / 0.3952 | 0.001 / 0.010 / 0.004 |

Before the motor map, the share of steps in the light was 0.0459 to 0.1294 in the gates of this
room with the same start (`docs/results/visible_decoy_gate_2026_10.md`,
`docs/results/map_binding_trim_gate_2026_10.md`), at other seeds. That is a description, not a
comparison of causes.

## Runs

Revision da3befc. Python 3.12.10. The control with all new flags off (seed 73) gave the md5
`7825766f42d3d85b032f8028c94dc1e0` for `metrics.csv`. Every run has 41 lines in `episodes.csv`
(11 in the path gate), 8001 lines in `metrics.csv` (2001) and one motor map entry per step. During
the last runs the free disk space fell below 20 GB, so the recorder skipped the heavy files (saved
weights and maps). The step records that the probe reads are complete in all 40 episodes of each run.

Source: `scripts/analysis/probe_motor_map.py`, output in `runs/_audit_followup/oct13/run_motor_g3.out`.

## Reading

- The prediction for the `active` arm was wrong. It did not average between the two objects. It
  did better than the binding's selection. A possible cause, not tested: the combined drive is
  largest at the sound place, so the active cells lie mostly on the side of the louder tone.
- The binding's selection is on the wrong object at part of the steps (P2 was 0.732 to 0.856 in the
  trimmed gate). The sound alone points the right way at almost every start (offline 1.000 at decoy
  levels 0.5 and 3.0). So in this room, the selection adds errors and no information.
- The motor map itself works. The agent leaves the walls, reaches an object in almost every episode,
  and is in the light at about 0.37 to 0.40 of the steps.

## Limits

- The orienting is a fixed reflex, with no learning.
- In this room the sound alone identifies the louder object. A room where binding is needed to know
  which object makes the sound is an open decision.
