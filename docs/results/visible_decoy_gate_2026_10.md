# Binding stage 3 with a visible decoy: gate FAILED on P2

Date: 2026-10-15. Rules: `docs/decisions/2026_10_15_visible_decoy.md`, pushed before the runs.

## Verdict

FAILED. P2 failed at 3 of 3 seeds. G1 held at 2 of 3 seeds. G3 held at 3 of 3 seeds. The
task share did not change with binding.

| Check | 181 | 182 | 183 | Holds at |
|---|---|---|---|---|
| P2, map on the louder object (> 0.5) | 0.448 | 0.308 | 0.426 | 0 of 3 |
| G1, alignment related minus unrelated (p < 0.05) | +0.02626, p 0.0010 | +0.00042, p 0.4675 | +0.02979, p 0.0010 | 2 of 3 |
| G3, leader changed (>= 0.05) | 0.0939 | 0.0520 | 0.0682 | 3 of 3 |
| Task, share of steps in the light, binding on / off | 0.1116 / 0.1086 | 0.0788 / 0.0816 | 0.1288 / 0.1294 | no change |

Steps with both objects in view, at the runs with binding on: 978, 367 and 491 (related 480,
194, 224; unrelated 498, 173, 267).

## Runs

Revision b1d0f24 (main with PR 51, plus the visible decoy). Python 3.12.10. The Gate B7 flags
with `--episodes 40 --dark-room-decoy-sound --dark-room-decoy-level 0.5,3.0
--dark-room-decoy-visible --dark-room-start-between`, once with `--map-binding adapted` and
once without it (`--record-episodes none`). Seeds 181, 182 and 183, one run at a time. Each
run has 41 lines in `episodes.csv` and 8001 lines in `metrics.csv`.

Sources: `probe_louder_object.py` (P2), `probe_binding_selectivity.py` (G1),
`probe_binding_decisive.py` (G3), and the `reward` column of `metrics.csv` (task: reward 1.0
only in the light).

## A reading after the gate, not a check

This table was computed after the verdict. It is not a pre-stated check, and no verdict rests
on it. It splits the counted steps of P2 by where the selected group lies.

| Seed | On the louder object | On the quieter object | On neither | On the light, whatever the tone |
|---|---|---|---|---|
| 181 | 0.448 | 0.027 | 0.526 | 0.181 |
| 182 | 0.308 | 0.114 | 0.578 | 0.158 |
| 183 | 0.426 | 0.057 | 0.517 | 0.177 |

When the group lies on an object, it lies on the louder one 3 to 17 times as often as on the
quieter one. So the tone picks the right object. P2 failed because the group lies on neither
object at about half of the steps. The earlier gates showed where such a group lies, at the
edge of the light (`docs/results/binding_map_stage3_2026_10.md`). This reading is a
hypothesis for a next test, not a result.

## Limits

- The agent starts each episode between the two objects. Without that start it stays at the
  walls (second diagnosis), so the test covers mostly the first steps of each episode.
- The task check is one comparison per seed and stays a hypothesis.
- G1 reads all steps with the light in view, as in stage 2, not only the steps with both
  objects in view.

## Where the group lies when it is on neither object (reading after the gate)

Computed after the gate on the same steps, read-only. Descriptive only, no rule. A group
"touches" an object when one of its cells is an 8-neighbour of a cell of that object.

| Seed | Steps on neither object | Touches only the louder | Touches only the quieter | Touches both | Touches none | Median group size, cells |
|---|---|---|---|---|---|---|
| 181 | 514 | 0.681 | 0.031 | 0.049 | 0.239 | 39 |
| 182 | 212 | 0.646 | 0.033 | 0.057 | 0.264 | 44 |
| 183 | 254 | 0.748 | 0.039 | 0.031 | 0.181 | 40.5 |

So most groups on neither object lie at the edge of the louder object, with more than half of
their cells off it. With the earlier table, the group is on or at the edge of the louder object
at 0.81, 0.68 and 0.81 of the counted steps, and on or at the edge of the quieter object only
at 0.04, 0.13 and 0.08 (products of the two tables, seeds 181, 182, 183). This supports, as a
hypothesis, that the sound selects the right object and that the remaining fault is the shape
of the group, which takes the edge and the dark around it.
