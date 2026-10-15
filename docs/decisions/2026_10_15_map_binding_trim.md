# Binding stage 3: keep only the bright cells of the selected group

Date: 2026-10-15. Status: rules written before any run. The owner chose this step on
2026-10-15, after the gate of the visible decoy (`docs/results/visible_decoy_gate_2026_10.md`).

## Why

The visible decoy gate FAILED on P2 at 3 of 3 seeds (0.448, 0.308, 0.426). A reading after
that gate, not a check, found that the selected group lies on the louder object or at its edge
at 0.81, 0.68 and 0.81 of the counted steps, and on or at the edge of the quieter object at
only 0.04, 0.13 and 0.08. The groups had a median size of 39 to 44 cells and took the dark
floor around the edge. So the sound selected the right object, and the shape of the group
failed the check.

## The rule

Behind the default-off value `--map-binding trimmed`. The adapted form selects its group as
before. Then only the cells of the group whose image brightness is above 0.3 stay. Brightness
of a cell is the mean luminance (0.299 R + 0.587 G + 0.114 B) of the image over that cell,
taken in the tectum before the encoder, as the eye sends the image to the tectum. The binding
factors come from the trimmed group. A group with no bright cell gives the factors 1.0.

| Source | Read | Statement used here |
|---|---|---|
| Ohshiro, Angelaki and DeAngelis, Nat. Neurosci. 14, 775 (2011) | Full text, equations | Input out of register adds little to the drive of a neuron. So a sound raises the response only where the eye also drives it |

Limit. Brightness is the visual signal because every object of the dark room is bright. A dark
object would never stay in the group. Both objects are bright, so the trim makes "on an object"
nearly true by its rule. It does not make "on the louder object" true, because the two objects
look the same. P2 still tests whether the sound selected the right object.

## Diagnosis before the build: PASSED

Read-only, on the recorded visible decoy gate runs. Rule, written before the run: the trimmed
group is on the louder object at more than 0.5 of the counted steps, at 3 of 3 seeds.

| Seed | On the louder object, before | After the trim | On the quieter object, after the trim | Trimmed group empty |
|---|---|---|---|---|
| 181 | 0.448 | 0.787 | 0.041 | 0.146 |
| 182 | 0.308 | 0.640 | 0.139 | 0.174 |
| 183 | 0.426 | 0.784 | 0.071 | 0.090 |

The replay cannot show the closed loop. In a live run the trimmed group also changes the
binding factors, so the agent can act differently.

## Gate

Seeds 187, 188 and 189, each run once with `--map-binding trimmed` and once without map
binding (`--record-episodes none`). The command of the visible decoy gate: the Gate B7 flags
with `--episodes 40 --dark-room-decoy-sound --dark-room-decoy-level 0.5,3.0
--dark-room-decoy-visible --dark-room-start-between`. One run at a time. Completion is checked
by the row counts of the csv files. Before the runs, a control with all new flags off must give
the md5 `7825766f42d3d85b032f8028c94dc1e0` for `metrics.csv` of seed 73.

| Check | Statement |
|---|---|
| P2 | At steps with both objects in view, the trimmed group is on the object with the louder tone at more than half of the steps, at 3 of 3 seeds |
| G1 | The joint alignment of vision and audio is higher at related than at unrelated steps, p < 0.05, at 3 of 3 seeds |
| G3 | The binding changes the leader at 0.05 or more of the steps, at 3 of 3 seeds |
| Task | The share of steps in the light (reward 1.0), binding on against binding off, at each seed. A hypothesis only |

PASS when P2, G1 and G3 all hold. FAIL when one fails at 2 or more seeds. Otherwise UNDECIDED.
