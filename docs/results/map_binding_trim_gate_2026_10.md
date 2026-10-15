# Binding stage 3, trimmed form: gate FAILED on G3

Date: 2026-10-15. Rules: `docs/decisions/2026_10_15_map_binding_trim.md`, pushed before the
runs (commit dfef70a).

## Verdict

FAILED. G3 failed at 3 of 3 seeds. P2 held at 3 of 3 seeds, for the first time in this test
bed. G1 held at 2 of 3 seeds. The task share did not change with binding.

| Check | 187 | 188 | 189 | Holds at |
|---|---|---|---|---|
| P2, trimmed group on the louder object (> 0.5) | 0.732 | 0.856 | 0.778 | 3 of 3 |
| G1, alignment related minus unrelated (p < 0.05) | -0.00576, p 0.7502 | +0.06144, p 0.0010 | +0.09276, p 0.0010 | 2 of 3 |
| G3, leader changed (>= 0.05) | 0.0479 | 0.0304 | 0.0355 | 0 of 3 |
| Task, share of steps in the light, binding on / off | 0.0551 / 0.0545 | 0.0803 / 0.0800 | 0.0459 / 0.0461 | no change |

Steps with both objects in view, at the runs with binding on: 157, 355 and 243 (related 66,
186, 116; unrelated 91, 169, 127).

## Runs

Revision fbe3bbf. Python 3.12.10. The control with all new flags off (seed 73, 10 episodes,
`--record-episodes none`) gave the md5 `7825766f42d3d85b032f8028c94dc1e0` for `metrics.csv`.
The gate command is in the decision document. Each gate run has 41 lines in `episodes.csv`
and 8001 lines in `metrics.csv`.

Sources: `probe_louder_object.py` (P2), `probe_binding_selectivity.py` (G1),
`probe_binding_decisive.py` (G3, mean factors), the `reward` column of `metrics.csv` (task).

## What the numbers show

The trim moved the selected group onto the right object. It also made the binding weaker. The
binding factor of a sense is 1 + 0.5 times the share of its drive inside the group, so a
smaller group gives smaller factors.

| Form, seeds | Mean factor vision | Mean factor audio | Leader changed |
|---|---|---|---|
| Adapted, visible decoy gate, 181 to 183 | 1.069 to 1.112 | 1.313 to 1.343 | 0.0520 to 0.0939 |
| Trimmed, 187 to 189 | 1.026 to 1.031 | 1.095 to 1.113 | 0.0304 to 0.0479 |

The two gates used different seeds, so this table is a description, not a comparison of
causes.

## Limits

- The agent starts each episode between the two objects.
- The task check is one comparison per seed and stays a hypothesis.
- G1 reads all steps with the light in view, as in stage 2.
