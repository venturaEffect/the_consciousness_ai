# Binding stage 3: orientation test FAILED, binding barely moves the agent

Date: 2026-10-15. Rule: `docs/decisions/2026_10_15_orientation_test.md`, pushed before the
runs (commit 7f7480d).

## Verdict

FAILED. With binding the agent was oriented to the louder object at the same number of
episodes as without binding, at every seed.

| Seed | Oriented, binding on | Oriented, binding off | Difference |
|---|---|---|---|
| 194 | 19 of 40 | 19 of 40 | 0.0 |
| 195 | 20 of 40 | 20 of 40 | 0.0 |
| 196 | 17 of 40 | 17 of 40 | 0.0 |
| Pooled | 56 of 120 (0.467) | 56 of 120 (0.467) | one-sided Fisher p 0.551 |

Share of steps in the light, binding on / off: 0.0583 / 0.0580, 0.1151 / 0.1164, 0.0930 / 0.0932.

## Runs

Revision 7f7480d. Python 3.12.10. The control with all new flags off (seed 73) gave the md5
`7825766f42d3d85b032f8028c94dc1e0` for `metrics.csv`. Each run has 41 lines in `episodes.csv`
and 8001 lines in `metrics.csv`.

## Why the counts are equal (checked after the verdict)

The equal counts are not a copy of one run. The two arms differ from the first step on.

| Seed | Episodes with a different action | Steps with a different winner, of 8000 | Same outcome | Agent place gap after 10 steps, median / largest | Mean action gap |
|---|---|---|---|---|---|
| 194 | 40 of 40 | 4240 | 40 of 40 | 1.46 / 30.29 px | 0.0321 |
| 195 | 40 of 40 | 4483 | 40 of 40 | 0.96 / 10.80 px | 0.0291 |
| 196 | 40 of 40 | 4141 | 40 of 40 | 1.17 / 10.15 px | 0.0352 |

The action range is -1 to 1 on each axis, and a step moves the agent up to 10 room pixels.
So the binding changes the winner of the workspace at about half of the steps, but it changes
the action by about 3 percent of its range, and the agent ends at almost the same place.
Without binding the agent was oriented to the louder object at 0.467 of the episodes, below
half. So the agent does not turn to the louder tone, with or without binding.

## Reading

The map can select the object with the louder tone (P2 of the trimmed gate, 3 of 3 seeds).
This test shows that the selection does not reach the behaviour. The agent's actions depend
very little on the workspace winner, and the agent has not learned to approach a tone. This
matches the known separate finding that the policy cannot use information that is present in
its inputs (the competence wall in the project record). Binding cannot show "used" in
behaviour until the agent acts on what it perceives.

## Limits

- One test length (10 steps) and one form of binding (trimmed).
- 40 episodes per run.
