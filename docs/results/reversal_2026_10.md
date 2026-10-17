# The reversal: the gate PASSED, R1 to R3 at 3 of 3 seeds

Date: 2026-10-17. Rules: `docs/decisions/2026_10_16_reversal.md`, pushed before the runs (commit
ea3eabd).

## A defect in the room flag, found after the runs

FAILED: `--dark-room-food-switch 30` moved the food at training episode index 29, not at 30 as the
rules state. The flag counted every reset of the room, and `scripts/training/train_rlhf.py` resets
the room once before its first episode (`seed_environment`). The second count of the runs found it:
in all three reversal runs the first episode with the food at the decoy is index 29. The offline
diagnosis made no such reset, so there the food moved at index 30.

What this changes: in episode index 29 the agent chose its target before any reward of the new rule
came, so R1 still measured behaviour learned before the move. The R2 window began one episode later
after the move than planned. The checks are reported below as written, and again with every window
moved by one episode (after the data, reported only). Both readings pass.

The fix is in the same pull request: the room now counts an episode at its first step, so a reset
without a step does not count. A test pins it with `seed_environment`. With the fixed flag the gate
runs are the runs of `--dark-room-food-switch 29`. On seed 701 the fixed flag at 29 gave the files
of the gate run byte for byte (md5 `2752358f446218ac351e78cf65dfefca` for `metrics.csv`).

## Verdict

PASS. At seeds 701, 702 and 703 the agent first learned that the light gives food. When the food
moved to the decoy, it changed its choice from reward alone. In episodes 46 to 60 it reached the
decoy first in 43 of 45 episodes, against 3 of 45 in the arm where the food stayed at the light.

## Checks, as written

First object: within the first 60 steps, the first object closer than 60 room pixels.

| Check | Rule | 701 | 702 | 703 |
|---|---|---|---|---|
| R1 learns before the move | reversal arm, episodes 21 to 30, light first ≥ 0.8 | 0.90 | 0.90 | 1.00 |
| R2 reverses | reversal arm, episodes 46 to 60, decoy first ≥ 0.8 | 1.00 (15 of 15) | 0.93 (14 of 15) | 0.93 (14 of 15) |
| R3 the change follows the food | decoy first in episodes 46 to 60, reversal arm above the stay arm | +0.87 | +0.93 | +0.87 |

Pooled R3: 43 of 45 against 3 of 45, one-sided Fisher p 1.4e-19. A second count with separate code,
read directly from the step records, gave the same shares for all six runs. The abstract model of
the decision document predicted 0.943 for the reversal arm in episodes 46 to 60; the pooled value
is 0.956.

## Checks with every window moved by one episode (after the data, reported only)

| Check | Window | 701 | 702 | 703 |
|---|---|---|---|---|
| R1 | light first, episodes 20 to 29 | 0.90 | 0.90 | 1.00 |
| R2 | decoy first, episodes 45 to 59 | 1.00 | 0.93 | 1.00 |
| R3 | reversal arm minus stay arm, decoy first, episodes 45 to 59 | +0.87 | +0.93 | +0.93 |

Pooled 0.978 against 0.067, one-sided Fisher p 6.7e-21.

## Reported only

| Measure | 701 | 702 | 703 |
|---|---|---|---|
| Light first in the 5 episodes after the move | 1.00 | 0.80 | 0.80 |
| Episodes after the move until the first choice of the decoy's tone | 7 | 3 | 3 |
| Episodes after the move until 4 of 5 reach the food first | 10 | 5 | 7 |
| Learned values at the end, reversal arm: light tone / decoy tone | 0.058 / 0.999 | 0.160 / 1.000 | 0.168 / 1.000 |
| Learned values at the end, stay arm: light tone / decoy tone | 0.975 / 0.000 | 1.000 / 0.000 | 1.000 / 0.000 |
| Share of steps with food (reward 1.0): reversal arm / stay arm | 0.6221 / 0.7222 | 0.6596 / 0.7213 | 0.6475 / 0.7095 |

- The abstract model put the episodes until 4 of 5 at a median of 6 (95th percentile 10); the runs
  gave 5, 7 and 10.
- The workspace leader changed at 0.0000 of the steps in all three reversal runs (vision and audio
  factor 1.466 to 1.467, memory 1.494 to 1.495, body 1.492 to 1.496). The probe's own label
  "UNDECIDED" refers to its earlier ten-seed hypothesis, not to this gate.

## Runs

Revision ea3eabd, Python 3.12.10. The control with all new flags off (seed 73) gave the md5
`7825766f42d3d85b032f8028c94dc1e0` for `metrics.csv`. Each of the six runs has 61 lines in
`episodes.csv`, 12001 lines in `metrics.csv`, 60 recorded episodes and 60 closed choices in
`value_gate.json`. Output: `runs/_audit_followup/oct13/run_reversal.out`; the second count
`reversal_second_count.out`; the moved windows `reversal_aligned.out`.

After the fix (revision 961dc1c) the control gave the same md5, and seed 701 with
`--dark-room-food-switch 29` gave the same `metrics.csv` (`2752358f446218ac351e78cf65dfefca`),
`episodes.csv` (`0e21a61305093abee8c2feda3050c359`) and `value_gate.json`
(`ba464e91176fc8036aead3c4cdd6fa5e`) as the gate run. Output:
`runs/_audit_followup/oct13/run_reversal_repro.out`.

## What this shows, and what it does not

- The agent changed a learned choice when the reward changed, from feedback alone, within 5 to 10
  episodes after the move. Where the food stayed at the light, it kept going to the light. This is
  the flexibility that the rubric names as weak for AE-1, measured in one room.
- The mechanism is the unchanged value gate: one fast value per tone, learned by prediction error.
  No new part was needed, as the abstract model predicted.
- Not tested: competing goals (two needs at the same time), serial reversals, irregular rhythms.
  The orienting stays a reflex, and the policy's movement network is not used. The AE-1 score stays
  the owner's decision.
