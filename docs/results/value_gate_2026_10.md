# The value gate: the gate PASSED, L1 to L3 at 3 of 3 seeds

Date: 2026-10-16. Rules: `docs/decisions/2026_10_16_value_gate.md`, pushed before the runs
(commit 6d8143f).

## Verdict

PASS. At seeds 601, 602 and 603 the agent learned from reward which heard tone belongs to food, and
used that value to choose which bound object to follow. In episodes 21 to 40 it reached the light
first in 56 of 60 episodes, against 28 of 60 with the learning rate at 0 and 34 of 60 with the
sound's rhythm destroyed. When the decoy was the louder object, it still went to the light in 28 of
30 of those episodes.

## Checks

Light first: within the first 60 steps, the first object closer than 60 room pixels is the light.

| Check | Rule | 601 | 602 | 603 |
|---|---|---|---|---|
| L1 learns from feedback | learn arm, episodes 21 to 40, ≥ 0.8 and above frozen | 0.90 (frozen 0.55) | 1.00 (frozen 0.50) | 0.90 (frozen 0.35) |
| L2 value over salience | learn arm, episodes 21 to 40 with the decoy louder, ≥ 0.7 | 0.8 (8 of 10) | 1.0 (10 of 10) | 1.0 (10 of 10) |
| L3 the learning needs the binding | learn arm above the scrambled arm, episodes 21 to 40 | 0.90 against 0.65 | 1.00 against 0.60 | 0.90 against 0.45 |

Learn against frozen: per seed +0.35, +0.50, +0.55; pooled 0.933 against 0.467, one-sided Fisher
p 9.9e-09. Learn against scrambled: per seed +0.25, +0.40, +0.45; pooled 0.933 against 0.567,
p 2.2e-06. A second count with separate code, read directly from the step records, gave the same
numbers for all nine runs.

## Reported only

| Measure | 601 | 602 | 603 |
|---|---|---|---|
| Learned value of the light's tone (band 18) / the decoy's tone (band 23) | 1.000 / 0.000 | 1.000 / 0.000 | 1.000 / 0.000 |
| Light first in episodes 1 to 20: learn / frozen / scrambled | 0.80 / 0.50 / 0.60 | 0.80 / 0.45 / 0.60 | 0.95 / 0.50 / 0.60 |
| Frozen arm with the decoy louder, episodes 21 to 40 | 0.1 | 0.2 | 0.1 |
| Share of steps in the light: learn / frozen / scrambled | 0.690 / 0.392 / 0.480 | 0.671 / 0.336 / 0.441 | 0.724 / 0.354 / 0.404 |

- The learning is fast, as the abstract model predicted (about 5 episodes): episodes 1 to 20 are
  already 0.80 to 0.95.
- With learning the agent spent 1.8 to 2.0 times as many steps in the light as the frozen agent.
- The workspace leader changed at 0.0000 of the steps at all three seeds (vision and audio factor
  1.466 to 1.467, memory and body 1.495 to 1.498). The learning acts through the tectum's selection
  and motor map, not through the workspace.

## Runs

Revision 6d8143f, Python 3.12.10. The control with all new flags off (seed 73) gave the md5
`7825766f42d3d85b032f8028c94dc1e0` for `metrics.csv`. Each of the nine runs has 41 lines in
`episodes.csv`, 8001 lines in `metrics.csv`, 8000 entries in `map_binding.json` and in
`tectal_orienting.json`, and 40 closed choices in `value_gate.json`. Output:
`runs/_audit_followup/oct13/run_value_gate.out`.

After the gate, the probe's labels were corrected: it named the rows of the frozen and scrambled
arms after the learning run of the same seed. The numbers did not change; the probe now labels rows
by seed.

## What this shows, and what it does not

- The agent learns from feedback which of two identical-looking objects feeds it, by the pitch of
  the tone that the binding joins to each object, and it selects its movement to pursue that goal.
  This is the "learning from feedback and selecting outputs so as to pursue goals" clause of AE-1,
  measured in one room.
- Not tested: "flexible responsiveness to competing goals" (for example a reversal of which object
  feeds, or a second need). AE-1 stays PARTIAL until the owner decides.
- The orienting is a fixed reflex; what is learned is which target it gets. The rhythms are steady.
  The policy's movement network (the relay defect) is not used.
