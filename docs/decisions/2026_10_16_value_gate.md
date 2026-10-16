# The value gate: the agent learns from reward which heard object to follow

Date: 2026-10-16. Status: rules written before any training run. The owner chose this step (option
2, learning) and this design on 2026-10-16.

## Why

AE-1 of the indicator rubric asks for "learning from feedback and selecting outputs so as to pursue
goals"; the rubric names the learning clause as the harder failure. The tectum now binds a seen
object to its tone by timing, and its motor map turns the agent toward the bound object
(`docs/results/rhythm_binding_gate_2026_10.md`). But the agent goes to the object with the louder
tone, and the light (food) is the louder one in only half of the episodes. Nothing taught the agent
which object feeds it. The policy's movement network cannot learn it, because its relay layer is in
no loss term (`docs/results/action_core_relay_defect_2026_10.md`).

## Sources

| Source | Read | Statement used here |
|---|---|---|
| Kim and Hikosaka, Neuron 79, 1001 (2013) | Full text, summary | Monkeys learn the value of objects from reward. The caudate head tracks fast-changing values, within 30 to 40 trials; the tail keeps stable values. Inactivation removes the value bias |
| Yasuda and Hikosaka, J. Neurophysiol. (2014); Hikosaka, Takikawa and Kawagoe, Physiol. Rev. 80, 953 (2000) | Summaries, abstract | The value path runs from the caudate through the substantia nigra to the superior colliculus; high value releases the nigra's hold on gaze toward that object |
| Anderson, Laurent and Yantis, PNAS 108, 10367 (2011) | Abstract | A stimulus paired with reward captures attention beyond its salience |
| Schultz, Dayan and Montague, Science 275, 1593 (1997) | Abstract | Dopamine signals the reward prediction error that drives value learning |
| Redgrave, Prescott and Gurney, Neuroscience 89, 1009 (1999); Grillner and Robertson, Curr. Biol. 26, R1088 (2016) | Abstracts | The basal ganglia are the selection switch of vertebrates, conserved from lamprey |

## Designs compared

| Design | What learns | Can it pick the food object here? |
|---|---|---|
| Repair the policy network (relay in the loss, a real actor-critic) | Movement from the workspace broadcast | Only if the broadcast carries which object feeds; in June, learners on the broadcast stayed far below a pixel network |
| A learned go/no-go gate on the one bound target | Whether to follow the louder object | No: the louder object feeds in half of the episodes |
| **A learned value of each heard tone that decides which bound object the tectum follows (chosen)** | A value per cochlear band, by reward prediction error | Yes |

The relay defect stays recorded and open; this test does not use the policy's movement.

## The mechanism (`models/core/value_gate.py`, `models/core/rhythm_binding.py`)

| Part | Rule, set before any run |
|---|---|
| Candidates | The binder binds the two largest local peaks of the cochlear band profile (one band for each tone) to their own groups, by the coherence of the gate of 2026-10-15 (threshold 0.55, wait 12 steps, body removed) |
| Salience | The slow-mean envelope of the band over the largest |
| Choice | Once per episode, at the first step with groups: softmax over salience + 2 × value, temperature 0.25, from its own generator. The choice is kept to the episode end |
| Outcome | Food (a reward above 0.5) within 60 steps after the choice gives 1, otherwise 0. In recorded runs the reward is 0.998 to 1.002 in the light and -0.013 to -0.005 outside |
| Learning | error = outcome − value; value += 0.3 × error, once per episode. A step without sound starts the next episode and closes an open choice with 0 |
| Movement | The motor map follows the group of the chosen tone. No target means no movement (`--tectal-orienting-idle still`) |

Flags: `--tectal-orienting value` (needs `--map-binding rhythm` or `rhythm-scrambled`, which then bind
two bands) and `--value-learning-rate` (default 0.3; 0 is the frozen arm). Records: `value_gate.json`
(per episode the choice, priorities, value before and after, outcome, error).

## Checks before any training run

**Abstract model** (no room): 2000 sequences, the quieter tone's salience 0.34 to 0.48 (measured in
the cochlea at the start), the chosen object reached at 0.958 (as in the rhythm gate). Learning arm
light first 0.871 in episodes 1 to 20 and 0.954 in 21 to 40; frozen arm 0.500. Learning is fast
(about 5 episodes), so no check compares early with late episodes; the learning and frozen arms
differ only in the learning rate.

**Offline diagnosis 1, FAILED.** The real room, the real cochlea, the motor map, no policy; rule
written first. The first version bound the two LOUDEST bands. Those were always the louder tone and
its neighbour band (18 and 17 at level 0.5; 23 and 22 at level 3.0), so the quieter tone was never a
choice. The value of the light's tone was learned (1.0), but behaviour did not change: light first
0.55, 0.55, 0.45 on held-out sequences 504 to 506. The fix, before any training run: the two largest
local peaks of the band profile (18 and 23 in 6 of 6 checks).

**Offline diagnosis 2, PASSED.** Same rule, fresh held-out sequences 507 to 509.

| Sequence | Learning arm, light first in episodes 21 to 40 (rule ≥ 0.8) | Decoy louder (rule ≥ 0.7) | Frozen arm (rule ≤ 0.65, below learning) |
|---|---|---|---|
| 507 | 0.95 | 0.90 | 0.60 |
| 508 | 0.95 | 0.90 | 0.50 |
| 509 | 0.90 | 0.80 | 0.45 |

Smoke run in the training loop (3 episodes per arm): one choice and one update per episode in every
arm.

## Gate

Seeds 601, 602 and 603, 40 episodes, one run at a time. The Gate B7 flags with
`--dark-room-decoy-sound --dark-room-decoy-level 0.5,3.0 --dark-room-decoy-visible
--dark-room-start-between --dark-room-pulse --dark-room-sound-direction off --tectal-orienting value
--tectal-orienting-idle still`, and each arm's flags:

| Arm | Flags | Prediction |
|---|---|---|
| learn | `--map-binding rhythm` | light first ≥ 0.8 in episodes 21 to 40 |
| frozen | `--map-binding rhythm --value-learning-rate 0` | about 0.5 |
| scrambled | `--map-binding rhythm-scrambled` | about 0.5 or lower |

Light first: within the first 60 steps, the first object closer than 60 room pixels is the light; no
contact counts as not. Read by `scripts/analysis/probe_value_gate.py`.

| Check | Rule, at 3 of 3 seeds |
|---|---|
| L1 learns from feedback | learn arm, episodes 21 to 40: light first ≥ 0.8; above the frozen arm, pooled one-sided Fisher p < 0.05, difference > 0 at each seed |
| L2 value over salience | learn arm, episodes 21 to 40 with the decoy louder: light first ≥ 0.7 |
| L3 the learning needs the binding | learn arm above the scrambled arm in episodes 21 to 40, pooled one-sided Fisher p < 0.05, difference > 0 at each seed |

PASS when L1 to L3 hold. FAIL when one fails at 2 or more seeds. Reported only: episodes 1 to 20,
the learned values, the share of steps in the light, the workspace leader changes. A control with
all new flags off must give the md5 `7825766f42d3d85b032f8028c94dc1e0` for `metrics.csv` of seed 73.

## Limits, stated before the runs

- The light is told apart by its pitch, and the rhythms are steady (binding failed with irregular
  rhythms, `docs/results/rhythm_binding_harder_rooms_2026_10.md`).
- One value per tone, one learning rate (a fast learner like the caudate head); no stable second
  memory. The "competing goals" clause of AE-1 (for example a reversal of which object feeds) is a
  later test.
- The orienting stays a reflex; the gate decides which target it gets.
