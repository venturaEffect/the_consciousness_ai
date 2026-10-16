# FAILED: the movement network ignores its input, because its last layer is never trained

Date: 2026-10-15. Read-only measurements on recorded runs. No code changes with this document.

## Verdict

The default policy (`--policy gonogo`, `ActionSelectionCore`) gives almost the same action at
every step of every episode. The agent drifts in one fixed direction for each seed and stays at a
wall. Nothing it perceives changes this. The cause is a defect in the code: the last layer of the
network, `BasalGanglia.thalamic_relay`, is in no loss term, so it keeps its random start values.

## Measurements

**1. The output does not depend on the input.** Seed 194 without binding (`runs/or_off_s194`),
episode 10, replayed with the weights saved for that episode and the recorded broadcasts.

| Quantity | Value |
|---|---|
| Action mean over 200 steps | (-0.188, 0.327) |
| Spread of the action mean over the steps | (0.0017, 0.0056) |
| Same replay with the broadcasts in a random order | (-0.188, 0.327), spread (0.0017, 0.0048) |
| Action mean for an input of all zeros | (-0.192, 0.329) |
| Recorded action minus the replayed mean (the exploration noise) | spread (0.108, 0.115) |
| Variation of the input: mean cosine of the broadcast to its mean | 0.477 |
| Go, No-Go and STN outputs, mean | go (0.086, -0.179), no-go (0.73, 0.71), STN 0.067 |

The other parts of the network learn: the prefrontal part changed by 0.885 and the basal ganglia
part by 0.802 of their norm from episode 0 to episode 39.

**2. The relay layer never changes.** `scripts/training/train_rlhf.py` builds the policy in
`init_components`. In `update_policy` (`models/self_model/action_selection_core.py`), the loss
uses the go, no-go and STN outputs and the value. It does not use `thalamic_relay`. A scan of every
recorded run with saved weights in `runs/` compared the first and the last saved episode.

| Runs scanned | Relay weights changed | Largest change of the go pathway |
|---|---|---|
| 172 (all gate runs from Gate B to the orientation test, DMTS runs included) | 0 of 172 (largest change exactly 0.0) | 0.002 to 0.274 |

**3. The relay bias sets the drift.** The go signal is small and the no-go signal is large, so the
product that enters the relay is small, and the action is close to tanh(relay bias).

| Run | tanh(relay bias) | Measured mean action |
|---|---|---|
| `or_off_s194` | (-0.198, 0.351) | (-0.18 to -0.27, 0.31 to 0.45) over 6 episodes |
| `or_off_s195` | (-0.241, 0.210) | (-0.20 to -0.23, 0.20 to 0.23) over 6 episodes |

At these seeds the agent touched a wall at 0.71 to 0.93 of the steps of the episodes read.

**4. The same in DMTS.** Added 2026-10-16. Two later DMTS runs at seed 42 saved their policy
weights. `fixcheck_dmts_s42` uses the plain defaults (5 episodes of 200 steps). `f3_py312_dmts`
adds `--rssm-latent-mode continuous --capsule-workspace-source all_levels` (10 episodes of 500
steps). The last saved episode of each was replayed as in measurement 1. Output:
`runs/_audit_followup/oct13/dmts_relay_replay.out`.

| Run, episode | Action mean | Spread over the steps | Input of zeros | tanh(relay bias) |
|---|---|---|---|---|
| `fixcheck_dmts_s42`, 4 (200 steps) | (0.417, 0.279, -0.086, -0.406, -0.232) | 0.0001 or less | (0.420, 0.276, -0.086, -0.404, -0.233) | (0.416, 0.277, -0.081, -0.402, -0.237) |
| `f3_py312_dmts`, 9 (500 steps) | (0.245, -0.053, 0.107, 0.226, -0.404) | 0.0001 or less | (0.245, -0.052, 0.106, 0.222, -0.406) | (0.246, -0.053, 0.104, 0.222, -0.404) |

The largest value of the mean is choice 0 at every step of both episodes. Over each whole run,
the recorded choices come in the order of tanh(relay bias): 0 at 0.675, 1 at 0.302 and 2 at
0.021 in `fixcheck_dmts_s42`; 0 at 0.413, 3 at 0.370, 2 at 0.172 and 1 at 0.044 in
`f3_py312_dmts`. The exploration noise spreads them. In these runs the choice did not depend on
what the agent saw.

The trained DMTS runs of June and July 2026 (`docs/results/collapse_locus_trained_2026_06_21.md`,
`docs/results/b1_continuous_latent_2026_07.md`) used the same default policy for 100 episodes
and saved no policy weights. For them the same result follows from the code. It was not
measured on them.

## Results that read behaviour from this drift

The verdicts stand as written. But each of these checks read a behaviour that came from the fixed
output of the policy, not from perception. In the dark room that is the drift plus the start
position; in DMTS it is the same order of choices at every step.

| Document | Check |
|---|---|
| `docs/results/dark_room_senses_2026_09.md` | The task criterion of Gates B2 to B8 (audio share against the share of steps with the light out of view, per episode) |
| `docs/results/visible_decoy_gate_2026_10.md` | Task, share of steps in the light, binding on against off |
| `docs/results/map_binding_trim_gate_2026_10.md` | The same task check |
| `docs/results/orientation_test_2026_10.md` (PR 56) | The orientation test |
| `docs/consciousness_indicators_butlin.md`, AE-1 | The note that DMTS task reward is flat |
| `docs/results/collapse_locus_trained_2026_06_21.md`, `docs/results/b1_continuous_latent_2026_07.md` | The flat DMTS task reward of the trained runs (measurement 4) |

The June 2026 diagnosis (`docs/results/agent_competence_diagnosis_2026_06_01.md`) found that a
plain actor-critic on the same broadcast tied with this policy. So the relay defect is not the only
reason for weak control on the broadcast. It is the reason that the default policy cannot react
at all.

## What follows

A separate decision document (next PR) adds a motor map to the
tectum, so that the selected place drives an orienting movement. A repair of the learned policy
(relay in the loss, a real actor-critic) is an open decision for the owner.
