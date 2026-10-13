# Why vision takes nearly every step with `top_winner`: the reentrant cycles

Date: 2026-10-13. Read-only probe on existing session records. No new run.

## Verdict, failure first

**The mechanism reading as I pre-stated it is NOT SUPPORTED. Prediction D1 is false at 4
of 6 seed pairs.** Predictions D2 and D3 are true at 6 of 6.

| Prediction | Result |
|---|---|
| D1. With `legacy`, the leader's bid rises more than the other bid, for both leaders | **FALSE at 4 of 6 pairs** (seeds 73, 74, 84, 85). With audio in the lead, the vision bid rose more than the audio bid there |
| D2. With `top_winner`, the other bid rises more than the leader's bid, for both leaders | TRUE at 6 of 6 |
| D3. More audio leads flip to vision with `top_winner` than with `legacy` | TRUE at 6 of 6 |

So my account of the default merge was wrong in part. The part about `top_winner` held.

Probe: `scripts/analysis/probe_reentrant_boost.py`, 13 tests in
`tests/test_reentrant_boost_probe.py`. The predictions and the verdict rule were
committed and pushed (`aecc071`) before the probe ran on any record. The docstring
states what an exploration had shown before that.

## What was measured

The bids before the reentrant cycles (`raw_bids`) and after them
(`settle.final_bids`), at every recorded step except step 0 of an episode. The leader is
the sense with the higher bid before the cycles. Runs with the default merge and with
`top_winner` at the same seed.

Mean change of a bid in the cycles, and the share of leads that flip.

| Seed | Merge | Vision leads: change of vision / of audio | flips | Audio leads: change of audio / of vision | flips |
|---|---|---|---|---|---|
| 73 | legacy | +0.174 / +0.010 | 0.032 | +0.058 / +0.065 | 0.082 |
| 73 | top_winner | +0.082 / +0.095 | 0.035 | +0.060 / **+0.369** | **0.586** |
| 74 | legacy | +0.116 / +0.022 | 0.026 | +0.048 / +0.091 | 0.117 |
| 74 | top_winner | +0.072 / +0.097 | 0.024 | +0.055 / **+0.372** | **0.532** |
| 75 | legacy | +0.160 / +0.022 | 0.036 | +0.066 / +0.032 | 0.044 |
| 75 | top_winner | +0.047 / +0.096 | 0.034 | +0.059 / **+0.392** | **0.559** |
| 83 | legacy | +0.152 / +0.017 | 0.036 | +0.067 / +0.057 | 0.085 |
| 83 | top_winner | +0.050 / +0.100 | 0.032 | +0.064 / **+0.359** | **0.615** |
| 84 | legacy | +0.194 / +0.014 | 0.037 | +0.038 / +0.087 | 0.178 |
| 84 | top_winner | +0.034 / +0.097 | 0.130 | +0.065 / **+0.384** | **0.562** |
| 85 | legacy | +0.167 / +0.010 | 0.044 | +0.048 / +0.159 | 0.210 |
| 85 | top_winner | +0.065 / +0.102 | 0.058 | +0.071 / **+0.401** | **0.674** |

Seeds 73 to 75 have no learned valence (`runs/gate_b7_s*` against
`runs/carrier_t1_s*`). Seeds 83 to 85 have the learned valence with the saturating
boost (`runs/gate_b8_s*` against `runs/carrier_t2_s*`).

## What the numbers show

- With `top_winner`, when audio leads before the cycles, the cycles raise the vision bid
  by 0.359 to 0.401 on average. They raise the audio bid by 0.055 to 0.071. Between 53
  and 67 percent of the audio leads become vision leads.
- With `top_winner`, when vision leads, the cycles raise the audio bid by 0.095 to
  0.102. Between 2 and 13 percent of the vision leads flip.
- So the change is about 4 times larger for vision than for audio in the same
  situation. The rule in the code is the same for both senses.
- With `legacy`, 4 to 21 percent of the audio leads flip. The lead mostly survives the
  cycles.

This locates the loss of the competition under `top_winner` in the reentrant cycles.
It does not yet give the cause of the difference between the senses.

## An observation after the run. Not pre-stated, 3 runs, a hypothesis only

The rule computes pe = |broadcast - own vector| / |own vector| and adds 0.1 * pe to the
bid when pe is above 0.3 (`SensoryTectum.receive_broadcast`,
`AuditorySpecialist.receive_broadcast`). I read the recorded vectors of three runs.

| Run | Median size of the vision vector | Median size of the audio vector | Median pe of vision when the audio vector is the broadcast | Median pe of audio when the vision vector is the broadcast |
|---|---|---|---|---|
| `carrier_t1_s73` | 1.35 | 16.00 | 11.86 | 1.00 |
| `gate_b7_s73` | 1.22 | 16.00 | 13.10 | 1.00 |
| `carrier_t2_s83` | 1.39 | 16.00 | 11.68 | 1.01 |

The audio vector is about 12 times larger than the vision vector. The rule divides by
the size of the module's own vector. So when the audio vector is the broadcast, the
vision bid gets 0.1 * 12, which the code cuts at the maximum bid of 1.0. When the vision
vector is the broadcast, the audio bid gets 0.1 * 1. If this is the cause, the vision
takeover under `top_winner` comes from the different scales of the two vectors and not
from their content. This was seen after the result and is not tested.

## Replay test of the size hypothesis, 2026-10-13. VOID

The owner approved a read-only replay as the test. Probe:
`scripts/analysis/probe_reentrant_replay.py`, 15 tests. Its predictions and rule were
committed and pushed (`3abf53a`) before it ran on any record.

**The verdict is VOID. The validity condition R1 failed, so the test says nothing about
the size hypothesis.** The replay had to give the same flip or no flip as the record at
0.90 or more of the steps. At audio-led steps it agreed at 0.606 to 0.705.

| Run | Audio leads: flips in the record | flips in the replay | agreement | Vision leads: agreement |
|---|---|---|---|---|
| `carrier_t1_s73` | 0.586 | 0.952 | 0.634 | 0.965 |
| `carrier_t1_s74` | 0.532 | 0.926 | 0.606 | 0.976 |
| `carrier_t1_s75` | 0.559 | 0.940 | 0.620 | 0.966 |
| `carrier_t2_s83` | 0.615 | 0.951 | 0.664 | 0.968 |
| `carrier_t2_s84` | 0.562 | 0.946 | 0.617 | 0.870 |
| `carrier_t2_s85` | 0.674 | 0.956 | 0.705 | 0.942 |

The replay flips 93 to 96 percent of the audio leads. The record flips 53 to 67
percent. The replay leaves out the ignition test. At a step with no ignition there is no
broadcast, and then the rule cannot raise the vision bid. That is the probable reason
for the gap, and it was not tested.

The numbers of the unit-size arm are in the probe output. They are not reported here as
a result, because the replay they come from is not valid.

Next options, for the owner. A second replay that uses only steps with ignition, stated
before it runs. Or a default-off flag that scales both vectors before the rule, with a
pre-stated run at 3 seeds or more. The second one tests the hypothesis in the agent
itself.

## What this means for the open decision

- `top_winner` cannot be judged as a routing rule while this rule gives one sense a
  pe about 12 times larger than the other. The collapse of the competition at 6 of 6 seeds
  (`docs/results/broadcast_carrier_2026_10.md`) may be an effect of vector scale.
- The rule itself raises the bid of the module that is not in the broadcast. The
  sources in `docs/sensory_selection_evidence.md` describe the opposite. There the
  signals that are not selected are suppressed.

## Next step, for the owner to approve

The read-only replay was run and is VOID, see above. The size hypothesis is still not
tested.

## Limits

- Six seed pairs, 10 episodes each. The pairs differ only in the merge setting.
- The probe reads bids. It does not read the vectors. The vector table above is a
  separate look at 3 runs.
- Steps where one sense bids 0, and ties, are left out (17 to 213 steps of 1990 per
  run).
- A statement about bids. No indicator changes.

## Reproduce

```
python -m scripts.analysis.probe_reentrant_boost \
  --pair runs/gate_b7_s73 runs/carrier_t1_s73 --pair runs/gate_b7_s74 runs/carrier_t1_s74 \
  --pair runs/gate_b7_s75 runs/carrier_t1_s75 --pair runs/gate_b8_s83 runs/carrier_t2_s83 \
  --pair runs/gate_b8_s84 runs/carrier_t2_s84 --pair runs/gate_b8_s85 runs/carrier_t2_s85
```
