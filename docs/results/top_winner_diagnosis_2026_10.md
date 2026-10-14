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

The read-only replay was run and is VOID, see above. The size hypothesis was then
tested on the real loop on 2026-10-14. The result is NOT SUPPORTED by the pre-stated
rule, see the last section. Two decisions are open for the owner.

1. A pre-stated test of the observation that the winner shares return to the range of
   the default merge when the sizes are equal. It needs fresh seeds.
2. The feedback rule raises the bid of the module that is not in the broadcast, at any
   vector size. The sources in `docs/sensory_selection_evidence.md` describe
   suppression of the signals that are not selected. A change of the rule is a design
   decision.

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

## Size test on the real loop (2026-10-14): NOT SUPPORTED by the pre-stated rule

**Prediction S1 fails at 2 of 3 seeds (73 and 75). By the rule the size hypothesis is
NOT SUPPORTED.** The control check C0 and the check S0 hold.

The replay above was VOID, so the hypothesis was tested on the training loop itself.
`scripts/analysis/run_unit_feedback.py` starts the loop with one change made in memory.
Before each of the two feedback rules runs, the broadcast is scaled to the size of the
module's own vector. No production file changes. The predictions and the rule were
pushed in `4d3cbcb` before the first run. 25 tests in
`tests/test_unit_feedback_launcher.py`.

| Check | Result |
|---|---|
| C0. Seed 73 of the control command, run again at this revision, equals `runs/carrier_t1_s73` | HOLDS. Both `metrics.csv` files have md5 `7391f2eca5bbf87ed4abf9bbc8a6ae1e` |
| S0. Each unit run has scaled calls for vision and for audio | HOLDS. 3734 and 3733 at seed 73, 2738 and 2736 at seed 74, 3283 and 3282 at seed 75 |
| S1. The share of audio leads that flip is below half of the control share | **FAILS at seeds 73 and 75.** Holds at seed 74 |

| Seed | Audio leads that flip, control | Half of it | Unit arm | S1 |
|---|---|---|---|---|
| 73 | 0.586 | 0.293 | 0.302 | false |
| 74 | 0.532 | 0.266 | 0.221 | true |
| 75 | 0.559 | 0.280 | 0.439 | false |

### What else the runs show. Description, not a pre-stated result

| Seed | Vision bid change when audio leads, control / unit | Vision leads that flip, control / unit | Audio bid change when vision leads, control / unit |
|---|---|---|---|
| 73 | +0.369 / +0.198 | 0.035 / 0.309 | +0.095 / +0.168 |
| 74 | +0.372 / +0.154 | 0.024 / 0.147 | +0.097 / +0.156 |
| 75 | +0.392 / +0.236 | 0.034 / 0.181 | +0.096 / +0.157 |

| Seed | Winner share at ignited steps, control (`top_winner`) | Unit arm (`top_winner` with equal sizes) | Default merge (`runs/gate_b7_s*`) |
|---|---|---|---|
| 73 | vision 1.000 | vision 0.454, audio 0.546 | vision 0.564, audio 0.436 |
| 74 | vision 0.999, audio 0.001 | vision 0.658, audio 0.342 | vision 0.787, audio 0.213 |
| 75 | vision 0.997, audio 0.003 | vision 0.783, audio 0.217 | vision 0.736, audio 0.264 |

- The size of the vectors is a part of the effect and not all of it. With equal sizes
  the cycles still raise the vision bid by 0.154 to 0.236 when audio leads, against
  0.369 to 0.392 in the control arm.
- With equal sizes the cycles flip leads in both directions. The share of vision leads
  that flip rises from 0.024 to 0.035 in the control arm to 0.147 to 0.309. So the
  rule "the module behind gains" flips leads at any vector size, as the expectation
  written before the runs said.
- The winner shares changed more than the flip shares. In the control arm vision wins
  0.997 to 1.000 of the ignited steps. In the unit arm vision wins 0.454 to 0.783 and
  hearing 0.217 to 0.546. I saw this after the result. No prediction covered it, so it
  is a hypothesis for a next test and it is not a finding.

### How the runs were made, with two errors of mine

- The first launch of seed 73 of the unit arm ended after 10 seconds with no data row.
  A website build ran on the machine at that moment. I do not know the cause. The run
  was made again.
- A stopped chain of runs was not stopped in fact. So seed 73 and seed 75 of the unit
  arm ran at the same time for about 3 minutes, and seed 75 was then made a second
  time alone. I made seed 73 once more alone. Its `metrics.csv` is identical to the file
  of the parallel run (md5 `cce4d1b6f401982ad4e1fe9279ed4d4c`). The table uses the run
  made alone. Both runs of seed 75 gave the same call counts.
- Every run has 10 episode rows and 2000 metric rows.

### Limits

- Three seeds, one configuration (no learned valence), 10 episodes for each run. This
  is hypothesis grade.
- The two arms see different frames after the first changed action. The comparison is
  of run statistics at matched seeds.
- The launcher is a test instrument. It is not a proposed mechanism. A mechanism for
  the feedback rule is a design decision, and it needs the review of the biology first.
- It is a statement about bids. It is not evidence about consciousness, and it moves no
  indicator.

### Reproduce

```
python -m scripts.analysis.run_unit_feedback <control command> --seed 73 --log-dir runs/unit_t1_s73
python -m scripts.analysis.run_unit_feedback judge --pair runs/carrier_t1_s73 runs/unit_t1_s73 \
    --pair runs/carrier_t1_s74 runs/unit_t1_s74 --pair runs/carrier_t1_s75 runs/unit_t1_s75
```

## Second test on the real loop, the winner shares (2026-10-14): SUPPORTED at 3 of 3 fresh seeds

**With equal vector sizes the vision takeover under `top_winner` is gone at all 3 fresh
seeds. W0, W1 and W2 hold at 3 of 3.**

The first test above gave NOT SUPPORTED for the flip share, and after it I saw the
change of the winner shares at seeds 73 to 75. This test was written for that
observation. The predictions and the rule were pushed in `ba67da1` before the first
run. Seeds 93, 94 and 95 were not used before. Both arms were made new with the same
command, on Python 3.12.10 with torch 2.14.1+cu126 and pyphi 1.2.0. Each run was made
alone, one after the other.

| Seed | Control arm, `top_winner` | Unit arm, `top_winner` with equal sizes | W1, control vision at 0.95 or more | W2, unit vision below 0.95 |
|---|---|---|---|---|
| 93 | vision 0.983, audio 0.017 | vision 0.604, audio 0.396 | true | true |
| 94 | vision 0.994, audio 0.006 | vision 0.649, audio 0.351 | true | true |
| 95 | vision 0.974, audio 0.026 | vision 0.645, audio 0.355 | true | true |

The share is the share of the ignited steps that a module wins. The silence is 0.230 to
0.260 in the control arm and 0.224 to 0.267 in the unit arm. The scaled calls of the
unit runs are 3807 to 4014 for each sense. Every run has 10 episode rows, 2000 metric
rows and no `proxy` row. Source: `runs/_audit_followup/oct13/winner_test.json`.

### What this shows and what it does not

- The takeover of the competition under `top_winner` depends on the sizes of the two
  sense vectors. This holds at 3 fresh seeds in one configuration, and it agrees with
  the observation at seeds 73 to 75 on Python 3.8.
- The flip share of the first test does not show it. At these seeds the share of audio
  leads that flip goes from 0.587, 0.659 and 0.648 to 0.356, 0.370 and 0.508. That is
  again not below half. So the flip share was the wrong measure for the takeover, and
  the verdict of the first test stands as it is.
- It does not show that equal sizes are the right mechanism. The feedback rule still
  raises the bid of the module that is not in the broadcast, and with equal sizes 25 to
  34 percent of the vision leads flip. The sources in
  `docs/sensory_selection_evidence.md` describe suppression of the signals that are not
  selected.
- It does not show a link to the task. No task measure was read in this test.
- Three seeds and one configuration. It changes no default, and it moves no indicator.

### Decision for the owner

At the seeds tested here, the collapse of the competition with `top_winner`
(`docs/results/broadcast_carrier_2026_10.md`) depends on the vector sizes in the feedback
rule. The three seeds of that document with the learned valence were not tested. So
`top_winner` was not judged as a routing rule yet. Two ways are open.

1. Put the size correction into the code behind a default-off flag, and run the gate
   for `top_winner` again with it at 10 fresh seeds.
2. Review the feedback rule first, because it rewards the module that lost. That is a
   design decision and needs the review of the biology.

### Reproduce

```
<python 3.12> -m scripts.training.train_rlhf <control command> --seed 93 --log-dir runs/top_c_s93
<python 3.12> -m scripts.analysis.run_unit_feedback <control command> --seed 93 --log-dir runs/top_u_s93
<python 3.12> -m scripts.analysis.run_unit_feedback judge-winners --pair runs/top_c_s93 runs/top_u_s93 \
    --pair runs/top_c_s94 runs/top_u_s94 --pair runs/top_c_s95 runs/top_u_s95
```
