# Python versions: what the repo says, what ran, and a pre-stated check for a new environment

Date: 2026-10-13. Status: decided. The check and the follow-up check PASSED, and new
result runs move to Python 3.12.10 with pyphi 1.2.0. The sections below are in the
order in which they were written.

## The mismatch

Every result in `docs/results/` to date was produced with Python 3.8.3 and torch
2.4.1+cu121. The repo says something else in four places, and the requirement file
cannot be installed on the Python that produced the results.

| Place | Statement or state | Source |
|---|---|---|
| The machine that produced the results | Python 3.8.3, torch 2.4.1+cu121, numpy 1.24.4, pandas 2.0.3, scikit-learn 1.3.2, pyphi 1.2.0 | `environment.json` of each recorded run, `pip freeze` |
| `README.md` badge and Requirements | Python 3.10+ | `README.md` |
| `docs/contributing.md` | Python 3.8 or higher | `docs/contributing.md` |
| Continuous integration | Python 3.10 | the workflow file |
| `requirements.txt` | `numpy>=1.26.0`, `pandas>=2.2.0`, `scikit-learn>=1.5.0`, `matplotlib>=3.8.0`. None of these has a release for Python 3.8 | `requirements.txt`, package index |
| `requirements.txt` | `pyphi>=1.2.0`. pyphi 1.2.0 does not import on Python 3.10 or later | `tests/test_phi_error_reporting.py` |
| `requirements.txt` | Lists accelerate, bitsandbytes, faiss-cpu, librosa, soundfile, fastapi, uvicorn, wandb, black and qwen-vl-utils. None is installed on the result machine | `pip freeze` |
| Thermodynamic extras | thrml 0.1.4 and nir 1.0.8 need Python 3.10 or later. They are installed on no interpreter of the result machine, so their tests skip | package index, test log |

The code itself is consistent. All 353 tracked Python files parse on Python 3.8, 3.10
and 3.12. A scan of the syntax tree found 0 places where the newer type syntax
(`list[int]`, `X | None`, `match`) is evaluated at run time without
`from __future__ import annotations`. The scan does not cover newer library calls such
as `zip(strict=True)`.

## Why Python 3.8 cannot stay as the only environment

Read from the package index on 2026-10-13.

| Package | Newest release | Python it needs | Newest release that accepts Python 3.8 |
|---|---|---|---|
| torch | 2.14.1 | 3.10 or later | 2.4.1 (the last release with a Python 3.8 wheel) |
| transformers | 5.19.0 | 3.10 or later | 4.46.3 |
| numpy | 2.5.4 | 3.12 or later | 1.24.4 |
| scipy | 1.18.1 | 3.12 or later | 1.10.1 |
| jax | 0.11.2 | 3.12 or later | 0.4.13 |
| Brian2 | 2.10.1 | 3.12 or later | 2.5.1 |
| thrml | 0.1.4 | 3.10 or later | none |
| nir | 1.0.8 | 3.10 or later | none |
| pyphi | 1.2.0 (2019), and 2.0.0rc3 (2026-10-02) | 1.2.0 states no limit and fails on 3.10 or later. 2.0.0rc3 needs 3.13 or later | 1.2.0 |

Python 3.8 ended upstream support in October 2024. Pretrained model families released
after transformers 4.46.3 cannot load on it. The optional thermodynamic libraries
cannot be installed on it.

## The pyphi constraint

pyphi is the one package that holds the result environment on Python 3.8.

- pyphi 1.2.0 is the release behind every phi value in the record. It implements IIT 3.0.
- pyphi 2.0.0 is at its third release candidate on the package index (2026-10-02), and
  the changelog in its repository dates the 2.0.0 release at 2026-10-12. It needs Python
  3.13 or later. The changelog describes a rework around IIT 4.0 with formalism objects
  that also cover IIT 3.0, and a changed interface. No test here shows that its IIT 3.0
  value equals the 1.2.0 value on this project's inputs. Until such a test exists, a
  value from it is not comparable with the `phi` column of earlier runs.
- pyphi is licensed under GPL 3.0 or later. A product that ships it must meet that
  licence.
- The project's own status of phi: the Phi-1 prediction is closed, and phi is contentless
  at 3 seeds (`docs/instrument_inventory.md`).

Options for a new environment, for the owner:

1. No pyphi on the new Python. The training loop writes the phi column as not computed.
   Phi stays available in the Python 3.8 record environment.
2. pyphi 2.0 on Python 3.13, after its final release. `models/evaluation/iit_phi.py`
   needs a port to the new interface. Phi from it registers as UNPROVEN, and it is
   compared with the old column only after an equality test on recorded inputs passes.
3. pyphi 1.2.0 on a new Python through a compatibility import. Tested below.

## What a run on a new Python does today

`models/evaluation/iit_phi.py` imports pyphi inside `try` and sets `pyphi = None` on an
`ImportError`. On Python 3.10 or later, pyphi 1.2.0 raises that error
(`cannot import name 'Iterable' from 'collections'`). The code then computes
`compute_phi_proxy_from_tpm`, which its own docstring calls an unvalidated
approximation, and labels the row `proxy` in `phi_method`.

Phi is not only logged. The training loop uses it at three places.

| Use | Place |
|---|---|
| Exploration: `phi_exploration_scale = max(0.2, 1.0 - phi * 10.0)` scales arousal | `scripts/training/train_rlhf.py`, `run_episode` |
| Reward: the phi-delta term `phi - prev_phi` | the same function |
| Memory: `attention_level=phi`, `priority=phi` | the same function |

So a run on Python 3.10 or later, with the code as it is, is driven by the proxy and
not by pyphi. It is a different agent from the one in the record. Continuous
integration uses Python 3.10, so its tests run this path.

## pyphi 1.2.0 on Python 3.12 with compatibility names: measured

pyphi 1.2.0 imports names that Python 3.10 removed from `collections` (they are in
`collections.abc`). When a launcher adds those names back before the import, pyphi
1.2.0 imports and computes on Python 3.12.10 with numpy 2.5.4.

| Network and state | Python 3.8.3 (numpy 1.24.4) | Python 3.12.10 (numpy 2.5.4) |
|---|---|---|
| `basic_network`, (1,0,0) | 2.3125 | 2.3125 |
| `xor_network`, (0,0,0) | 1.874999 | 1.875 |
| `fig4`, (1,0,0) | 1.916665 | 1.916666 |
| random 3-node TPM, 3 states | 0.254835, 0.344342, 0.2486 | 0.254835, 0.344345, 0.248604 |
| random 4-node TPM, 2 states | 0.877657, 0.630178 | 0.877659, 0.630173 |
| random 5-node TPM, 3 states | 0.849791, 1.308864, 1.17786 | 0.849795, 1.308865, 1.177869 |

The 11 values agree to within 0.00001. Two are identical. pyphi rounds to 6
decimals, and the solver libraries differ between the two environments (for example pyemd 1.0.0
against 2.0.0). These are 11
values on small networks. They are not a test on the project's gate TPM.

## Pre-stated check of the new environment. Written before the runs

**Question.** Does the Gate B7 result hold when the same command runs on Python
3.12.10 with torch 2.14.1+cu126 and pyphi 1.2.0 through the compatibility names?

**Reference.** Gate B7, seeds 73 to 82, Python 3.8.3, torch 2.4.1+cu121, revision
`4a01ba2` (`docs/decisions/2026_09_16_precision_weighted_bids.md`). It PASSED both
of its questions.

**Runs.** One run per seed. A run is repeated only when its csv files are incomplete
(10 episode rows, 2000 metric rows).

1. Control. Seed 73 with the Gate B7 command on current main and Python 3.8.3.
2. Test. Seeds 73 to 82 with the same command on Python 3.12, started through a
   launcher that adds the removed names to `collections` and then runs
   `scripts.training.train_rlhf` unchanged. No file of the repo is changed.
3. Description only. Seed 73 on Python 3.12 without the launcher, so that phi comes
   from the proxy.

**Predictions.**

- E1. The control gives a `metrics.csv` with the same md5 as the Gate B7 file of seed 73
  (`6099064c2361d70fd1d2ced48462511d`). If it does not, the code changed behaviour after
  `4a01ba2`, and I report that before any reading of the test.
- E2. In every test run no row has `phi_method` equal to `proxy`. If one has, the
  launcher did not reach pyphi and the test is void.
- E3. No test file is identical to its Python 3.8 file. The torch versions differ.
- E4. The test gives the same verdict as Gate B7 on each criterion of
  `scripts/analysis/probe_gate_b2.py`, read with the Gate B7 rules. No module has the
  highest bid at 0.95 or more of the ignited steps at any seed, and the task link
  holds (pooled rho above 0, one-sided p below 0.05, rho above 0 at 7 or more seeds).

**Verdict rule.** PASSED when E2 and E4 hold. FAILED when E4 does not hold. E1 and E3
are reported as they come out. A PASSED verdict says that the gate result holds in the
new environment. It does not say that the two environments give the same numbers, and
it moves no indicator.

**What I expect to differ.** Per-seed shares. Ten seeds give a spread of 0.522 to
0.789 for the top share on Python 3.8, so a seed can move by a large amount when the
arithmetic changes in the last bit.

## Result of the check, 2026-10-13. PASSED

**Two things failed before the runs, and they are not part of the verdict.** The first
build of the Python 3.12 environment failed, because that Python install has no bundled
pip file. It was built again without it. The first start of the test runs failed at
once at all seeds, because the launcher did not put the repo folder on the module path.
No run had started and no csv file existed. The launcher was corrected and each seed was
then run once.

All 12 runs are complete (10 episode rows and 2000 metric rows each).

| Prediction | Result |
|---|---|
| E1. The control on current main and Python 3.8.3 equals the Gate B7 file of seed 73 | HOLDS. `metrics.csv` has md5 `6099064c2361d70fd1d2ced48462511d` in both, and `episodes.csv` is identical too. The code changes after `4a01ba2` do not change this configuration |
| E2. No row of a test run has `phi_method` equal to `proxy` | HOLDS. Each test run has 399 `pyphi` rows, 1600 `skipped` and 1 `insufficient_data`, the same counts as its Python 3.8 run |
| E3. No test file is identical to its Python 3.8 file | HOLDS. The first row that differs is row 1 or 2 at every seed. At seed 73 it differs in the sixth decimal of two gate values |
| E4. The same verdict as Gate B7 on each criterion | HOLDS. See the table |

| Seed | Top share, 3.8 / 3.12 | Silence, 3.8 / 3.12 | Selectivity difference (null p95), 3.8 / 3.12 | rho, 3.8 / 3.12 |
|---|---|---|---|---|
| 73 | 0.564 / 0.519 | 0.351 / 0.342 | 0.103 (0.014) / 0.102 (0.015) | +0.469 / +0.488 |
| 74 | 0.787 / 0.783 | 0.268 / 0.265 | 0.264 (0.072) / 0.267 (0.069) | +0.671 / +0.789 |
| 75 | 0.736 / 0.777 | 0.300 / 0.303 | 0.150 (0.069) / 0.132 (0.061) | +0.244 / +0.094 |
| 76 | 0.683 / 0.766 | 0.314 / 0.319 | 0.118 (0.044) / 0.131 (0.039) | +0.508 / +0.502 |
| 77 | 0.715 / 0.686 | 0.311 / 0.313 | 0.140 (0.029) / 0.134 (0.022) | +0.322 / +0.255 |
| 78 | 0.522 / 0.523 | 0.333 / 0.335 | 0.092 (0.019) / 0.092 (0.019) | -0.313 / -0.305 |
| 79 | 0.656 / 0.678 | 0.303 / 0.292 | 0.070 (0.008) / 0.069 (0.006) | +0.620 / +0.768 |
| 80 | 0.653 / 0.642 | 0.310 / 0.308 | 0.131 (0.047) / 0.129 (0.047) | -0.289 / -0.277 |
| 81 | 0.789 / 0.785 | 0.355 / 0.352 | 0.074 (0.017) / 0.072 (0.017) | +0.529 / +0.501 |
| 82 | 0.664 / 0.692 | 0.295 / 0.300 | 0.137 (0.070) / 0.136 (0.070) | +0.726 / +0.726 |

| Question of the gate | Python 3.8.3 (Gate B7) | Python 3.12.10 |
|---|---|---|
| A module with the highest bid at 0.95 or more of the ignited steps | none, top share 0.522 to 0.789 | none, top share 0.519 to 0.785 |
| Task link, pooled over 100 episodes | rho 0.388, one-sided p 0.0005, rho above 0 at 8 of 10 seeds | rho 0.370, one-sided p 0.0005, rho above 0 at 8 of 10 seeds |
| Criteria 1 to 3 of the probe | pass at 10 of 10 seeds | pass at 10 of 10 seeds |

**Verdict. PASSED.** The Gate B7 result holds on Python 3.12.10 with torch
2.14.1+cu126 and pyphi 1.2.0 through the compatibility names. The largest move of a
top share at one seed is 0.083 (seed 76). The numbers are not the same, as predicted.

**The run with the proxy, description only, one seed.** Seed 73 on Python 3.12
without the launcher has 399 `proxy` rows. Its mean phi is 0.22463 and its largest phi
is 0.4875. With pyphi the mean is 0.00096 on Python 3.8 and 0.00088 on Python 3.12, and
the largest values are 0.0061 and 0.0044. The exploration rule is
`max(0.2, 1.0 - phi * 10.0)`. So with a typical proxy value the rule sits at its lower
limit of 0.2, and with pyphi it sits near 1.0. This is one seed and no gate. It shows
the size of the difference in the phi that drives the agent.

**Cost.** One run took 3 minutes 50 seconds on Python 3.8 and 4 minutes 13 seconds to 4
minutes 30 seconds on Python 3.12 with pyphi.

## What this establishes and what it does not

- It establishes that one gate result, at 10 seeds, is the same verdict in the new
  environment when pyphi 1.2.0 is made to import there.
- It does not establish that other configurations give the same verdicts. DMTS runs,
  runs with the learned valence, and the probes that load checkpoints were not run on
  Python 3.12.
- It does not establish that checkpoints saved on Python 3.8 load and behave the same
  on Python 3.12.
- No indicator and no clock value changes.

## Decisions for the owner

1. **The environment for new result runs.** Option A, stay on Python 3.8.3. Option B,
   move to Python 3.12 with pyphi 1.2.0 through the compatibility names. Option B needs
   one code change, a guarded import of those names in `models/evaluation/iit_phi.py`,
   and it makes every later run not bit-comparable with the record. My recommendation
   is option B, after one more pre-stated check on a DMTS configuration and a
   checkpoint load.
2. **The silent proxy.** On a Python where pyphi does not import, the training loop
   runs with the proxy and does not stop. A proposal. Make the training entry point
   stop with a clear message when pyphi is missing, unless a flag asks for the proxy.
   This changes behaviour on Python 3.10 or later only.
3. **pyphi 2.0.** A later and separate decision, on Python 3.13, with a port and an
   equality test first.
4. **Continuous integration.** The workflow file uses Python 3.10, where the tests run
   the proxy path and the thermodynamic tests skip. I do not edit that file. The
   proposed change is in the text of the pull request.

## Reproduce

```
# control, Python 3.8.3
python -m scripts.training.train_rlhf <Gate B7 flags> --seed 73 --log-dir runs/env_ctrl_py38_s73
# test, Python 3.12, through a launcher that adds the removed names to `collections`
<python 3.12> launcher.py <Gate B7 flags> --seed <73..82> --log-dir runs/env_py312_s<seed>
python -m scripts.analysis.probe_gate_b2 --runs runs/env_py312_s73 ... runs/env_py312_s82
```

The launcher is a short script outside the repo. It sets `collections.Iterable` and the 9 other removed names
from `collections.abc`, puts the working directory on `sys.path`, imports pyphi, and
calls `runpy.run_module("scripts.training.train_rlhf", run_name="__main__")`.

## Owner decisions of 2026-10-13, and the follow-up check. Written before its runs

The owner decided three points.

1. New result runs move to Python 3.12 with pyphi 1.2.0, after one more check on a
   DMTS run and a checkpoint load.
2. Training stops with a message when pyphi cannot be imported, unless a flag asks for
   the stand-in formula.
3. pyphi 2.0 stays a later and separate decision.

**The code change.** `models/evaluation/iit_phi.py` now puts the removed names into
`collections` for the import of pyphi and takes them out again. On Python 3.8 and 3.9 it
adds nothing. `scripts/training/train_rlhf.py` has a new flag `--phi-proxy`, off by
default, and stops at the command line when pyphi is missing and the flag is not set.
Tests: `tests/test_pyphi_import.py`.

**The follow-up check.** One run per item. A run is repeated only when its csv files are
incomplete.

- F1. Python 3.8.3, seed 73, the Gate B7 command, this revision. Prediction: the
  `metrics.csv` has md5 `6099064c2361d70fd1d2ced48462511d`, the Gate B7 file. The change
  does nothing on Python 3.8.
- F2. Python 3.12, seed 73, the Gate B7 command, this revision, started as
  `python -m scripts.training.train_rlhf` with no launcher. Prediction: the `metrics.csv`
  is identical to the file of the launcher run `runs/env_py312_s73`, and no row is
  `proxy`.
- F3. DMTS on both Pythons. `--env dmts --episodes 10 --max-steps 500 --seed 42
  --rssm-latent-mode continuous --capsule-workspace-source all_levels --existence-drive
  on`. Prediction: both runs complete with 10 episode rows, and the Python 3.12 run has
  no `proxy` row. The two runs are not identical. Their episode rewards are reported
  side by side. No pass line is set on the rewards, because 10 episodes of an untrained
  agent carry no verdict.
- F4. Checkpoint load on Python 3.12. The recording of model 42 (`runs/capfix_alllevels/
  tectum.pt`, a checkpoint saved on Python 3.8) is made with the `record` command of the
  survival probe, 280 trials, no sensory noise. Prediction: the checkpoint loads, the
  three label arrays are identical to the recording made on Python 3.8, and the three
  vector arrays are close but not identical. The largest absolute difference is
  reported. Then the project's linear decode of shape on `obs_map` and on `z_state`, on
  the project's split, is run on the new recording with the Python 3.12 libraries.
  Prediction: both are within one test row (0.012) of 0.988, the value on Python 3.8.

**Verdict rule.** PASSED when F1 and F2 hold, both DMTS runs complete with no `proxy`
row, the checkpoint loads with identical labels, and both decode values are within
0.012 of 0.988. FAILED when any of these does not hold. The move to Python 3.12 for new
result runs follows a PASSED verdict only.

## Result of the follow-up check (2026-10-13)

Revision `2c4b93d`, pushed before the first run. Each run was made once. Completion was
checked by csv row counts.

| Prediction | Result |
|---|---|
| F1. On Python 3.8 the `metrics.csv` is the Gate B7 file | HOLDS. md5 `6099064c2361d70fd1d2ced48462511d`, 10 episode rows, 2000 metric rows |
| F2. On Python 3.12 with no launcher the file equals the launcher run, and no row is `proxy` | HOLDS. Both files have md5 `7825766f42d3d85b032f8028c94dc1e0`. 399 `pyphi` rows, 0 `proxy` rows |
| F3. Both DMTS runs complete, and the Python 3.12 run has no `proxy` row | HOLDS. 10 episode rows and 5000 metric rows in each run. 999 `pyphi` rows and 0 `proxy` rows in each run. The files are not identical |
| F4. The checkpoint loads, labels identical, vectors close, both decode values within 0.012 of 0.988 | HOLDS. See the two tables below |

**F3, DMTS, seed 42, episode rewards side by side.**

| Episode | Python 3.8.3 | Python 3.12.10 |
|---|---|---|
| 0 | -58.1643 | -58.1643 |
| 1 | -55.5187 | -55.5191 |
| 2 | -57.2899 | -57.2899 |
| 3 | -60.7193 | -60.7196 |
| 4 | -61.4273 | -61.6288 |
| 5 | -54.1681 | -53.9585 |
| 6 | -55.7889 | -55.3772 |
| 7 | -63.1590 | -63.9581 |
| 8 | -57.4499 | -57.6486 |
| 9 | -63.4876 | -63.2900 |
| Mean | -58.7173 | -58.7554 |

The first metric row that differs is row 14, in `broadcast_mag`. The mean phi over the
999 `pyphi` rows is 0.00206 on Python 3.8 and 0.00221 on Python 3.12. The largest phi is
0.01690 and 0.01763. No pass line was set on these numbers.

**F4, the recording of model 42 made on Python 3.12, against the recording made on
Python 3.8.** 280 trials in each.

| Array | Identical | Largest absolute difference | Largest absolute value |
|---|---|---|---|
| `shape`, `color`, `size` (labels) | yes | not applicable | not applicable |
| `obs_map` | no | 0.0000021 | 3.237 |
| `z_state` | no | 0.0000014 | 4.260 |
| `tectum_content` | no | 0.00000003 | 0.068 |

| Linear decode of shape, project split, seed 42, run with the Python 3.12 libraries | `obs_map` | `z_state` |
|---|---|---|
| Recording made on Python 3.12 | 0.988 | 0.988 |
| Recording made on Python 3.8 | 0.988 | 0.988 |
| Value in `docs/results/thermodynamic_survival_2026_10.md`, Python 3.8 libraries | 0.988 | 0.988 |

Each value is 83 of 84 test rows. The Python 3.12 environment has scikit-learn 1.9.1
and numpy 2.5.4.

**Verdict. PASSED.** Every condition of the verdict rule holds. By the owner decision
above, new result runs move to Python 3.12.10 with pyphi 1.2.0.

**Limits.**

- One seed per item. This is a check that the environment works on two more
  configurations. It is not a replication of a DMTS result.
- A run on Python 3.12 is not bit-comparable with a run on Python 3.8. A comparison of
  two arms needs both arms from the same environment. The record of the Python 3.8
  environment is `requirements-py38-record.txt`.
- The pca and MLP decode, the survival evaluation with its channel, and the other
  probes that load checkpoints were not run on Python 3.12.
- The DMTS run took 26 minutes 25 seconds on Python 3.8 and 16 minutes 38 seconds on
  Python 3.12. One run each, so this is not a speed result.
- No indicator and no clock value changes.

**Tests on this branch.** `python -m pytest tests/ -q --ignore=tests/test_emergence_governor.py`
gives 1729 passed, 12 skipped on Python 3.8.3 and 1734 passed, 7 skipped on Python
3.12.10.

**Reproduce.**

```
python -m scripts.training.train_rlhf <Gate B7 flags> --seed 73 --log-dir runs/f1_py38_s73
<python 3.12> -m scripts.training.train_rlhf <Gate B7 flags> --seed 73 --log-dir runs/f2_py312_s73
<python> -m scripts.training.train_rlhf --env dmts --episodes 10 --max-steps 500 --seed 42 \
    --rssm-latent-mode continuous --capsule-workspace-source all_levels --existence-drive on --log-dir <folder>
<python 3.12> scripts/analysis/probe_thermodynamic_survival.py record --models 42 --noise 0 --out-dir <folder>
```
