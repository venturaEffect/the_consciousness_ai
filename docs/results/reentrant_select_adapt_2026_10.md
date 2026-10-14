# Gate R1: the select_adapt feedback rule at 10 fresh seeds, with both broadcast merges

Date: 2026-10-14. Owner decision of the same day: build option A with C of
`docs/decisions/2026_10_14_reentrant_feedback_rule_review.md` and test it.

## The rule

`--reentrant-feedback select_adapt` (`models/core/reentrant_processor.py`). In each
reentrant cycle the sense that leads the competition gains a factor 1.1 and the other
sense loses a factor 0.9. The broadcast vector is not read. After each step the sense
that won adapts, and its next bids start lower by up to half. gain 0.1, rate 0.1 and
depth 0.5 are assumed values and were not fitted. Default `legacy`, unchanged.

## Pre-stated gate. Written before its runs

Command: the Gate B7 command plus `--reentrant-feedback select_adapt`, once with
`--broadcast-merge legacy` (the default) and once with `--broadcast-merge top_winner`.
Seeds 100 to 109, which no document and no run folder names. Python 3.12.10. Each run
is made alone. Read with `scripts/analysis/probe_gate_b2.py`, criteria unchanged.

| Check | Statement |
|---|---|
| C0 | The Gate B7 command at seed 73 without the new flag, at this revision on Python 3.12.10, gives the `metrics.csv` of `runs/f2_py312_s73` (md5 `7825766f42d3d85b032f8028c94dc1e0`). If not, the gate is VOID |
| (1) Competition | At each seed the runner-up takes at least 0.05 of the ignited steps |
| (2) Silence | At each seed the steps without a winner are below 0.50 |
| (3) Selectivity | At each seed, as defined in `probe_gate_b2.py` |
| (4) Task | Pooled over the 10 seeds, Spearman rho of the out-of-view share against the audio share above 0 with a one-sided p below 0.05, and rho above 0 at 7 or more seeds |
| KILL | Any module at 0.95 or more of the ignited steps at any seed |

The gate PASSES for a merge when C0 holds and criteria 1 to 4 pass with no KILL. It is
judged for each merge on its own. Reported only, never a gate: the winner shares of
each seed, the share of audio against the episode index, and the comparison with Gate
B7 (`legacy` rule, `legacy` merge, seeds 73 to 82).

My expectation is not firm. Option A strengthens the leader, so a lock-in of one sense
can happen, and then the KILL rule fires. Option C is the guard against that.

It is a gate on the competition of the workspace. It is not evidence about
consciousness, and it moves no indicator.

## Result (2026-10-14): PASSED for both merges

**Gate R1 PASSES with `--broadcast-merge legacy` and with `--broadcast-merge top_winner`.
C0 holds, criteria 1 to 4 pass, and the KILL rule does not fire at any of the 20 runs.**

C0: the control run `runs/r1_c0_s73` has md5 `7825766f42d3d85b032f8028c94dc1e0`, the
same as `runs/f2_py312_s73`. Every run has 10 episode rows and 2000 metric rows. Each
run was made alone, in one chain, from revision `1174f68`.

| Criterion | `legacy` merge | `top_winner` merge |
|---|---|---|
| (1) Runner-up share, 10 seeds | 0.255 to 0.442 | 0.291 to 0.437 |
| (2) Silence, 10 seeds | 0.222 to 0.290 | 0.193 to 0.291 |
| (3) Selectivity | passes at 10 of 10 | passes at 10 of 10 |
| (4) Task, pooled rho over 100 episodes | 0.359, one-sided p 0.0005 | 0.395, one-sided p 0.0005 |
| (4) Seeds with rho above 0 | 9 of 10 | 9 of 10 |
| KILL, largest top share | 0.745 | 0.709 |

In every run vision has the top share and hearing is the runner-up. The vision share
per seed is 0.558 to 0.745 with `legacy` and 0.563 to 0.709 with `top_winner`.

Source: `runs/_audit_followup/oct13/gate_r1_legacy.txt` and `gate_r1_top_winner.txt`,
from `scripts/analysis/probe_gate_b2.py` on `runs/r1_legacy_s100` to `_s109` and
`runs/r1_top_winner_s100` to `_s109`.

### Reported only, never a gate

- Gate B7 (`legacy` rule, `legacy` merge, seeds 73 to 82, Python 3.8.3) had a top share
  of 0.522 to 0.789 and a pooled rho of 0.388. Those runs used other seeds, another
  Python and another rule, so this is not a comparison of effect.
- With the `legacy` rule, `top_winner` gave vision 0.974 to 1.000 of the ignited steps
  (`docs/results/top_winner_diagnosis_2026_10.md`). With `select_adapt` it gives 0.563
  to 0.709. The seeds differ, so this is also not a measured effect of the rule alone.

### Limits

- One configuration (no learned valence), 10 seeds for each merge, 10 episodes each.
- The constants gain, rate and depth were assumed and not varied.
- The adaptation is not reset between episodes.
- The rule changes which sense the cycles favour. No document here shows that it
  changes what the agent perceives or learns.
- No default changes, and no indicator moves. A change of default is a decision for the
  owner.
