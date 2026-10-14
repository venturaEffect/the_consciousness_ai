# Stage 2 of the binding design: the agreement of place in the coupling

Date: 2026-10-14. Owner decision of the same day: stages 1, 2 and 3 of
`docs/decisions/2026_10_14_oscillator_binding_design.md`, in this order.

## The change

`--binding-relation place`, default `none`, unchanged.

- The tectum computes the agreement r of the place of the sound and the visual peak on
  the 16 by 16 map: r = exp(-d^2 / 8) for a distance d in cells
  (`models/core/sensory_tectum.py`, `place_agreement`). The sigma of 2 cells is an
  assumed value. It uses only signals of the agent and no ground truth.
- Before each binding the workspace gives r to the binding layer. The coupling of the
  vision and the audio oscillators becomes 2r - 1. So a sound at the visual peak keeps
  the coupling, and a sound far from it makes the two oscillators repel. The couplings
  of memory, body and semantic do not change.
- The basis is the spatial rule of multisensory integration (Stein and Meredith 1993):
  inputs in spatial register are combined, inputs at different places are not.
- At the end of a run the file `binding_relation.json` gives the number of calls with a
  relation, the mean r, and the share of r above 0.5.

## Pre-stated gate. Written before its runs

Runs: the Gate B7 command with `--episodes 20`, plus `--dark-room-decoy-sound
--dark-room-decoy-level 0.5,3.0 --binding-relation place`, seeds 125, 126 and 127, which
no run folder names. (Earlier versions of this text named seeds 113 to 115 and 119 to
121. Those runs were never made, because stage 1 attempts 2 and 3 were VOID. The test
bed of the fourth attempt, with 20 episodes and the corrected probe, replaces them.) Python 3.12.10. Each run alone. The test bed is the one of stage 1, fourth
attempt (`scripts/analysis/probe_binding_selectivity.py`, corrected).

| Check | Statement |
|---|---|
| C0 | The Gate B7 command at seed 73 without the new flags, at this revision, gives the `metrics.csv` of `runs/f2_py312_s73` (md5 `7825766f42d3d85b032f8028c94dc1e0`). If not, VOID |
| S0 | Each run has more than 0 calls with a relation. If not, VOID |
| B1 | Each run has 100 or more related and 100 or more unrelated steps. If not, VOID |
| G1 and G2 | The residual joint alignment (after the fit on the five bids) is higher at related steps than at unrelated steps, difference above 0 and one-sided p below 0.05, at 3 of 3 seeds |
| G3 | The binding changes the leader at 0.05 or more of the steps (`probe_binding_decisive.py`), at 3 of 3 seeds |
| G4 | `sync_r` takes 100 or more distinct values in each run |

Stage 2 PASSES when C0, S0 and B1 hold and G1 to G4 hold at 3 of 3 seeds. It FAILS when
one of G1 to G4 fails at 2 or more seeds. Otherwise it is UNDECIDED. The baseline is the
present layer on the same test bed, from the fourth attempt of stage 1 (seeds 122 to
124). If that attempt is VOID, this gate is not run.

My expectation is not firm. The other couplings still link vision and audio through the
other three oscillators, so the repulsion of one pair may not be enough.

This is a gate on the binding layer. It is not evidence about consciousness, and it
moves no indicator.

## Result (2026-10-14): FAILED

**Stage 2 FAILS. G1 fails at 2 of 3 seeds and G3 fails at 2 of 3 seeds.** C0, S0, B1
and G4 hold.

### The stage 1 baseline, fourth attempt

Seeds 122 to 124, 20 episodes, decoy levels 0.5 and 3.0, the present layer. B1 holds:
related and unrelated steps 419 and 244, 101 and 249, 337 and 266. B2 fails at 2 of 3
seeds: the residual difference is +0.00008 (p 0.001), -0.00003 (p 1.0) and +0.00029
(p 0.015). So the pre-stated reading "the present layer is not selective" is REJECTED.
The differences are very small. The first three attempts were VOID, and two of them
were judged with a probe that left out the decoy level. That error is corrected in
`scripts/analysis/probe_binding_selectivity.py`, and its docstring records it.

### Stage 2, seeds 125 to 127

| Check | Seed 125 | Seed 126 | Seed 127 |
|---|---|---|---|
| C0, md5 of the control run | `7825766f42d3d85b032f8028c94dc1e0`, identical | | |
| S0, calls with a relation | 8107 | 8381 | 8126 |
| B1, related and unrelated steps | 240 and 369 | 428 and 451 | 568 and 249 |
| G1 and G2, residual difference (p) | -0.102 (1.0) | -0.021 (1.0) | +0.039 (0.001) |
| G3, share of steps where the binding changes the leader | 0.029 | 0.084 | 0.044 |
| G4, distinct values of `sync_r` | 3980 | 3980 | 3979 |

Every run has 20 episode rows and 4000 metric rows. Sources:
`runs/_audit_followup/oct13/binding_selectivity_stage2.json` and
`binding_decisive_stage2.json`.

### What changed, description only

| Measure | Present layer, seeds 122 to 124 | Stage 2, seeds 125 to 127 |
|---|---|---|
| R squared of `sync_r` on the five bids | 0.993 to 1.000 | 0.089 to 0.122 |
| Share of steps where the binding changes the leader | 0.0000 to 0.0003 | 0.029 to 0.084 |
| Mean gap of the vision and audio factors | (not read) | 0.154 to 0.208 |

So the coupling term acts. The layer no longer follows the bids, and it now changes
the leader at some steps. But the synchrony is not higher when the two senses report
the same place. At two seeds it is lower.

### A probable reason, not tested

The mean agreement r was 0.128, 0.167 and 0.127, and r was above 0.5 at only 0.09 to
0.13 of the calls. So at most steps the vision and audio oscillators repelled each
other, also at related steps. A probable reason is that the visual peak (the cell with
the largest feature norm) does not mark the light. The agent's own mark is drawn at the
centre of the view, and it may be the peak. r is not recorded for each step, so this
cannot be checked on these runs.

### Limits

- Three seeds, one configuration, 20 episodes.
- The joint alignment is read back from the binding factors. The pairwise phase
  coherence of vision and audio was not recorded.
- sigma = 2 cells and the form 2r - 1 were assumed.
- It moves no indicator.

### Next step, for the owner

Stage 3 puts the oscillators on the map, so that each place has its own oscillator and
no single peak is needed. That removes the weak part of stage 2. A cheaper step first is
one run that records r and the true place of the light at each step, to show if the
place estimate is the cause.
