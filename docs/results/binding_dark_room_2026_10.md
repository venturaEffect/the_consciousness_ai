# The binding layer in the dark room follows the bids and does not decide the competition

Date: 2026-10-14. Read-only probe on existing session records. No new run. No default
changes and nothing was built.

## Verdict

**The pre-stated reading is SUPPORTED. Both predictions hold at 10 of 10 seeds.**

| Prediction, pushed in `50bda2d` before the probe read a run | Result |
|---|---|
| O1. The binding changes the leader at less than 0.05 of the steps | HOLDS at 10 of 10. The largest share is 0.0015 |
| O2. A linear fit of `sync_r` on the five bids has an R squared of 0.90 or more | HOLDS at 10 of 10. The lowest value is 0.952 |

Before the prediction I had seen one step of one run. The docstring of the probe says
which one.

Probe: `scripts/analysis/probe_binding_decisive.py`, 16 tests in
`tests/test_binding_decisive_probe.py`. Data: the 10 runs of Gate B7,
`runs/gate_b7_s73` to `runs/gate_b7_s82`, 1990 recorded steps each.

## What was measured

The bids before the binding (`internals.modulated_bids`), the bids after it
(`bound_bids`) and `sync_r`. The leader is the module with the highest bid. The factor
of a module is its bound bid divided by its bid before the binding.

| Seed | Steps | Leader changed | R squared of `sync_r` on the bids | Mean factor, vision / audio / memory / body | Mean gap of the vision and audio factors |
|---|---|---|---|---|---|
| 73 | 1974 | 0.0000 | 0.969 | 1.493 / 1.493 / 1.478 / 1.487 | 0.0005 |
| 74 | 1975 | 0.0015 | 0.990 | 1.484 / 1.493 / 1.493 / 1.492 | 0.0084 |
| 75 | 1975 | 0.0000 | 0.991 | 1.491 / 1.482 / 1.492 / 1.487 | 0.0084 |
| 76 | 1990 | 0.0000 | 0.981 | 1.493 / 1.494 / 1.469 / 1.493 | 0.0025 |
| 77 | 1970 | 0.0000 | 0.990 | 1.496 / 1.495 / 1.491 / 1.496 | 0.0008 |
| 78 | 1989 | 0.0000 | 0.952 | 1.496 / 1.482 / 1.496 / 1.492 | 0.0134 |
| 79 | 1990 | 0.0000 | 0.994 | 1.496 / 1.490 / 1.497 / 1.489 | 0.0055 |
| 80 | 1970 | 0.0000 | 0.990 | 1.495 / 1.495 / 1.493 / 1.493 | 0.0007 |
| 81 | 1974 | 0.0000 | 0.998 | 1.500 / 1.499 / 1.500 / 1.498 | 0.0008 |
| 82 | 1990 | 0.0000 | 0.998 | 1.494 / 1.493 / 1.492 / 1.494 | 0.0004 |

Steps with a tie at the top are left out, 0 to 20 for each seed. The semantic module has
a bid of 0 at every step, so it has no factor.

## What the numbers show

- The layer is not frozen in the dark room. `sync_r` takes 1984 to 1990 values in the
  recorded steps of a seed, with a standard deviation of 0.050 to 0.077.
- The five bids explain 95.2 to 99.8 percent of the variance of `sync_r` with a linear
  fit. So `sync_r` reports the bids here, as it does on DMTS.
- The binding multiplies every bid by nearly the same number. The mean factor of a
  module is between 1.469 and 1.500. The largest factor that the code allows is 1.5
  (`1 + 0.5 * align`). The vision factor and the audio factor differ by 0.0004 to 0.0134.
- The leader changes at 3 of 19,797 steps. All 3 are at seed 74.

So in the one environment where the layer moves, it acts as one common gain of about
1.49 on every bid. It does not select a module.

## Limits

- One configuration, the Gate B7 command, at 10 seeds of 10 episodes each.
- The probe reads the order of the bids. It does not measure if the common gain
  changes how often the workspace ignites.
- The fit leaves 0.2 to 4.8 percent of the variance of `sync_r` without an account. I
  did not examine that part.
- The probe does not test if `sync_r` carries content in the dark room. On DMTS that
  test FAILED at 3 seeds (`sync_r_content_2026_09.md`, `akorn_omega_fix_2026_09.md`).
- It is not evidence about consciousness, and it moves no indicator.

## What the project documents already established

| Document | Finding |
|---|---|
| `oscillator_frozen_2026_09.md` | On DMTS the layer emits one value for 13,600 steps. A reset at the start of each episode gives a transient that does not decode the stimulus (0.172 against a null p95 of 0.222) |
| `modality_starvation_2026_09.md` | The freeze comes from a constant input. On DMTS one of five modules has sensory input |
| `sync_r_content_2026_09.md` | `sync_R` carries no stimulus identity at 3 seeds. At full synchrony it equals the mean of the bids |
| `akorn_omega_fix_2026_09.md` | With the natural frequency repaired behind a default-off flag, `sync_R` moves at every step and still carries no stimulus identity |
| `tectum_wave_detection_2026_09.md` | The phase gradient in the tectum state comes from the input path at 2 of 3 seeds, so it is not evidence of a traveling wave |
| `traveling_waves_evidence.md` | The review licenses measurement. It does not license a wave lattice build |

## What the literature says about oscillations in the midbrain and about phase reset

Read on 2026-10-14. Each item says how much of the source was read.

- **Sridharan, Boahen and Knudsen, Journal of Neurophysiology 105, 2005 (2011). Main
  text read in part.** In the optic tectum of the barn owl a visual or auditory
  stimulus at one place evokes gamma oscillations (25 to 140 Hz). The gamma power is
  sharply tuned for the place of the stimulus and follows the space map. The authors
  propose a role in spatial attention and state that they did not test it.
- **Goddard, Sridharan, Huguenard and Knudsen, Neuron 73, 567 (2012). Main text read
  in part.** The oscillations are generated inside the multisensory layers of the
  tectum by a circuit of excitatory and inhibitory cells, and a relay nucleus (Ipc)
  carries them to the superficial layers. After one brief pulse they last more than
  150 ms (median 284 ms in the superficial layers). The authors propose three
  functions. The rhythm synchronises one restricted column across the layers, it sends
  a synchronised priority signal to other brain areas, and it holds the place of a
  salient stimulus for a short time.
- **Lakatos and colleagues, Neuron 53, 279 (2007). Abstract-level summary read.** In
  the primary auditory cortex of the macaque a touch input resets the phase of the
  ongoing oscillations. A sound that arrives at the high-excitability phase gives a
  larger response, and a sound at the opposite phase a smaller one. A later paper
  argues that the same pattern can come from evoked responses. I did not read that
  paper.
- **Voloh and Womelsdorf, Frontiers in Systems Neuroscience 10, 18 (2016), a review.
  Abstract read.** A phase reset in response to a cue aligns low frequencies to which
  faster activity can synchronise, and the review links it to information transfer
  between areas and to behaviour.
- **Knudsen, Trends in Neurosciences (2018).** Read earlier in this audit, see
  `docs/sensory_selection_evidence.md`. The midbrain selects a place. The relay
  nucleus amplifies and synchronises the activity of the selected place.

| Property | Midbrain, by these sources | Binding layer in this code |
|---|---|---|
| Where the oscillation is | At the place of the stimulus on the map, in a restricted column | 5 units, one for each module, with no place |
| What starts and ends it | A stimulus starts it. It ends after some hundred milliseconds | It runs at all times. No event resets the phases |
| What it changes | Proposed: the processing at that place, and the signal sent to other areas | One gain of about 1.49 on every bid |

## Options for the owner

Nothing here is built and no default changes.

| Option | What it is | Evidence |
|---|---|---|
| A | Keep the layer as it is. State in the public documents that it acts as a common gain today and does not select | This document and the four result documents above |
| B | Measure first. A read-only probe on existing checkpoints asks if the tectum state already holds activity that is local to the place of the stimulus and that outlasts the stimulus | The two tectum papers describe that form. The waves review licenses measurement |
| C | Build an oscillation on the tectum map at the selected place | The review does not license a build before a measurement. This is an architecture decision |
| D | Add a phase reset on a salient event to the 5 oscillators | A reset at each episode start was tested and gave no content. The reset account in the literature is disputed |

My recommendation is A now, and B as the next step if the oscillator work continues. I
do not recommend D.

## Reproduce

```
python -m scripts.analysis.probe_binding_decisive --runs runs/gate_b7_s73 runs/gate_b7_s74 \
    runs/gate_b7_s75 runs/gate_b7_s76 runs/gate_b7_s77 runs/gate_b7_s78 runs/gate_b7_s79 \
    runs/gate_b7_s80 runs/gate_b7_s81 runs/gate_b7_s82
```
