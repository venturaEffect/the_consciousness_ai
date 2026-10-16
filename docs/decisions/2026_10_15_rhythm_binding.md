# Binding by timing: oscillators lock when a seen object changes in time with a heard tone

Date: 2026-10-15. Status: rules written before any training run. The owner chose this option
(option A) and the oscillator mechanism on 2026-10-15.

## Why

The owner's requirement since 2026-10-14 is that the oscillators work: units with related content
synchronise, units with unrelated content do not, and the binding is used
(`docs/decisions/2026_10_14_oscillator_binding_design.md`, gates G1 to G4).

The motor map (PR 58) lets the agent act on what its tectum selects. In that room, binding by
place selected the louder object (P2 at 3 of 3 seeds) but gave no behavioural gain, because the
sound's direction alone already named the target: from the start between the objects, the room's
own sound synthesis and estimator pointed to the louder object at 1.000 of 400 offline starts
(`docs/results/tectal_motor_map_2026_10.md`).

So this test bed makes binding necessary. The sound has no direction, and each object pulses in
time with its own tone. Only the match in timing between a seen object and the heard tone tells
where the louder tone comes from.

## Sources

| Source | Read | Statement used here |
|---|---|---|
| Van der Burg, Olivers, Bronkhorst and Theeuwes, J. Exp. Psychol. Hum. Percept. Perform. (2008) | Abstract and later reviews | A tone with no place information, in time with a change of one visual object, makes that object easier to find |
| Atilgan and colleagues, Neuron 97, 640 (2018) | Abstract | When a light's luminance is coherent in time with the amplitude of one sound in a mixture, the two are bound; the visual signal enters through the phase of neural oscillations |
| Shamma, Elhilali and Micheyl, Trends Neurosci. 34, 114 (2011) | Abstract | The features of one source are grouped by their coherence in time |
| Parise and Ernst, Nat. Commun. 7, 11543 (2016) | Full text, summary | A detector of correlation between the senses explains causal inference and synchrony judgements |
| Lakatos and colleagues, Science 320, 110 (2008) | Reference only | Neural oscillations entrain to the rhythm of a stimulus |
| Fujisaki and Nishida, IMRF 2007 | Abstract | People judge audiovisual synchrony up to about 4 Hz |

## The room (flags off by default)

| Flag | Rule |
|---|---|
| `--dark-room-pulse` (needs the visible decoy) | Each episode, one object gets the rhythm 1/10 and the other 1/6 cycles per step, at random, with random phases, from a generator of their own (so the places of the objects do not change). Brightness gain 0.85 ± 0.15, so the dimmest object (luminance 0.68) is always brighter than the agent's ring (0.57). Tone loudness gain 0.7 ± 0.3, with the same rhythm |
| `--dark-room-sound-direction off` | Every ear hears the same signal, with no delay and no level difference. Loudness still falls with distance |

## The oscillators (`models/core/rhythm_binding.py`)

| Part | Rule |
|---|---|
| Input of a place | The largest luminance in its cell (the image before the encoder). An object reads the same wherever it covers part of a cell |
| Input of a band | The mean hair cell envelope of each cochlear band of the step (`AuditorySpecialist.last_band_envelope`) |
| Oscillators | For each input, 8 damped driven oscillators at 1/16 to 1/4 cycles per step: z(t) = (1 - 1/6) z(t-1) + (1/6)(x - slow mean) e^(-i 2 pi f t). Slow mean rate 0.1 |
| Loudest band | The band with the largest slow mean (rate 0.1), so the rhythm cannot swap the order from moment to moment |
| Binding | At the loudest band's strongest frequency, the coherence of each place with the band: magnitude of the mean product over the square root of the mean powers (leaky means, rate 1/8) |
| Group | Places with coherence of at least 0.55, without the 32 places of the agent's own body |
| Wait | No group before 12 steps of sound since the last restart |
| Restart | Every oscillator restarts at a step without sound, which is the first step of every episode |

One change from the approved plan, made before any run: the plan said phase locking value. A
place without input (the dark floor) has an oscillator of size 0, whose phase reads as a constant,
and a constant phase can look locked. The amplitude-weighted coherence gives such a place exactly 0.
A test pins this.

One fault found by a smoke run of 2 episodes, before the gate. The first version had no wait, and
groups formed at step 2 with coherence 1.000, also in the scrambled arm, because with one or two
samples the coherence of any two signals is 1 by its formula. The diagnosis had scored from step 12,
and the plan had set that the oscillators need about 12 steps before the agent moves, so the wait of
12 steps implements the plan. A test pins it. The repeated smoke run formed the first group at
step 12.

## Diagnosis before any training run: PASSED

The real room with both flags, the real cochlea, a scripted agent with no learning. 30 layouts
(room seeds 301 to 330); decoy level 0.5 at even seeds and 3.0 at odd seeds. The threshold was
chosen on layouts 301 to 315 (0.55, mean score 0.863), then tested on layouts 316 to 330. The louder
object at a step is the one with the larger level × loudness at its distance (it can change when
the agent comes near the quieter one). Score: share of steps 12 to 60 with the group on the louder
object (object places from geometry).

| Scripted motion | Rule | Rhythm binding | Lowest layout | Correlation detector (reference) |
|---|---|---|---|---|
| Still | ≥ 0.9 | 0.992 | 0.959 | 1.000 |
| Toward the louder object from step 12 | ≥ 0.8 | 0.996 | 0.959 | 1.000 |
| Toward the quieter object from step 12 | ≥ 0.7 | 0.710 | 0.612 | 0.061 |

The third case passed by a small margin. The reference detector kept the loudest band of its first
step, so it could not follow the change of the louder object; its low score shows that limit.

## Gate on fresh seeds

Seeds 211, 213 and 214, 40 episodes, one run at a time. The Gate B7 flags with
`--dark-room-decoy-sound --dark-room-decoy-level 0.5,3.0 --dark-room-decoy-visible
--dark-room-start-between --dark-room-pulse --dark-room-sound-direction off
--tectal-orienting-idle still`, and each arm's own flags.

| Arm | Flags | Prediction |
|---|---|---|
| rhythm | `--map-binding rhythm --tectal-orienting group` | first contact with the louder object at 0.8 or more of the episodes |
| scrambled | `--map-binding rhythm-scrambled --tectal-orienting group` (the sound's rhythm destroyed) | about half or lower |
| active | `--map-binding rhythm --tectal-orienting active` (all bright places, no selection) | about half |

Counted steps: step 12 and later, both objects in view. Contact: within 60 room pixels, within the
first 60 steps. The agent starts 60 to 88 room pixels from each object, so the first contact shows
which object the agent goes to. The rhythm and scrambled arms wait (`still`) until their first
group at step 12. The active arm has no selection and no wait, so it moves from the first step;
this gives it more time, not less.

Read by `scripts/analysis/probe_rhythm_binding.py`. Before the runs, a control with all new flags
off must give the md5 `7825766f42d3d85b032f8028c94dc1e0` for `metrics.csv` of seed 73.

| Gate | Rule, at 3 of 3 seeds of the rhythm arm |
|---|---|
| G1 selective | Per episode, the mean coherence of the places on the louder object minus that on the other object is above 0; one-sided Wilcoxon p below 0.05 |
| G2 more than the bids | Per episode, that difference regressed on the difference in mean luminance of the two objects: the intercept is above 0, p below 0.05 |
| G3 used | First contact with the louder object: rhythm above scrambled AND rhythm above active, each with a pooled one-sided Fisher p below 0.05 and a difference above 0 at each seed |
| G4 not frozen | The group is on the light in episodes with level 0.5 and on the decoy in episodes with level 3.0, each at more than 0.7 of the counted steps |

PASS when G1 to G4 all hold. FAIL when one fails at 2 or more seeds. G3 is judged on its task
part ("a disruption lowers a task measure"); the leader changes of the workspace are reported only.
Both senses get the same binding factor (1 + 0.5 × the largest coherence in the group) because
they are one object; with no group, both get 1.

## Limits, stated before the runs

- The orienting is a fixed reflex, and the gate of the basal ganglia is fixed open.
- The rhythms are two known frequencies. A room with irregular rhythms is a later test.
- The agent waits until it has a target (`still`), so the policy's drift (the relay defect,
  `docs/results/action_core_relay_defect_2026_10.md`) cannot touch an object first.
