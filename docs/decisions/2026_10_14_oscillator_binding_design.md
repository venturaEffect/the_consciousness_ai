# Oscillatory binding: what "works" means, why it does not work today, and a staged design

Date: 2026-10-14. Status: proposal. The owner decided on 2026-10-14 that the oscillators
must work, and that a statement of their limits is not enough. This document says what
"works" means as tests, gives the cause of the present state, and proposes a design in
stages. No code changes with this document.

## The role that the project gives to the layer

| Source | Statement |
|---|---|
| `docs/architecture.md` | Kuramoto oscillators synchronize module representations. Modules that process related information phase-lock into unified percepts |
| `docs/feinberg_mallatt_approach.md`, feature 6 | Synchronized oscillations bind dispersed representations into unified percepts |
| `docs/biological_neural_architecture_research.md` | Signals that fire in synchrony are tagged as parts of the same object or event |
| `docs/consciousness_indicators_butlin.md` | The layer is named in the rationale of RPT-2 and GWT-2. Both are PARTIAL |

So the layer has one job. It must synchronize the modules whose content is related, and
it must not synchronize the others.

## Why it cannot do that job today

`WorkspaceBindingSystem.bind_bids` receives five numbers, the bids. It receives no
content. The coupling between two oscillators is the product of their two bids and a
learned weight. So the layer has no input that says if two modules process related
information. The measured results follow from that.

| Measured | Source |
|---|---|
| On DMTS the layer gives one value for 13,600 steps | `docs/results/oscillator_frozen_2026_09.md` |
| `sync_R` carries no stimulus identity at 3 seeds, with and without the natural frequency repair | `docs/results/sync_r_content_2026_09.md`, `docs/results/akorn_omega_fix_2026_09.md` |
| In the dark room the bids explain 95.2 to 99.8 percent of `sync_r`, and the layer changes the leader at 3 of 19,797 steps | `docs/results/binding_dark_room_2026_10.md` |
| The layer multiplies every bid by 1.469 to 1.500 | the same document |

A second fact. The dark room has one object, the light. The light gives the image and
the sound. So no recorded step has two senses with content that is not related. The
present environment cannot show selective binding, with any layer.

## What the literature says

Read on 2026-10-13 and 2026-10-14. Each item says how much was read.

| Source | Read | Finding |
|---|---|---|
| Sridharan, Boahen and Knudsen, J. Neurophysiol. 105, 2005 (2011) | Main text in part | In the owl tectum a stimulus evokes gamma (25 to 140 Hz) at its place on the space map. A role in attention is proposed and was not tested |
| Goddard and colleagues, Neuron 73, 567 (2012) | Main text in part | A local circuit in the multisensory layers makes the rhythm. It lasts some hundred milliseconds after a brief pulse. Proposed functions: synchronise one column, send a priority signal, hold a salient place for a short time |
| Knudsen, Trends Neurosci. 41, 789 (2018) | Main text, see `docs/sensory_selection_evidence.md` | The midbrain selects a place. A relay nucleus amplifies and synchronises the selected place |
| Miyato, Löwe, Geiger and Welling, ICLR 2025 (AKOrN) | Abstract | Kuramoto units replace threshold units and can use fully connected, convolutional or attention connectivity. Gains are reported for object discovery, among other tasks |
| Shadlen and Movshon, Neuron 24, 67 (1999) | Summary of the abstract | A critical review. It calls the evidence for binding by synchrony indirect |
| Ray and Maunsell (2010, 2015), as described by later papers | Not read directly | The gamma frequency changes with the stimulus, which is a problem for synchrony between the parts of one object |
| Karimian and colleagues, eLife (reviewed preprint 105482) | Summary | A model in which the stimulus dependence helps. Neighbours with similar frequencies synchronise and others do not. The reviewers note that no recording links it to perception directly |
| Lakatos and colleagues, Neuron 53, 279 (2007) | Summary of the abstract | A touch input resets the phase of oscillations in auditory cortex. A later paper disputes the reset account. I did not read that paper |

Three points follow for the design.

1. In the midbrain the oscillation is at a place on the map and a stimulus starts it. It
   is not one unit for one sense.
2. In the models where synchrony does work, the coupling depends on the content. Units
   with similar input synchronise and others do not.
3. Binding by synchrony is disputed in the cortex. So the project must show by its own
   tests that the layer does work. A reference to biology is not enough.

## "Works" as four gates

Each gate is written before its runs, at 3 or more seeds, with the default-off rule.

| Gate | Statement | Present state |
|---|---|---|
| G1. Selective | Synchrony between two modules is higher when their content is related than when it is not | Cannot be tested. The environment has no step with content that is not related |
| G2. More than the bids | The synchrony carries the relation after the bids are removed by a fit | Fails. The bids explain 95.2 to 99.8 percent |
| G3. Used | The binding changes the outcome of the competition for related content, and a disruption of synchrony lowers a task measure | Fails. The leader changes at 3 of 19,797 steps |
| G4. Not frozen | The layer moves when the input moves, and returns after an event | Holds in the dark room. Fails on DMTS |

## Design in three stages

| Stage | What | Changes | Cost |
|---|---|---|---|
| 1. A test bed | A dark room option with a second sound source that is not at the light, behind a default-off flag. Then G1 can be measured. A read-only probe measures the present layer on it, to set the baseline | Environment only. The agent does not change | My estimate is 1 day. Short runs |
| 2. Content in the coupling | The coupling between two oscillators gets a term for the relation of the two contents. For the two senses the relation is the agreement of their place on the tectum map. Behind a default-off flag. The baseline stays bit-identical | `models/core/oscillatory_binding.py` and the call in the workspace. `run_episode` and `init_components` get only the flag | My estimate is 2 days. Gates G1 to G4 at 3 seeds, then at fresh seeds |
| 3. Oscillators on the map | One oscillator for each cell of the 16 by 16 tectum grid, with local coupling, driven by the visual and the auditory map. The synchronised place gives the priority signal. This is the form that the tectum sources describe, and the AKOrN abstract names convolutional connectivity as one of its options | A new module in the tectum. An architecture change | My estimate is 1 to 2 weeks. It starts only when stage 2 has passed or failed with a clear reason |

Why this order. Stage 1 is needed for every design, because without it no gate can
show selectivity. Stage 2 is the cheapest test of the main idea, that content must
enter the coupling. Stage 3 follows the biology most closely and costs the most. The
waves review of the project (`docs/traveling_waves_evidence.md`) does not permit a
build on the map before a measurement, and stages 1 and 2 give that measurement.

## What I do not propose

- A phase reset on an event for the 5 oscillators. A reset at each episode start was
  tested and gave no content (`docs/results/oscillator_frozen_2026_09.md`).
- A change of a default. Each stage stays behind a flag until a gate passes at fresh
  seeds and the owner decides.
- A new score for any indicator. RPT-2 and GWT-2 stay PARTIAL. Phi-1 stays closed.

## Decision for the owner

1. Approve stage 1 and stage 2 in this order, or go to stage 3 directly.
2. For stage 2, the relation of two contents. My proposal is the agreement of place on
   the tectum map for vision and hearing. For memory and body I have no proposal yet,
   and I would leave their coupling as it is in the first test.
