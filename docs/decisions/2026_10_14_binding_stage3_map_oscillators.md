# Binding stage 3: one oscillator for each place on the tectum map

Date: 2026-10-14. Status: proposal for the owner, reviewed in depth on the same day (last section). The owner decided on 2026-10-14 that
the oscillators must work, in the order stage 1, stage 2, stage 3. Stages 2 and 2b did
not pass. No code changes with this document.

## What stages 2 and 2b showed

Source: `docs/results/binding_place_relation_2026_10.md`.

- A relation term in the coupling of the five workspace oscillators moves the layer away
  from the bids (R squared of `sync_r` on the bids 0.08 to 0.24, against 0.99 to 1.00).
- It does not make the binding selective at most seeds, and it changes the leader at
  few steps. Two forms of the relation gave this.
- The light is about 13 cells wide on the map. One peak cell and one sound cell cannot
  describe an object of that size. Five oscillators, one for each module, have no place
  at all.

## What the sources describe

| Source | Read | Statement used here |
|---|---|---|
| Sridharan, Boahen and Knudsen, J. Neurophysiol. 105, 2005 (2011) | Main text in part | A stimulus evokes gamma at its place on the tectum map, sharply tuned for place |
| Goddard, Sridharan, Huguenard and Knudsen, Neuron 73, 567 (2012) | Main text in part | A local excitatory and inhibitory circuit makes the rhythm. It synchronises one restricted column and lasts some hundred milliseconds after a brief input |
| Knudsen, Trends Neurosci. 41, 789 (2018) | Main text | One nucleus (Imc) inhibits all places except the strongest. Another (Ipc) amplifies and synchronises the selected place |
| Terman and Wang, Physica D 81, 148 (1995) | Summary of the abstract | Oscillators coupled to their neighbours, plus one common inhibitor, synchronise within each connected input region and desynchronise different regions |
| Miyato and colleagues, ICLR 2025 (AKOrN) | Abstract | Kuramoto units can use convolutional, that is local, connectivity |

So the form that the sources describe is a map of oscillators with local coupling and
one global inhibitor. An object is a group of neighbouring active cells that
synchronise. Two objects at different places do not synchronise.

## The design

Behind a default-off flag, `--map-binding on`. The present layer stays as it is when the
flag is off.

| Part | Design | Source |
|---|---|---|
| Units | One 2-D oscillator for each of the 16 by 16 cells of the tectum map | Sridharan 2011 |
| Drive | The activity of each cell: the visual activity (feature norm, scaled as in stage 2b) and the auditory activity on the same map (the cell of the sound with its blur, as the tectum already places it) | the code of `TopographicMap` |
| Coupling | Each cell is coupled to its 8 neighbours, with a strength equal to the product of their drives. So active neighbours synchronise and quiet cells do not take part | Terman and Wang 1995, Goddard 2012 |
| Global inhibitor | One term that pushes the phase of each active cell away from the mean phase of all active cells that are not its neighbours. So separate active regions desynchronise | Terman and Wang 1995, Imc in Knudsen 2018 |
| Read-out | The synchronised group with the largest summed drive is the selected place. The vision bid and the audio bid get their binding factor from how much of their activity lies in that group | Ipc in Knudsen 2018 |
| Reset | The phases of a cell restart when its drive rises from near 0, so a new stimulus starts a new rhythm | Goddard 2012 |

The five workspace oscillators stay for memory, body and semantic, which have no place
on the map.

## Gates

The gates G1 to G4 of stage 2, unchanged, at 3 fresh seeds on the stage 1 test bed (20
episodes, decoy levels 0.5 and 3.0). G1 reads the joint alignment of vision and audio
from the binding factors, as before, so the result is comparable with stages 2 and 2b.

One more check before the gates, on the same runs: the selected place is on the light
at more than half of the steps with the light in view and the light tone louder. The
reference is the light pixels of the recorded frames, which the re-analysis of stage 2
showed to be the reliable reference.

## Cost and risk

- Build: my estimate is 3 to 4 days with tests, then 3 runs of about 9 minutes each.
- The waves review of the project (`docs/traveling_waves_evidence.md`) does not license
  a wave build before a measurement. This design builds no wave. It is a map of
  oscillators with local coupling, which the owner decided on 2026-10-14 after stages 2
  and 2b. I state the tension here so that the decision is explicit.
- The global inhibitor adds a constant (its strength). It is assumed, and it is not
  fitted to the gate data.
- No default changes and no indicator moves until a gate passes at fresh seeds and the
  owner decides.

## Decision for the owner

Approve the design as it is, or change a part of it before the build starts.

## Review in depth (2026-10-14, after the first version)

The owner asked if this design is the best and most accurate one, and if the documents
and papers were read in depth. They were not. The first version used five sources, two
of them at abstract level only, and it did not compare designs. This section adds the
reading and the comparison. The first version above is unchanged, so the change is
visible.

### Project documents read for this review

| Document | What it adds |
|---|---|
| `docs/traveling_waves_evidence.md` | Section 1.9 names oscillator networks for segmentation, among them Liboni and colleagues (PNAS 2025). Section 2: the tectum ConvGRU is the only spatial recurrent substrate in the code. Section 3: the agent has no millisecond clock, so a phase quantity is in cycles per step and carries no Hz reading |
| `docs/isomorphic_visual_mapping_research.md` | Superior colliculus: the visual map is the anchor, the auditory map is calibrated to it, auditory fields are 40 to 80 degrees wide against 10 to 30 for vision, and only their centres must agree. The three rules of Stein and Meredith (space, time, inverse effectiveness) |
| `docs/biological_neural_architecture_research.md`, phase 2 | The original plan for binding: coincidence in time, with a Kuramoto model as the recommended start |
| `docs/results/tectum_wave_detection_2026_09.md` | The phase gradient in the ConvGRU state comes from the input path at 2 of 3 seeds, so propagation in that state is not shown |
| `docs/results/binding_place_relation_2026_10.md` | Stages 1, 2 and 2b, and the diagnostic: one peak cell and one sound cell cannot describe an object 13 cells wide |

### Sources added

| Source | Read | Statement |
|---|---|---|
| Miyato, Löwe, Geiger and Welling, ICLR 2025 (AKOrN), method and experiments | Main text in part, through a tool summary | Each position holds N-dimensional unit oscillators (N = 4 in the main results). The input enters as a stimulus vector that pulls each oscillator. The coupling is convolutional or attentive and is learned with a training objective. Objects are read out by clustering. Stated limit: a unit oscillator cannot signal the presence of a feature |
| Liboni, Budzinski, Busch, Löwe, Keller, Welling and Muller, PNAS 122, e2321319121 (2025) | Main text in part, through a tool summary | Complex-valued units on a grid, Gaussian distance coupling, and the input sets the frequency of each unit. Two layers: foreground against background, then one wave for each object. The dynamics are solvable in closed form. About 93 percent of pixels correct on 1,000 two-shape images and about 86 percent on three-shape images. Stated limit: it works only for objects in a range of sizes that the coupling width sets |
| Lowet and colleagues, PLoS Comput. Biol. (2015) | Abstract | In cortex the gamma frequency changes with the input strength and differs between nearby sites. In a model with local coupling this input dependence organises the synchrony instead of preventing it |
| Romano and colleagues, Neuron 85, 1070 (2015) | Abstract | In the zebrafish tectum the activity groups are compact on the map, group cells with similar function, act all-or-none, and are shaped by competition between groups |
| Sridharan, Boahen and Knudsen (2011), read again | Main text in part | In the owl tectum the gamma frequency and coherence do not change much with modality or contrast. So the input-frequency principle of cortex is not shown for the tectum |

### Four designs compared

| Design | Fit with the tectum sources | Size of the object (the failure of stage 2) | Needs a training objective | Cost |
|---|---|---|---|---|
| A. Local coupling weighted by the drive, plus one global inhibitor (this document, after Terman and Wang) | High. Local excitatory and inhibitory circuit (Goddard 2012), global inhibition (Imc, Knudsen 2018), compact all-or-none groups that compete (Romano 2015) | Groups form by connected active regions, not by a fixed width, so a large object is one group | No | 3 to 4 days |
| B. Input sets the frequency, Gaussian coupling, two layers (Liboni 2025) | Medium. The principle is shown for cortex (Lowet 2015). In the owl tectum the frequency does not change much with contrast | A fixed coupling width sets the range of sizes. That is the cause of the stage 2 failure | No | About 3 days. Solvable in closed form, which helps the tests |
| C. Convolutional AKOrN with learned coupling | Medium | Learned | Yes. The agent learns online by reward, so a new objective is a large change | More than a week |
| D. Use the existing ConvGRU state, with no new oscillators | Low. It does not oscillate, and propagation in it is not shown at 2 of 3 seeds | Not applicable | No | 1 to 2 days |

### Verdict of the review

Design A stays the best match for the tectum of the four. It is the only one that gives
compact groups that compete, as the tectum sources describe, and that groups an object
by its connected area rather than by a fixed width. B is the stronger choice for a
cortex-like map, and its closed-form solution is an asset that A does not have.

Three corrections to the first version follow from the review.

1. **The auditory drive is a broad area, not one cell.** Auditory fields are 2 to 4
   times wider than visual ones, so the drive of the sound covers about 3 to 5 cells
   around the direction of the sound, centred on it.
2. **Time.** The oscillators run a fixed number of iterations inside each environment
   step. A phase quantity is in cycles per step, with no Hz reading. The phase of a cell
   restarts when its drive rises from near 0.
3. **Evidence status.** A role of tectal gamma in selection is proposed by its authors
   and not tested. Binding by synchrony is disputed in cortex. So stage 3 tests a
   hypothesis, and its gates decide.

What this review does not establish. I cannot claim that A is the best possible design.
The full texts of Terman and Wang, Romano and Lowet were not read, only summaries. No
design was tried on the agent. The comparison is a judgement on the sources listed here.

## Status after the tests (2026-10-15)

Option B was tested in five forms and none passed its gate. The owner stopped the map line
on 2026-10-15, and option A was not built. Results:
`docs/results/binding_map_stage3_2026_10.md`. Reviews of the drive, of adaptation and of
grouping the whole object: `docs/decisions/2026_10_14_map_binding_drive.md`,
`docs/decisions/2026_10_14_map_binding_adaptation.md` and
`docs/decisions/2026_10_15_map_binding_object_grouping.md`.

The owner reopened the map line on the same day with a visible decoy. Its gate FAILED on P2
at 3 of 3 seeds: `docs/decisions/2026_10_15_visible_decoy.md` and
`docs/results/visible_decoy_gate_2026_10.md`.
