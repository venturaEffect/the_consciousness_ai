# Binding stage 3: one oscillator for each place on the tectum map

Date: 2026-10-14. Status: proposal for the owner. The owner decided on 2026-10-14 that
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
