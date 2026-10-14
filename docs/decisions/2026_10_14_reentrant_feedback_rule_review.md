# Review of the reentrant feedback rule against the biology

Date: 2026-10-14. Status: proposal. The owner asked for this review before any change to
`top_winner`. No code changes with this document.

## The rule in the code

`SensoryTectum.receive_broadcast` and `AuditorySpecialist.receive_broadcast` run in each
cycle of `ReentrantProcessor.settle`.

| Step | What the code does |
|---|---|
| Error | pe = \|broadcast - own vector\| / \|own vector\| |
| pe above 0.3 | bid + 0.1 * pe, at most 1.0 |
| pe at or below 0.3 | bid * (1 - 0.1 * pe) |
| No tensor in the broadcast | vision bid * 0.95, audio bid unchanged |

So the module whose vector is in the broadcast keeps its bid or lowers it a little. The
module whose vector is not in the broadcast raises its bid. The code comments describe
this as "not in the spotlight, increase bid".

## What the project measured

| Result | Source |
|---|---|
| With `top_winner`, when audio leads, the cycles raise the vision bid by 0.359 to 0.401, and 53 to 67 percent of the audio leads flip to vision | `docs/results/top_winner_diagnosis_2026_10.md` |
| With equal vector sizes the rise is 0.154 to 0.236, and 15 to 34 percent of the vision leads now flip to audio | the same document |
| With equal vector sizes the vision takeover is gone at 3 of 3 fresh seeds | the same document |

So the size of the vectors made the effect larger. The direction of the effect comes
from the rule. With any vector size, the rule moves bids toward the module that lost.

## What the sources say happens to the signals that are not selected

Each item says how much was read.

| Source | Read | Statement |
|---|---|---|
| Desimone and Duncan, Annu. Rev. Neurosci. 18, 193 (1995), and Desimone (1998) | Abstract and secondary summary | Stimuli in view compete. A bias resolves the competition for the relevant stimulus, and the resolution suppresses the representations of the others |
| Knudsen, Trends Neurosci. 41, 789 (2018), and Mysore and Knudsen (2013) | Main text and author manuscript, see `docs/sensory_selection_evidence.md` | In the midbrain one nucleus (Imc) inhibits all places except the strongest one. Another (Ipc) amplifies and synchronises the selected place. Adaptation lowers the response to a repeated stimulus, so the winner changes over time |
| Dehaene and Changeux, Neuron (2011) | Abstract | Conscious access shows late amplification of the relevant activity, long-distance synchrony and ignition of a large network |
| Rao and Ballard, Nat. Neurosci. 2, 79 (1999) | Abstract-level summary | Feedback carries a prediction. The feedforward path carries the residual error. The error is a signal that goes up. It is not a priority for access |

| Property | The sources | The code |
|---|---|---|
| The selected signal | Amplified and held | Kept, or lowered a little |
| The signals that are not selected | Suppressed | Raised by 0.1 * pe |
| How the winner changes | Over time, by adaptation or by a new salient input | Within one step, by the feedback cycles |
| Where an error goes | Up, as a residual signal | Into the bid of the module |

## Reading

The rule is opposite to the sources for the signals that are not selected. The sources
describe amplification of the winner and suppression of the rest. The rule raises the
rest. The predictive coding source gives a role to a mismatch, but that role is a
signal that is sent up, and not a higher bid for access. So none of the four sources
supports the present rule. This is my reading of abstracts and reviews, and a test on
the agent decides if a new rule works.

## Options for a new rule

Each is behind a default-off flag, with the baseline bit-identical, and with
predictions written before the runs.

| Option | Rule | Basis | Risk |
|---|---|---|---|
| A. Amplify and suppress | In the cycles the bid of the module in the broadcast rises by a gain g, and the other bids fall by g | Imc and Ipc in the midbrain, biased competition | A winner can lock in. The project saw vision lock in before |
| B. Error goes up, not into the bid | The mismatch is kept as a surprise signal of the module and does not change its bid in the cycles | Predictive coding | The cycles then change no bid, so the loop has less effect |
| C. Adaptation | The bid of the module that won falls a little at each step that it wins, and recovers when it loses | Adaptation in the midbrain | Alone it does not resolve a competition |

My recommendation is A with C. A gives the selection that the sources describe. C lets
the winner change over time and is the guard against lock-in. The test uses the gate
criteria of the project (competition, silence, selectivity and the task link) at 10
fresh seeds, with both broadcast merges.

A and stage 3 of the binding design describe the same circuit. Ipc amplifies and
synchronises the selected place. So the gain of A can later come from the synchrony of
the oscillators on the map. The design of stage 3 will take this into account.

## Decision for the owner

Approve A with C as the rule to build and test, or choose another option. No default
changes and no indicator moves with this document.
