# Binding stage 3: which cells of the map are stimulated

Date: 2026-10-14. Status: written after the option B runs and before any new run. It
states the candidates and the rule that chooses between them, before the comparison is
computed. Source of the failure: `docs/results/binding_map_stage3_2026_10.md`.

## The failure that this answers

Option B grouped every cell with a combined drive of at least 0.05 into connected areas.
In the three runs (seeds 132 to 134) the map held one group at 0.951 to 0.994 of the
steps. That group had a median of 111 to 115 of the 256 cells and held a median of 0.980
to 0.987 of the drive of each sense. So vision and audio got a factor near 1.5 at almost
every step, and nothing was selected.

Two causes, both measured offline on the recorded frames with the encoder weights saved
with each episode (episodes 0, 10 and 19 of each run).

1. The visual drive is the feature norm scaled between the map median and the map
   maximum. By that definition about half of the cells have a drive above 0. At the
   threshold 0.05 the vision part alone was active at a median of 68 to 105 cells. The
   audio area of sigma 2 cells is above 0.05 within a radius of about 4.9 cells, about 75
   cells.
2. The feature norm marks the light before training and less after it. The area under
   the curve of the drive, light cells against the other cells, had a median of 0.976 and
   0.999 in episode 0 (seeds 132 and 133) and 0.314 to 0.714 in episodes 10 and 19.

In the agent-centred view the light is a disc that covers up to about 45 percent of the
frame, the walls are grey, the floor is black, and the agent's own ring is always at the
centre (recorded frames of seed 133, episode 10).

Both causes also apply to option A, the full oscillator equations, because option A uses
the same stimulated cells. So the drive must change before either option can be tested.

## Project documents read

| Document | What it adds |
|---|---|
| `docs/decisions/2026_10_14_binding_stage3_map_oscillators.md` | The approved design. The read-out is the synchronised group with the largest summed drive |
| `docs/sensory_selection_evidence.md` | Knudsen 2018, as read there. The priority of a place combines physical salience (size, luminance, motion, novelty) with task relevance. Other senses add strength at a place and do not add identity. One nucleus (Imc) suppresses all places except the strongest. Adaptation lowers the response to a repeated stimulus |
| `docs/results/binding_map_stage3_2026_10.md` | The option B gate and its result |
| `models/core/sensory_tectum.py`, `TopographicMap` | The visual map is the anchor and the sound is placed on it with a blur |

## Sources added

| Source | Read | Statement used here |
|---|---|---|
| White, Berg, Kan, Marino, Itti and Munoz, Nat. Commun. 8, 14263 (2017) | Main text and methods, through a tool summary | The activity of visual neurons of the superficial colliculus was predicted by a salience model at 27 of 34 neurons. The model computes centre against surround contrast with a difference of Gaussians. The surround is 6.7 times the centre for luminance and colour features |
| Wang, Sarnaik, Rangarajan, Liu and Cang, J. Neurosci. 30, 16573 (2010) | Results and discussion, through a tool summary | Superficial colliculus neurons of the mouse prefer flashes of 6 to 10 degrees radius. The response to a 30 degree flash had a median of 0.02 of the peak. So a large uniform area drives these cells very little |
| Chen, Wang and Liu, IEEE Trans. Neural Netw. 11, 1106 (2000) | Model and algorithm sections, text extract (the equations did not extract) | Restates the image version of LEGION by Wang and Terman (Neural Comput. 9, 805, 1997). One oscillator per pixel, 8 nearest neighbours, one global inhibitor. For grey-level images every oscillator is stimulated, and the weight between two neighbours falls with the difference of their values. A group needs a leader, a cell at the centre of a large homogeneous area. Areas without a leader stay in the background |
| Wang and Terman, Neural Comput. 9, 805 (1997) | Abstract only. The full text was not available | The lateral potential lets only cells with strong neighbour links become leaders, and noisy areas become the background |
| Quiles, Wang, Zhao, Romero and Huang, Neural Netw. 24, 54 (2011) | Abstract only. The full text is not open | A salience map, a LEGION segmentation and an object selection stage. The system selects salient objects, not salient locations, one at a time |

## Candidates

All candidates use the encoder features of the tectum map, as the run did. The audio
drive stays a Gaussian area of sigma 2 cells. The binding factor stays 1 + 0.5 times the
share of each sense's drive inside the selected group. Values marked assumed are not
fitted to data.

| Candidate | Visual drive | Stimulated cells and links | Selected group | Sources |
|---|---|---|---|---|
| K0, the built rule (control) | Feature norm, median to maximum | Combined drive of 0.05 or more, 4 neighbours | Most summed drive | none for the threshold |
| K1, half maximum (control) | As K0 | Combined drive of 0.5 or more, 4 neighbours | Most summed drive | the half maximum convention only |
| K2s, salience | Contrast of a centre Gaussian (sigma 1 cell, assumed) against a surround Gaussian 6.7 times wider, scaled by its map maximum | Combined drive of 0.5 or more, 4 neighbours | Most summed drive | White 2017, Wang 2010, Wang and Terman 1994 |
| K2m | As K2s | As K2s | The group that holds the strongest cell | as K2s, and Imc in Knudsen 2018 |
| K3s, segments with salience | As K2s | Every cell takes part. Two of the 8 neighbours link when the cosine of their feature vectors is 0.9 or more (assumed, the value already used by the read-out of the phase version). A leader is a cell linked to all its neighbours. A group is a linked area with a leader. Other cells are background | Most summed combined drive | Chen, Wang and Liu 2000, Quiles 2011 |
| K3m | As K3s | As K3s | The group that holds the strongest cell | as K3s, and Imc in Knudsen 2018 |

## The comparison and the choice rule. Written before the comparison

Data. Episodes 0, 10 and 19 of `runs/s3b_s132`, `runs/s3b_s133` and `runs/s3b_s134`
(the episodes with saved weights), with the recorded frames, the recorded sound
direction and the step record. Related steps as in `probe_map_selection.py`.

Control first. K0 recomputed offline must give the recorded group (the same cells) at
0.9 or more of these steps. If not, the offline recomputation is not faithful and the
comparison is VOID.

A candidate qualifies when all three hold.

1. Two or more groups at 0.5 or more of the steps that have a group.
2. The selected group is on the light at more than half of the related steps, pooled.
3. The same holds in each run that has 50 or more related steps in these episodes.

K0 and K1 cannot be chosen. They are controls. Among the qualifying candidates the
highest pooled share on the light is chosen. A candidate within 0.05 of that share is
preferred when it has fewer assumed values (K2 before K3), and then when it keeps the
approved read-out (summed drive before strongest cell).

If no candidate qualifies, no agent run follows, and the owner decides.

The chosen candidate then gets a new value of `--map-binding` and the pre-stated gate of
option B, unchanged, on fresh seeds. These episodes are old seeds, so the comparison is
a diagnosis and not a test.

## Change before the comparison. Two candidates removed

K2m and K3m are removed. The unit tests showed that they cannot separate groups. Each
drive is scaled so that its peak is exactly 1, and the either-one rule gives 1 at the
peak of either sense. So the group with the salience peak and the group with the sound
peak always tie, and the rule would select by label order. The comparison therefore has
the controls K0 and K1 and the candidates K2 (K2s above) and K3 (K3s above). The choice
rule is unchanged. This change was made before the comparison was computed.

## Comparison result. VOID

The control failed, so the comparison is VOID by its own rule and chooses no candidate.
Output: `runs/_audit_followup/oct13/map_drive_candidates.json`
(`probe_map_drive_candidates.py`, after its first launch FAILED with an index error,
because episode 0 has no sound entry at step 0. The fix finds each entry by its recorded
step, with a regression test.)

| Control, K0 offline gives the recorded group | Seed 132 | Seed 133 | Seed 134 |
|---|---|---|---|
| Share of steps with the same cells (0.9 needed) | 0.467 | 0.129 | 0.313 |

Cause, measured on seed 133 only. The step alignment is right. With the record of the
same step the share of same cells was 0.259, 0.041 and 0.076 in episodes 0, 10 and 19.
One step earlier or later it was 0.005 to 0.046. The mean overlap of the cells (shared
cells over all cells of the two groups) was 0.950, 0.901 and 0.965. In episode 0 the
share of same cells rose over the four quarters of the episode, 0.04, 0.04, 0.40 and
0.571. So the encoder changes during an episode, and the weights saved with the episode
are those of its end. The offline groups are close to the recorded ones, but they are
not the same cells.

The candidate values are listed for the record only and decide nothing.

| Pooled over the 9 episodes | K0 | K1 | K2 | K3 |
|---|---|---|---|---|
| Two or more groups, share of steps | 0.032 | 0.160 | 0.863 | 0.009 |
| On the light, share of 422 related steps | 0.867 | 0.867 | 0.872 | 0.308 |

Seed 133 has 408 of the 422 related steps, seed 132 has 14 and seed 134 has none, so the
values come almost only from one seed.

## What this review does not establish

The equations of Wang and Terman 1997 and the text of Quiles 2011 were not read. The
salience model of White 2017 uses six feature types, motion among them. The candidates
use one contrast on learned features, and no motion or adaptation. The agent's own ring
is always at the centre of the view. In the brain, adaptation lowers the response to a
stimulus that does not change. No candidate has adaptation, so the ring may win the
selection.
