# Binding stage 3: grouping the whole object on the map

Date: 2026-10-15. Status: review for the owner, before any build. The owner chose on the
same day to continue with the oscillators by grouping the whole object (option 2 after
the FAILED second gate of the adapted form).

## The problem that this answers

Source: `docs/results/binding_map_stage3_2026_10.md` (on the branch of PR 51 until it is
merged). The adapted form failed its second gate on P. A read-only check of the related
steps of its 6 seeds showed that the selected group touched the light at 0.705 to 0.910
of the steps, held half or more of the light cells at only 0.179 to 0.550 of the steps,
and was about half light and half floor. So the map finds the light but selects a band
across its edge. The drive is a centre against surround contrast, and such a contrast is
strong at the edge of a large uniform area and weak inside it.

## Project documents read

| Document | What it adds |
|---|---|
| `docs/decisions/2026_10_14_map_binding_drive.md` | The salience drive. Candidate K3 (segments by feature similarity, after Wang and Terman 1997) made one segment of almost the whole map offline, with a cosine of 0.9 on the encoder features. Its control failed, so that number decides nothing |
| `docs/decisions/2026_10_14_map_binding_adaptation.md` | The adaptation that the adapted form uses |
| `docs/decisions/2026_10_14_binding_stage3_map_oscillators.md` | The approved design. An object is a group of neighbouring active cells. Liboni and colleagues (2025) separate foreground and background first, then objects |
| `docs/sensory_selection_evidence.md` | Knudsen 2018, as read there. The midbrain selects across space. Selection across objects is assigned to forebrain networks |
| `docs/traveling_waves_evidence.md` | Lists Roelfsema 2023 by its record only |

## Sources added

| Source | Read | Statement used here |
|---|---|---|
| Roelfsema, Neuron 111, 1003 (2023), perspective | Summary, the sections on incremental grouping, on synchrony and on the time course, from the text of the article | Features of an unfamiliar object are grouped by "incremental grouping". Enhanced activity starts at one part and spreads to parts that are connected, collinear or similar in colour and texture, through horizontal and feedback connections. One object is grouped at a time. The author concludes that enhanced firing binds and that synchrony does not. His evidence includes that gamma synchrony decays over a few millimetres of cortex and that it did not follow grouping in the curve tracing task |
| Jeurissen, Self and Roelfsema, eLife 5, e14320 (2016) | Main text, through a tool summary (about 70 percent of the page) | Humans judged if two points lay on one region. Reaction times fitted a model in which grouping spreads from one point through the inside of the region, faster where the region is wide (41 to 86 percent of the variance, about 22 to 31 ms per step). A model that fills inward from the edges predicted the wrong direction (r -0.30) |
| Self, van Kerkoerle, Supèr and Roelfsema, Curr. Biol. 23, 2121 (2013) | Abstract, through a search summary | In monkey V1 the edges of a figure are detected first (about 70 ms, horizontal connections), and about 30 ms later feedback fills the whole figure with enhanced activity |
| Chen, Wang and Liu, IEEE Trans. Neural Netw. 11, 1106 (2000) | Read for the drive review | In the image version of LEGION, the link between two neighbours is strong when their values are similar. A group needs a leader inside a homogeneous area |
| Conroy, Leite, Adeli, Zelinsky and McPeek, Vision Sciences Society (2025), abstract | A search summary of the abstract | In the monkey superior colliculus, activity rose over the whole of an extended object when a saccade went to one part of it, and the signal differed when a small gap cut the object. The authors leave open if the spread comes from upstream areas or from inside the colliculus |

Summary of the sources. An object is grouped by a spread of activity from one selected
point to neighbours that are connected and similar. The spread goes through the inside of
the object and is fastest in wide, uniform parts. In cortex this is shown in detail. In
the colliculus one abstract reports a spread over a connected object.

A tension that the owner should know. Roelfsema (2023) argues that this spread is a rise
of firing rate and that synchrony does not bind. LEGION and the owner's design use
synchrony. The grouping rule is the same in both views (connected and similar
neighbours), and option B computes only the result of the grouping. Option A, the full
oscillator equations, is the part that would differ.

## Designs compared

All designs keep the salience drive, the adaptation and the selection of the group with
the most drive. They change which cells join the selected group. Values marked assumed
are not fitted to data.

| Design | What changes | Fit with the sources | Assumed values | Known risk |
|---|---|---|---|---|
| A. Spread from the selected group | The selected group is the seed. Neighbours (4 per cell) join when their features are similar to the cell next to them in the group, and the spread repeats until no cell joins | High. A spread from one selected part through connected and similar neighbours (Roelfsema 2023, Jeurissen 2016, the colliculus abstract). The recruitment of similar neighbours by a leader in LEGION | a similarity threshold | Encoder features are similar almost everywhere (K3 offline). The similarity must be measured against the scene, for example on features minus their map mean |
| B. Fill the inside of the selected edge | Cells enclosed by the selected group join it | Medium. Edge first, then filling of the figure (Self 2013). The inward filling model predicted the wrong direction for human grouping (Jeurissen 2016) | none | Fails when the light is cut by the border of the view, because the edge is then not closed |
| C. Segments of the whole map first | Every cell is linked to similar neighbours, all segments form, then the segment with the most drive is selected (grey-level LEGION, as K3) | Medium. LEGION forms all segments in turn. Roelfsema 2023 reports grouping of one object at a time | a similarity threshold, a leader rule | Same feature risk as A. The floor becomes a large segment that can win |
| D. A wider centre of the salience drive | The centre Gaussian of the drive is wider, so a large area has contrast inside too | Low. Larger fields exist in deeper layers, but no source asks for this change | the new width | Chosen after seeing the failure. Small objects lose contrast |

## Recommendation for the owner

A. It is the form of grouping that the sources describe most directly. It starts from the
place that the map already selects, which keeps the midbrain selection by place, and it
then adds the connected and similar rest of the object. It adds one assumed value. The
similarity must be measured against the scene, because the raw encoder features were
similar almost everywhere.

## Test for the next runs

The checks of option B stay as they are, with fresh seeds and 40 episodes per run. The
seeds and the full gate will be stated in the result document before the runs.

## What this review does not establish

- No design was tried on the agent.
- The cortical sources are about V1 and V4 and human behaviour. For the colliculus there
  is one abstract, read through a summary.
- The similarity of encoder features after training is not known. Option K3 used the raw
  features and linked almost all cells, and its offline control failed.
- Roelfsema (2023) is one author's perspective. Supporters of synchrony are not reviewed
  here.

## Diagnosis after the owner chose A, before any build

Read-only, on the episodes with saved weights of seeds 146, 157 and 159 (episodes 0, 10,
20, 30 and 39), related steps only. The encoder was rebuilt from the weights saved with
each episode, so its features match the run only approximately (the drive review measured
a cell overlap of 0.90 to 0.97 for the recorded groups). The seed is the recorded selected
group. Two cells link when the cosine of their features minus the map mean is 0.9 or more.
This is a diagnosis of the design and not a test.

| Seed | Related steps | Recorded group, median cells, P | A, spread from the whole group, median cells, P | A with a background layer first, median cells, P |
|---|---|---|---|---|
| 146 | 172 | 37, 0.058 | 238, 0.000 | 54, 0.000 |
| 157 | 65 | 53, 0.908 | 97, 0.877 | 90, 0.877 |
| 159 | 199 | 22, 0.126 | 93, 0.095 | 93, 0.126 |

FAILED, both forms. Spread from the whole selected group reached 128 or more cells at
0.971 of the steps of seed 146, because the group lies across the edge of the light and
its floor cells link to the rest of the floor. In the second form the background is the
linked area with the most cells on the map border (after Liboni and colleagues 2025, who
separate foreground from background first). At seed 146 that area was mostly light at
0.983 of the steps, because the agent often stands on the light and the light then fills
most of the view.

So design A cannot be built as written. The selected place is an edge, and an edge belongs
to two surfaces. Choosing the surface that owns the edge is the figure and ground problem,
which no design of this review covers.

## Owner decision after the diagnosis, with a stop rule

2026-10-15. The owner chose an in-depth review of figure and ground (border ownership),
including the sound as a cue for the side of the edge, then one design. Stop rule, stated
now: if that design fails its gate on fresh seeds, the map line of binding stage 3 is
documented as a negative result and stopped. An UNDECIDED or VOID result goes back to the
owner.

## Review of figure and ground (border ownership)

### Sources

| Source | Read | Statement used here |
|---|---|---|
| von der Heydt, Front. Psychol. 6, 1695 (2015), review | All sections and figure legends, through a tool summary | Border ownership cells, especially in V2, respond to an edge more strongly when the figure lies on their preferred side. The signal appears 10 to 35 ms after the response onset. Cues are closure, compact shape, T junctions, disparity and motion. A proposed circuit has grouping cells that sum edge signals on ring templates and feed back to the edge cells that drove them. Attention acts on the grouping cells. The grouping cells are postulated and not yet identified. The review names no role of the superior colliculus |
| Zhou, Friedman and von der Heydt, J. Neurosci. 20, 6594 (2000) | Abstract, through a search summary | Border ownership modulated more than half of the neurons of V2 and V4 and 18 percent of the neurons in the top layers of V1 |
| Mihalas, Dong, von der Heydt and Niebur, PNAS 108, 7583 (2011) | Main text, through a tool summary (the equations are images and were not read) | A model with edge cells, border ownership cells and grouping cells with ring shaped fields at three scales (radii 2, 4 and 8). Attention enters at the grouping cells and takes the place and the size of the object |
| Martin and von der Heydt, J. Neurosci. 35, 6860 (2015) | Abstract, through a search summary | Border ownership cells fed by one grouping circuit showed more spike synchrony when one object drove them, and other pairs did not. The rise was small, and the authors conclude that binding is coded by response strength through feedback |
| Gori, Burr and Campus, Curr. Biol. 34, R235 (2024) | Full text | A conversation raised the share of time that people saw faces in the vase and faces figure from 0.51 to 0.78, and to 0.84 with a blurred image. Intention alone had no significant effect. The cue was the meaning of the sound, not its place. The figure and ground cues they list are size, symmetry, convexity and enclosure |

### Diagnosis of a simple grouping cell, before any build

Read-only and approximate, on the same episodes as above. The edge drive is the salience
drive without the adaptation of the run, whose state cannot be rebuilt offline. A grouping
cell at a centre and a radius (2, 4 or 8 cells) takes the mean edge drive on the ring of
cells at that distance. The selected cell is the one with the largest grouping drive
combined with the audio drive at its centre, and the group is its disc.

| Seed | Related steps | Group, median cells | P | Holds half the light | Radius chosen |
|---|---|---|---|---|---|
| 146 | 172 | 13 | 0.105 | 0.012 | always 2 |
| 157 | 65 | 13 | 0.892 | 0.031 | always 2 |
| 159 | 199 | 13 | 0.372 | 0.171 | always 2 |

FAILED. The smallest ring always won, because a small ring can lie inside the band of
contrast, which is 2 to 3 cells wide. In the source model the grouping cells sum thin,
oriented edge signals with a side of figure, which this map does not have.

### What follows

The mechanism that the sources describe for choosing the side of an edge needs oriented
border ownership cells and grouping cells at several scales. That is a larger build with
more assumed values, and its key cell type is postulated, not identified. Three designs
for the whole object (spread, background first, simple grouping cell) failed their
diagnosis on recorded episodes before any build.

## Owner decision. The map line is stopped

2026-10-15. After the three failed diagnoses above, the owner chose to stop the map line of
binding stage 3 and to document it as a negative result. The stop rule above names a
failed gate. Here no design reached a gate, so the stop is the owner's decision and not
the outcome of the rule.

What the map line established, with sources in `docs/results/binding_map_stage3_2026_10.md`.

1. A map of the tectum can make the binding change which sense wins. The salience form and
   the adapted form changed the leader at 0.05 or more of the steps at 7 of 9 seeds.
2. None of the five forms passed its gate. The adapted form failed its second gate on P.
3. The selected place on this map is the edge of the light, and an edge belongs to two
   surfaces. Three designs that add the rest of the object failed before any build.

What stays in the code. `--map-binding` (off, limit, salience, adapted) and the probes stay,
default off, so the documented runs can be repeated. Option A, the full oscillator
equations, was not built and not run.

What the sources add for any later work. The binding studies read here (Roelfsema 2023,
Martin and von der Heydt 2015) find that binding is carried by stronger firing through
feedback, and that synchrony is small and follows from shared input. A later attempt on
the map needs oriented border ownership signals and grouping at several scales, or a test
bed in which the light is a small object.

## The map line was reopened (2026-10-15)

On the same day the owner reopened the map line with a new test bed, a second object that
looks like the light (`docs/decisions/2026_10_15_visible_decoy.md`). Before that, a choice by
chance from the size of light clusters, with the sound added by the rule of the colliculus,
failed its diagnosis on the recorded runs. The gate in the new test bed FAILED on P2 at 3 of 3
seeds (`docs/results/visible_decoy_gate_2026_10.md`). A reading after that gate, not a check,
found the selected group on or at the edge of the object with the louder tone at most counted
steps, and rarely at the other object.
