# Binding stage 3: a second object that looks like the light

Date: 2026-10-15. Status: rules written before any run. The owner chose this option on
2026-10-15, after a sixth design of the map failed its diagnosis (last section).

## Why the test bed changes

The dark room has one object, the light. The decoy of stage 1 is a tone with no image. The
design document of the oscillator layer said so already
(`docs/decisions/2026_10_14_oscillator_binding_design.md`). This room cannot show selective
binding with any layer. Five forms of the map and one more design failed in this room. Each
failed at a new place, and the room stayed the same.

With a second object that looks like the light, the eye cannot tell the two objects apart.
Only the tone can. So the binding has a real job. It must join the louder tone to its own
object.

## The change

Behind a default-off flag, `--dark-room-decoy-visible`, which needs `--dark-room-decoy-sound`.

| Part | Rule |
|---|---|
| Image | The decoy is drawn in the light colour and at the light size, at the place of its tone |
| Reward | None. Only the light gives reward, by the code that was there before |
| Place | From the decoy generator. A direction and a distance from the light, 120 to 176 room pixels, drawn again until the place is inside the room. At 120 (two light radii plus two agent radii) the discs do not overlap and the agent cannot touch both. At 176 (two view radii plus two light radii) the agent between them sees both. The first diagnosis used a place anywhere in the room (see below) |
| Records | `_truth_decoy_in_view`, by the rule of `_truth_light_in_view`. No agent code reads `_truth_` keys |
| Flag off | No change. The agent, the light and a hidden decoy keep their draws |

The plan said "two object widths" (160 pixels). I changed it to 120 before any run. With 160,
both objects are in view only when the agent stands near the middle, and the diagnosis below
could not pass.

## Sources

| Source | Read | Statement used here |
|---|---|---|
| Stein, Meredith, Huneycutt and McDade, J. Cogn. Neurosci. 1, 12 (1989) | Abstract | A tone at the place of a dim light raised the approach to it. A tone at another place lowered it |
| Plank and colleagues (PubMed 21425399) | Summary of the abstract | One sound at the place of the target or of a distractor changes speed and accuracy |
| Van der Burg, Olivers, Bronkhorst and Theeuwes, J. Exp. Psychol. Hum. Percept. Perform. (2008) | Abstract and later reviews | A tone in time with a change of one object makes that object easier to find, also when the tone has no place |

## Diagnosis before the gate (Part 3)

### First diagnosis, decoy anywhere in the room. FAILED

Three runs, seeds 171, 172 and 173 (commit 833ee45). The Gate B7 command with `--dark-room-decoy-sound
--dark-room-decoy-level 0.5,3.0 --dark-room-decoy-visible`, map binding off, 10 episodes, one
run at a time. Read by `scripts/analysis/probe_louder_object.py`.

Rule: each run has at least 100 steps with both objects in view and one tone louder. If one
run has fewer, the test bed cannot test binding, and I stop and report.

| Seed | Steps with both in view | Light louder | Decoy louder |
|---|---|---|---|
| 171 | 27 | 26 | 1 |
| 172 | 122 | 122 | 0 |
| 173 | 16 | 14 | 2 |

FAILED at seeds 171 and 173. The control with the flag off gave the md5 of the earlier
control, `7825766f42d3d85b032f8028c94dc1e0`. Two causes, from the step records. The two places
were often far apart. In the episodes with the loud decoy (level 3.0) both objects were in view
at 0, 0 and 2 steps, so the decoy was almost never the louder one with both in view.

### Second diagnosis, decoy 120 to 176 pixels from the light

Three fresh runs, seeds 174, 175 and 176, with the same command and the place rule above.

Rule: each run has at least 100 steps with both objects in view and one tone louder, and at
least 30 of them with the decoy louder. If one run has fewer, I stop and report.

## Gate (Part 4)

Three fresh seeds, 181, 182 and 183, each run with the adapted map binding on and once with it
off. 40 episodes each, one run at a time. The adapted form is in PR 51. The gate runs start
after that PR is on main.

| Check | Statement |
|---|---|
| P2 | At steps with both objects in view, the map selects the object with the louder tone at more than half of the steps, at 3 of 3 seeds |
| G1 | The joint alignment of vision and audio is higher at related than at unrelated steps, p < 0.05, at 3 of 3 seeds (as in stage 2) |
| G3 | The binding changes the leader at 0.05 or more of the steps, at 3 of 3 seeds |
| Task | The share of steps in the light, binding on against binding off, at each seed. A hypothesis only, one comparison |

PASS when P2, G1 and G3 all hold. FAIL when one fails at 2 or more seeds. Completion is checked
by the row counts of the csv files.

## The design that failed before this one

On 2026-10-15 the owner proposed a choice by chance, set by the size of the cluster of light
pixels, with the sound added by the rule of the colliculus (Ohshiro, Angelaki and DeAngelis,
Nat. Neurosci. 14, 775, 2011). A replay of the six recorded gate runs tested it before any
build. Rule, written first: at every seed, the gain of related over unrelated steps at the same
light size is above 0.05, and the gain is larger for a small light than for a large light.
FAILED. It held at 1 of 6 seeds (159). The gain was above 0.05 at 4 of 6 seeds (139, 140, 157,
159). The gain was larger for a small light at 2 of 6 seeds (138, 159). The light was the
largest bright cluster at 0.632 to 0.989 of the steps.

## Second diagnosis: FAILED (commit ba589dc)

| Seed | Steps with both in view | Light louder | Decoy louder | Steps at a wall, of 2000 |
|---|---|---|---|---|
| 174 | 5 | 0 | 5 | 1885 |
| 175 | 12 | 0 | 12 | 1830 |
| 176 | 25 | 21 | 4 | 1809 |

FAILED at all 3 seeds. The cause is the agent, not the places. In 5 of the 6 diagnosis runs
the agent touched a wall at 1672 to 1890 of 2000 steps (the `collision` record). It sees an
object only when the object is near its wall. The one run in which the agent left the walls
(seed 172, 212 wall steps) is the one run that had enough steps. Both controls held. With
the flag off the md5 of `metrics.csv` was the earlier control md5, also with
`--record-episodes none`.

## Third diagnosis: the agent starts between the two objects

The owner chose this on 2026-10-15. Behind the default-off flag `--dark-room-start-between`,
which needs `--dark-room-decoy-visible`. Each episode starts with the agent midway between
the light and the decoy, so both are in view at the start. The agent place is still drawn
first from the global stream, so the light and the decoy keep their places.

Three fresh runs, seeds 177, 178 and 179, 10 episodes each, the command of the first
diagnosis plus `--dark-room-start-between`.

Rule, scaled from the gate length of 40 episodes to 10: each run has at least 25 steps with
both objects in view and one tone louder, and at least 8 of them with the decoy louder. If
one run has fewer, the agent leaves the objects too fast for any binding test in this room,
and I stop and report.

## Third diagnosis: PASSED (commit 14e4824)

| Seed | Steps with both in view | Light louder | Decoy louder |
|---|---|---|---|
| 177 | 58 | 26 | 32 |
| 178 | 162 | 78 | 84 |
| 179 | 110 | 51 | 59 |

PASSED at all 3 seeds. The control with the flags off gave the earlier control md5 again.
The gate uses `--dark-room-start-between`. The gate runs with binding off use
`--record-episodes none`, because the task check reads `episodes.csv` only and the second
diagnosis showed that `metrics.csv` does not change without the record.
