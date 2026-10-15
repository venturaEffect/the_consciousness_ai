# The motor map of the tectum, and the G3 test of binding

Date: 2026-10-15. Status: rules written before any run. The owner approved this plan on
2026-10-15.

## Why

The map binding now selects the object with the louder tone (P2 at 3 of 3 seeds,
`docs/results/map_binding_trim_gate_2026_10.md`). But no binding test could show an effect on
behaviour, because the default policy does not react to its input
(`docs/results/action_core_relay_defect_2026_10.md`): its last layer is never trained, and the
agent drifts in one fixed direction for each seed. G3 of the binding design
(`docs/decisions/2026_10_14_oscillator_binding_design.md`) asks that "a disruption of synchrony
lowers a task measure". That needs a path from the selected place to the movement.

## What biology has and our tectum lacked

| Source | Read | Statement used here |
|---|---|---|
| Isa, Marquez-Legorreta, Grillner and Scott, Curr. Biol. 31, R741 (2021) | Review, first 100,000 characters | The deep tectal layers hold a motor map aligned with the sensory maps. The strongest stimulus wins and triggers orienting to its place. The senses converge on the same output neurons. The basal ganglia inhibit the tectal output, from lamprey to mammals |
| Lee, Rohrer and Sparks, Nature 332, 357 (1988) | Abstract | The movement is the weighted average over the active population of the colliculus |
| Saitoh, Ménard and Grillner, J. Neurophysiol. 97, 3093 (2007) | Abstract | Stimulation of the lamprey tectum evokes eye, body orientation and locomotor movements, by site |
| Hikosaka, Takikawa and Kawagoe, Physiol. Rev. 80, 953 (2000) | Abstract | The substantia nigra inhibits the colliculus; its release gates the movement |
| Stein, Meredith, Huneycutt and McDade, J. Cogn. Neurosci. 1, 12 (1989) | Abstract | A tone at the place of a dim light raised the approach to it |

## The change

Behind `--tectal-orienting {off,group,active,audio}`, default off, which needs `--map-binding`.
Code: `models/core/tectal_motor_map.py`, wired in `main()` by `_wire_tectal_orienting`, which wraps
the policy's `select_action`. The policy is always called first, so every random stream stays the
same. `run_episode` and `init_components` do not change.

| Part | Rule, set before any run |
|---|---|
| Target cells | `group`: the cells of the group the map binding selected. `active`: every cell with a combined drive of at least half maximum, with no selection. `audio`: the cells with a sound drive of at least 0.5 |
| Body | Cells whose centre is less than 20 room pixels (the agent radius) from the map centre show the agent itself and are removed. 32 cells. In the trimmed gate runs, 0.55 to 0.59 of the selected cells lay on the agent's own ring |
| Command | Drive-weighted centre of the target cells, as a room offset. Action = 0.5 × the unit vector toward it, plus noise of std 0.1 from its own generator (seeded with the run seed), inside [-1, 1] |
| No target | The policy's action, unchanged |
| Gate of the basal ganglia | Fixed open, an innate reflex. A learned gate is a later decision |
| Record | `tectal_orienting.json`, one target offset (or none) for each step |

Smoke run, one seed, 2 episodes of 50 steps, not a result: 86 of 100 steps had a target, and at all
of them the agent moved toward it. The agent touched a wall at 0.03 of the steps.

## Path gate (Step 2), before any test of binding

Seeds 197, 198 and 199, 10 episodes each, the Gate B7 flags with `--dark-room-decoy-sound
--dark-room-decoy-level 0.5,3.0 --dark-room-decoy-visible --dark-room-start-between --map-binding
trimmed --tectal-orienting group`. One run at a time. Read by `scripts/analysis/probe_motor_map.py`.

| Check | Rule, at each of the 3 seeds |
|---|---|
| Toward | At steps with a target, the agent's move has a positive dot product with the target at more than 0.8 of the steps |
| Wall | The share of steps at a wall is below 0.5 (earlier runs: 0.71 to 0.93) |
| Contact | The agent comes within 60 room pixels of the light or the decoy within 40 steps in more than 0.8 of the episodes |

If one check fails, I stop and report. This gate tests the motor map, not binding.

## G3 test (Step 3)

The arms differ only in the target cells of the motor map. All arms use `--map-binding trimmed`,
so the bids, the drives and the binding factors are built the same way.

| Arm | Target cells | Prediction, written before the runs |
|---|---|---|
| `group` | the group the binding selected | first contact with the louder object high |
| `active` | all active cells, no selection | the movement averages over the objects (Lee and colleagues 1988), so lower |
| `audio` | the sound place | about as high as `group`. From the start between the objects, the room's own sound synthesis and estimator point to the louder object at 1.000 of 400 offline starts at decoy level 0.5 and at level 3.0 |

Seeds 201, 202 and 203, 40 episodes, one run at a time, 9 runs. The same flags as the path gate,
with the arm's `--tectal-orienting` value.

Primary measure: in each episode, the first object touched within 40 steps is the one with the
louder tone (no contact counts as not). The agent starts midway, so the decoy level decides
which tone is louder: the light at level 0.5, the decoy at level 3.0.

Rule. G3 PASSES when the `group` arm is higher than the `active` arm, with a pooled one-sided
Fisher exact p below 0.05 and a difference above 0 at 3 of 3 seeds. Otherwise G3 FAILS. The
`audio` arm is reported and is not a criterion. If `group` is not higher than `audio`, the result
says that in this room binding adds no behavioural value beyond the sound.

Secondary measures, reported only: the share of steps in the light, the share of steps at a wall.

## Limits, stated before the runs

- The orienting is a fixed reflex. The test shows whether the binding's selection, through a motor
  map, changes behaviour. It does not show learning.
- In this room the sound alone identifies the louder object, so binding can only add the place of
  the object, not which object it is.
