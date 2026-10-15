# Binding stage 3: an orientation test for "used"

Date: 2026-10-15. Status: rules written before any run. The owner chose this test on
2026-10-15.

## Why G3 is replaced here

G3 counts the steps at which the binding changes the leader of the workspace. In the visible
decoy gate and the trimmed gate, almost every change was audio overtaking vision (742, 408,
535 changes against 4, 4, 4 others at seeds 181 to 183; 377, 238, 281 against 3, 5, 1 at
seeds 187 to 189). The sound drive lies close around the sound place, so a large share of it
falls inside the group, and the sound gets the larger factor. Binding means that sight and
sound belong to one object, so this count does not show that the bound object is used. A
stronger factor would raise the count for the same reason. So G3 is not the measure of "used"
in this test. This is a new test with its own rule, not a change of the earlier gates, whose
verdicts stand.

## The test

After the orientation test of Stein, Meredith, Huneycutt and McDade (J. Cogn. Neurosci. 1,
12, 1989, abstract read): a tone at the place of a dim light raised the approach to that light,
and a tone at another place lowered it.

Each episode starts with the agent midway between the light and the visible decoy
(`--dark-room-start-between`). Both tones come from the same distance, so the decoy level
decides which one is louder (0.5, the light; 3.0, the decoy). An episode is oriented when,
after 10 steps, the agent is closer to the louder object than to the quieter one. Read by
`scripts/analysis/probe_orientation.py`.

## Runs

Seeds 194, 195 and 196. Each seed once with `--map-binding trimmed` and once without map
binding, both recorded (the probe reads the step records). The command of the trimmed gate:
the Gate B7 flags with `--episodes 40 --dark-room-decoy-sound --dark-room-decoy-level 0.5,3.0
--dark-room-decoy-visible --dark-room-start-between`. One run at a time. Completion by the
row counts of the csv files.

## Rule

PASS when the pooled oriented share with binding is higher than without binding, with a
one-sided Fisher exact p below 0.05, and the per-seed difference is above 0 at 3 of 3 seeds.
FAIL otherwise.

## Seen before this rule

A smoke run of the probe on the six earlier runs with binding on (seeds 181 to 183 and 187 to
189, no runs without binding to compare) gave 22, 17, 18, 14, 25 and 22 oriented episodes of
40, together 118 of 240. So the agent with binding oriented to the louder object at about half
of the episodes, near chance.

## Limits, stated before the runs

- The agent has learned little, and it stays at the walls when it does not start between the
  objects. So a result of "no difference" can come from the agent and not from the binding.
- With 40 episodes per run, a difference of about 0.1 in the oriented share is hard to detect.
- The trimmed form changes the factors only a little (mean audio factor 1.095 to 1.113 in its
  gate), so a small effect is the expected size.
