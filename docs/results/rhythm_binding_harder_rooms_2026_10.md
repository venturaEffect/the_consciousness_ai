# Binding by timing in harder rooms: FAILED with irregular rhythms and with a change of the louder tone

Date: 2026-10-16. Offline diagnoses on the real room, before any build, each with its rule written
in the script before the run. No code changed.

## Verdict

FAILED. The binding that passed its gate with steady rhythms
(`docs/results/rhythm_binding_gate_2026_10.md`) did not hold in two harder conditions: irregular
rhythms, and a change of which tone is louder while the agent moves. The owner decided on
2026-10-16 to record this and to stop tuning. RPT-2 stays PARTIAL.

## Method

The real room with `--dark-room-pulse --dark-room-sound-direction off`, the visible decoy and the start
between the objects, the real cochlea, and a scripted agent with no learning, for 60 steps. Score:
the share of steps 12 to 60 with the group on the object whose tone is louder at that step (object
places from geometry). Motions: still; still until step 12, then toward the louder object at 0.5;
the same toward the quieter object (its tone becomes the louder one as the agent comes near, so the
binding must change). Rule for each condition: still ≥ 0.9, toward the louder object ≥ 0.8, toward
the quieter object ≥ 0.7.

Irregular rhythm: for each object and episode, white noise smoothed over 4 steps, standardised and
clipped to [-1, 1], in place of the sine; brightness 0.85 + 0.15 v(t), tone 0.7 + 0.3 v(t).

## Diagnosis 3a: the gate's binding, irregular rhythms

Threshold 0.55 and the 12-step wait, as fixed for the gate. Room layouts 331 to 360.

| Motion | Score | Rule | Lowest layout |
|---|---|---|---|
| Still | 0.461 | ≥ 0.9, FAIL | 0.061 |
| Toward the louder object | 0.972 | ≥ 0.8 | 0.939 |
| Toward the quieter object | 0.618 | ≥ 0.7, FAIL | 0.408 |

## Diagnosis 3b: broadband coherence, steady and irregular rhythms

The same oscillators, with the match summed over all 8 frequencies:
C = |Σ_k <z_c,k conj(z_b,k)>| / sqrt(Σ_k <|z_c,k|²> Σ_k <|z_b,k|²>). The threshold was chosen on
layouts 361 to 375 (0.65), and the test used layouts 376 to 405.

| Rhythm | Motion | Score | Rule | Lowest layout | 3a or the gate's diagnosis |
|---|---|---|---|---|---|
| Steady | Still | 1.000 | ≥ 0.9 | 1.000 | 0.992 |
| Steady | Toward the louder object | 1.000 | ≥ 0.8 | 1.000 | 0.996 |
| Steady | Toward the quieter object | 0.512 | ≥ 0.7, FAIL | 0.224 | 0.710 |
| Irregular | Still | 0.771 | ≥ 0.9, FAIL | 0.265 | 0.461 |
| Irregular | Toward the louder object | 0.988 | ≥ 0.8 | 0.918 | 0.972 |
| Irregular | Toward the quieter object | 0.511 | ≥ 0.7, FAIL | 0.163 | 0.618 |

Source: `runs/_audit_followup/oct13/rhythm_diag_irregular.out` and `rhythm_diag_broadband.out`; the
scripts stay with the session notes.

## Reading

Two needs pull against each other. Telling two independent irregular rhythms apart needs more
samples than the comparison window of about 8 steps holds; over 8 samples, unrelated signals often
look related by chance. Following a change of the louder tone needs a short window, because every
window delays the change. The broadband form helped the first need (irregular, still: 0.461 to
0.771) and hurt the second (steady, toward the quieter object: 0.710 to 0.512). This is a limit of
binding by timing over a time window, and it is a hypothesis about the cause, not a tested one.

## What stands

- The gate result with steady rhythms stands as written.
- RPT-2 stays PARTIAL: the result did not carry over to the harder rooms.
- Not tried: a window taken from the correlation detector of Parise and Ernst (about 0.8 s), and a
  grace period after a change of the louder tone. The owner chose to record this and to move on.
