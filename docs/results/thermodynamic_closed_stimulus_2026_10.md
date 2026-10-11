# Closed stimulus set: the earlier readouts do not generalise for three of four streams

**Result, negative first.** With folds that hold out whole stimuli, the 6-class shape readout of `obs_map`, `tectum_content` and
`z_state` falls BELOW chance at all three models (0.06 to 0.14 against chance 0.167; for the plain readout the upper 95% bounds are
0.08 to 0.16). Only
`workspace_broadcast` stays above chance (0.48 to 0.55). So the earlier clean accuracies for those three streams (0.42 to 0.77) came
from recognising stimuli the readout had already seen. They did not show that shape is read from an unseen stimulus.

**What I got wrong.** Every earlier thermodynamic probe (fidelity, encoding, matched readout, transform cost, frontier, and the gate
series) used folds grouped by trial. DMTS draws its sample from 72 stimuli (6 shapes x 6 colours x 2 sizes), and trials repeat them.
A trial fold does not keep a stimulus out of training. I did not check for repeated inputs. A design review found it.

Probe and plan: `scripts/analysis/probe_thermodynamic_stimulus.py`. Definitions, predictions and the report plan were committed
(`c56ec9a`) before the data run. One pilot on a discarded draw (model 42, draw 999) preceded it. After the pilot I added the rule that
budget values are reported only where the clean readout is above chance, and the docstring says so. Data: the 30 cached recordings
(models 42, 43, 44; stimulus draws 300 to 309; 1200 trials per model). Raw output: `runs/thermo_fidelity/sti_*.json` and
`stimulus_summary.json` (local, gitignored).

## The closed set

| Model | Trials | Distinct stimuli | Majority-class share | Distinct rows per recording (of 600): obs_map | workspace_broadcast | tectum_content | z_state |
|---|---|---|---|---|---|---|---|
| 42 | 1200 | 72 | 0.185 | 55 to 64 | 56 to 65 | 600 | 600 |
| 43 | 1200 | 72 | 0.185 | 55 to 64 | 48 to 55 | 600 | 600 |
| 44 | 1200 | 72 | 0.185 | 55 to 64 | 52 to 60 | 600 | 600 |

`obs_map` is a function of the current frame, so its distinct rows identify the stimulus. `workspace_broadcast` has about as many
distinct rows. `tectum_content` and `z_state` also depend on the recurrent state, so every row differs, but they fail in the same way.

## Clean accuracy: trial folds against stimulus folds

Intervals come from a bootstrap that resamples stimuli (72 clusters), because trials of one stimulus are not independent.
`plain` is the tuned ridge readout of the fidelity probe. `full64` is the 64-axis equalised encoding of the frontier probe.

| Model | Stream | Arm | Trial folds | Stimulus folds (95% interval) | Difference (95% interval) |
|---|---|---|---|---|---|
| 42 | tectum_content | plain | 0.454 | 0.081 (0.059 to 0.105) | -0.373 (-0.414 to -0.329) |
| 42 | tectum_content | full64 | 0.463 | 0.080 (0.056 to 0.108) | -0.383 (-0.425 to -0.338) |
| 42 | workspace_broadcast | plain | 0.751 | 0.523 (0.428 to 0.612) | -0.229 (-0.289 to -0.170) |
| 42 | workspace_broadcast | full64 | 0.664 | 0.549 (0.450 to 0.642) | -0.115 (-0.156 to -0.075) |
| 42 | obs_map | plain | 0.759 | 0.095 (0.067 to 0.127) | -0.664 (-0.703 to -0.623) |
| 42 | obs_map | full64 | 0.754 | 0.087 (0.058 to 0.120) | -0.667 (-0.708 to -0.625) |
| 42 | z_state | plain | 0.769 | 0.120 (0.086 to 0.157) | -0.649 (-0.696 to -0.599) |
| 42 | z_state | full64 | 0.766 | 0.126 (0.092 to 0.164) | -0.640 (-0.687 to -0.589) |
| 43 | tectum_content | plain | 0.627 | 0.057 (0.038 to 0.077) | -0.570 (-0.602 to -0.533) |
| 43 | tectum_content | full64 | 0.648 | 0.066 (0.045 to 0.088) | -0.582 (-0.617 to -0.546) |
| 43 | workspace_broadcast | plain | 0.738 | 0.483 (0.394 to 0.570) | -0.255 (-0.312 to -0.201) |
| 43 | workspace_broadcast | full64 | 0.672 | 0.516 (0.420 to 0.605) | -0.156 (-0.200 to -0.116) |
| 43 | obs_map | plain | 0.754 | 0.066 (0.045 to 0.088) | -0.688 (-0.720 to -0.655) |
| 43 | obs_map | full64 | 0.755 | 0.075 (0.052 to 0.099) | -0.680 (-0.714 to -0.645) |
| 43 | z_state | plain | 0.771 | 0.096 (0.066 to 0.128) | -0.674 (-0.712 to -0.638) |
| 43 | z_state | full64 | 0.767 | 0.074 (0.050 to 0.100) | -0.693 (-0.726 to -0.658) |
| 44 | tectum_content | plain | 0.424 | 0.123 (0.097 to 0.151) | -0.301 (-0.340 to -0.259) |
| 44 | tectum_content | full64 | 0.433 | 0.110 (0.081 to 0.142) | -0.323 (-0.369 to -0.274) |
| 44 | workspace_broadcast | plain | 0.725 | 0.502 (0.414 to 0.588) | -0.223 (-0.276 to -0.171) |
| 44 | workspace_broadcast | full64 | 0.647 | 0.536 (0.447 to 0.622) | -0.112 (-0.152 to -0.072) |
| 44 | obs_map | plain | 0.763 | 0.129 (0.099 to 0.161) | -0.634 (-0.675 to -0.592) |
| 44 | obs_map | full64 | 0.765 | 0.132 (0.098 to 0.166) | -0.633 (-0.672 to -0.589) |
| 44 | z_state | plain | 0.722 | 0.131 (0.106 to 0.157) | -0.591 (-0.630 to -0.549) |
| 44 | z_state | full64 | 0.684 | 0.137 (0.103 to 0.174) | -0.547 (-0.587 to -0.508) |

- **S1, supported.** `obs_map` and `workspace_broadcast` fall under stimulus folds.
- **S2, contradicted.** `tectum_content` and `z_state` fall just as far as `obs_map`. I predicted they would change less.
- **S3.** The second branch holds: for `obs_map` the earlier results describe a lookup of 72 items.
- The three failing streams are below chance. For the plain readout every upper interval bound is below 1/6. For `full64` two bounds
  at model 44 reach 0.17. A readout at chance would sit near 1/6. A
  value below it means the readout assigns an unseen stimulus to a wrong shape systematically. A readout keyed to colour and size
  would do that, because the nearest training stimuli of a held-out combination have other shapes. I did not test that explanation.

## The one stream that generalises: workspace_broadcast

64-axis equalised encoding, p-bit channel, readout trained on the channel's own outputs, stimulus folds. rho* is taken against the plain
clean accuracy under stimulus folds. T is p-bit samples per axis.

| Model | A(4) | A(16) | A(64) | A(256) | A(1024) | A(4096) | A(16384) | Clean plain | T90 | T95 (lower bound) |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.39 | 0.50 | 0.55 | 0.55 | 0.55 | 0.55 | 0.55 | 0.523 | 16 | 64 (64) |
| 43 | 0.40 | 0.49 | 0.53 | 0.54 | 0.55 | 0.54 | 0.54 | 0.483 | 16 | 16 (64) |
| 44 | 0.37 | 0.48 | 0.53 | 0.54 | 0.54 | 0.54 | 0.54 | 0.502 | 16 | 64 (64) |

For an unseen stimulus the broadcast keeps shape at 0.48 to 0.55, and the equalised 64-axis encoding reaches 95% of the clean value at
16 to 64 samples per axis (lower bound 64). So the encoding result holds for the one stream where the readout generalises.

## What this changes in the earlier documents

- **The numbers stand as measured.** They describe the label of a KNOWN stimulus read through noise. For `obs_map`, `tectum_content` and
  `z_state` they are statements about telling 72 known items apart, not about reading shape.
- **Wording to withdraw.** Statements that these three streams "carry the class" or "carry the stimulus label" are withdrawn in the
  general sense. They carry the identity of a seen stimulus.
- **The noise findings keep their form.** The dose-response curves, the saving from equalisation, the matched-readout result, the cost
  accounting and the 64-axis knee are properties of noise on a known set. Whether they hold for a generalising readout was tested here
  only for `workspace_broadcast` with `full64`, where they do.
- **Earlier model 44 findings.** The "weak" model 44 streams and the "low-variance directions" explanation were about lookup readouts. They
  should not be read as findings about shape coding.
- Each earlier result document and the blueprint now carry a correction note that points here.

## Read with care

1. **Linear readout, few distinct inputs.** A stimulus fold trains on about 48 distinct stimuli per recording with a linear readout on up
   to 256 features. A nonlinear readout, or a readout trained on more distinct stimuli, might generalise where this one does not. The
   result says this readout does not generalise for three streams. It does not say the information is absent.
2. **Held-out by full combination.** A held-out stimulus is one (shape, colour, size) combination. Its shape still occurs in training with
   other colours and sizes. So this tests generalisation across colour and size, which is what a shape readout needs.
3. **Scope.** I checked only the thermodynamic probes. I did not check whether other decodability results of the project group their
   folds by stimulus. The same question applies to them, and it is the owner's decision whether to check.
4. **Three checkpoints,** one phase (`sample`), one label (`sample_shape`). Nothing is pooled across models.

Instruments stay UNPROVEN, and 0 are TRUSTED. This result says nothing about consciousness or Phi.

## What would change this, in order

1. **Use stimulus folds from now on** in every thermodynamic probe, and report trial folds only for comparison.
2. **Rebuild the reference on `workspace_broadcast`** for the memory and decision probes, since it is the stream with a generalising readout.
3. **A readout with more distinct stimuli or a nonlinear form** for `obs_map`, to learn whether shape is recoverable there.
4. **A stimulus-fold check of the project's other decodability results,** if the owner wants it.
