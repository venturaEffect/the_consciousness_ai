# Survival of the decodability result through a p-bit channel, on the project's own probe

**Result, negative first.**

1. **One prediction FAILED.** I predicted that sensory noise would lower every reading a little. It did not. With Gaussian pixel
   noise of standard deviation 8 on the sample (`DMTSEnv(sample_noise=8)`, pixel range 0 to 255), the project's shape decode falls
   to 0.131 to 0.321 on every stream, every model and both probes (chance 0.167). On the same trials without noise, `obs_map` gives
   0.988. This loss happens before any channel is applied.
2. **Shape is not read from an unseen stimulus.** With whole stimuli held out, the project's two probes give 0.054 to 0.239 for
   shape on `obs_map`, `z_state` and `tectum_content` at all three models. Every 95% interval includes chance or lies below it.
   This repeats the closed-stimulus result with the project's decoders instead of my readout.
3. **What does survive.** On the recordings without sensory noise, the project's reading (shape and colour on the project's split)
   and the 72-way identity reading both survive the p-bit channel. With the 64-axis encoding, `obs_map` shape reaches 90% of its
   clean value at 64 samples per axis with the linear probe at all three models, which is 4096 channel events per decision.
   Identity on `obs_map` reaches it at 16 samples per axis.

So the property that survives is "a known stimulus is told apart from the other known stimuli". It is the same property the
project's probe measures, and it has the same limit. This is a software emulation. It gives no hardware number, it does not test
section 13 of `docs/preregistered_predictions.md`, it moves no indicator, and it says nothing about consciousness or Phi.

Probe and plan: `scripts/analysis/probe_thermodynamic_survival.py`. The readings, the predictions, the cost limits and the report
plan were committed (`2bf86ef`) before the data run. There is no pass line. One pilot on a discarded configuration (model 42,
probe seed 999, 3 episodes) set the cost limits and nothing else. Raw output: `runs/thermo_survival/` (local, gitignored).
The run took 6 hours against an estimate of 5.

## Design in short

- **Data.** The recording loop of `probe_collapse_locus.py`, with the size label added. Checkpoints `capfix_alllevels` (42),
  `capfix_seed43`, `capfix_seed44`. 14 episodes of 20 trials, probe seed 42, one vector per trial, n = 280. A second set of
  recordings uses sensory noise 8. Nothing is pooled across models.
- **Decoders.** The project's two pipelines, unchanged. Linear is StandardScaler + LogisticRegression(max_iter=1000). PCA+MLP is
  StandardScaler + PCA(80) + MLP((64,), max_iter=2000). On raw input the probe compares its output with the project's own
  `linear_decode` and `pca_mlp_decode` and raises on any difference. It did not raise.
- **Readings.** R1 is the project's split (30% test, stratified) on shape and colour. R2 is the 72-way identity label on the same
  kind of split, chance 1/72. R3 holds out whole stimuli, shape and colour.
- **Channel.** Each vector is encoded, sent through independent p-bits with T samples per dimension, and decoded to an estimate.
  The decoder is fitted and tested on channel outputs. "Plain" sends every raw dimension. "full64" sends the top 64 principal
  axes, variance-equalised. "Clean" is the encoding with no channel. "Raw" is the project's input, untouched.
- **Intervals.** 95%, from a bootstrap that resamples test stimuli. T90 and T95 are the smallest tested dose that reaches 90% and
  95% of the clean above-chance accuracy. They are given only where the clean lower bound is above chance + 0.05.

## Audit of the data

| Recording | Trials | Distinct stimuli | Distinct rows: obs_map | z_state | tectum_content | Test rows with a stimulus seen in training: R1 shape | R1 colour | R2 | R3 | R2 test rows (dropped) |
|---|---|---|---|---|---|---|---|---|---|---|
| model 42, noise 0 | 280 | 67 | 67 | 278 | 278 | 0.976 | 0.940 | 1.000 | 0.000 | 76 (8) |
| model 42, noise 8 | 280 | 67 | 280 | 280 | 280 | 0.976 | 0.940 | 1.000 | 0.000 | 76 (8) |
| model 43, noise 0 | 280 | 67 | 67 | 278 | 278 | 0.976 | 0.940 | 1.000 | 0.000 | 76 (8) |
| model 43, noise 8 | 280 | 67 | 280 | 280 | 280 | 0.976 | 0.940 | 1.000 | 0.000 | 76 (8) |
| model 44, noise 0 | 280 | 67 | 67 | 278 | 278 | 0.976 | 0.940 | 1.000 | 0.000 | 76 (8) |
| model 44, noise 8 | 280 | 67 | 280 | 280 | 280 | 0.976 | 0.940 | 1.000 | 0.000 | 76 (8) |

The 280 trials hold 67 of the 72 stimuli. Without noise, `obs_map` has 67 distinct rows, one per stimulus. So on R1 the test
vector of `obs_map` is an exact copy of a training vector for 97.6% of the test trials (shape split). With noise 8 every row is
distinct, and the share of test trials whose stimulus is in training is unchanged.

## Step 1: the project's numbers, reproduced

Before this run I ran the project's probe as it is on the three checkpoints. Model 42 gave all eight documented values exactly
(`docs/results/b1_continuous_latent_2026_07.md`). Models 43 and 44 have no documented reference. The table sets those outputs
beside the raw R1 values of this run.

| Model | Stream | Label | Project probe, linear | This run, linear | Project probe, PCA+MLP | This run, PCA+MLP |
|---|---|---|---|---|---|---|
| 42 | `obs_map` | shape | 0.988 | 0.988 | 0.976 | 0.976 |
| 42 | `obs_map` | color | 1.000 | 1.000 | 0.988 | 0.988 |
| 42 | `z_state` | shape | 0.988 | 0.988 | 0.762 | 0.750 |
| 42 | `z_state` | color | 1.000 | 1.000 | 0.869 | 0.881 |
| 42 | `tectum_content` | shape | 0.524 | 0.512 | 0.833 | 0.821 |
| 42 | `tectum_content` | color | 0.988 | 0.988 | 0.976 | 0.952 |
| 43 | `obs_map` | shape | 0.988 | 0.988 | 0.976 | 0.976 |
| 43 | `obs_map` | color | 1.000 | 1.000 | 0.988 | 0.988 |
| 43 | `z_state` | shape | 0.988 | 0.988 | 0.845 | 0.845 |
| 43 | `z_state` | color | 1.000 | 1.000 | 0.940 | 0.940 |
| 43 | `tectum_content` | shape | 0.714 | 0.738 | 0.976 | 0.952 |
| 43 | `tectum_content` | color | 1.000 | 1.000 | 0.988 | 0.988 |
| 44 | `obs_map` | shape | 0.988 | 0.988 | 0.976 | 0.976 |
| 44 | `obs_map` | color | 1.000 | 1.000 | 0.988 | 0.988 |
| 44 | `z_state` | shape | 0.893 | 0.893 | 0.655 | 0.655 |
| 44 | `z_state` | color | 0.917 | 0.917 | 0.845 | 0.845 |
| 44 | `tectum_content` | shape | 0.226 | 0.226 | 0.429 | 0.440 |
| 44 | `tectum_content` | color | 0.524 | 0.512 | 0.857 | 0.869 |

All `obs_map` values and all `z_state` linear values agree exactly. Ten of the 36 values differ by one or two test trials of 84 (largest difference
0.024). I checked the cause. In one process, the project's recording function and mine return identical vectors (largest
absolute difference 0.0 on two episodes). The saved recordings differ from them by at most 2.5e-06, which is floating-point
rounding between processes. So a difference of that size moves one or two borderline test trials. Any difference of 0.02 or less
between two cells of this document is inside that noise.

## Raw readings, no sensory noise

Chance is 0.167 for shape and colour and 0.014 for identity.

| Model | Stream | Probe | R1 shape | R1 colour | R2 identity | R3 shape | R3 colour |
|---|---|---|---|---|---|---|---|
| 42 | `obs_map` | linear | 0.988 [0.96, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 0.163 [0.00, 0.36] | 0.913 [0.79, 1.00] |
| 42 | `obs_map` | PCA+MLP | 0.976 [0.93, 1.00] | 0.988 [0.96, 1.00] | 1.000 [1.00, 1.00] | 0.054 [0.00, 0.15] | 0.380 [0.16, 0.61] |
| 42 | `z_state` | linear | 0.988 [0.96, 1.00] | 1.000 [1.00, 1.00] | 0.987 [0.96, 1.00] | 0.141 [0.00, 0.33] | 0.946 [0.84, 1.00] |
| 42 | `z_state` | PCA+MLP | 0.750 [0.62, 0.85] | 0.881 [0.79, 0.95] | 0.803 [0.68, 0.90] | 0.109 [0.02, 0.23] | 0.696 [0.51, 0.84] |
| 42 | `tectum_content` | linear | 0.512 [0.35, 0.67] | 0.988 [0.96, 1.00] | 0.829 [0.72, 0.92] | 0.065 [0.00, 0.16] | 0.761 [0.56, 0.91] |
| 42 | `tectum_content` | PCA+MLP | 0.821 [0.73, 0.90] | 0.952 [0.91, 0.99] | 0.763 [0.66, 0.86] | 0.207 [0.06, 0.38] | 0.609 [0.41, 0.79] |
| 43 | `obs_map` | linear | 0.988 [0.96, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 0.098 [0.00, 0.26] | 0.978 [0.92, 1.00] |
| 43 | `obs_map` | PCA+MLP | 0.976 [0.94, 1.00] | 0.988 [0.96, 1.00] | 1.000 [1.00, 1.00] | 0.054 [0.00, 0.15] | 0.728 [0.51, 0.90] |
| 43 | `z_state` | linear | 0.988 [0.96, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 0.065 [0.00, 0.21] | 0.978 [0.92, 1.00] |
| 43 | `z_state` | PCA+MLP | 0.845 [0.75, 0.92] | 0.940 [0.87, 1.00] | 0.882 [0.79, 0.95] | 0.239 [0.06, 0.43] | 0.620 [0.42, 0.80] |
| 43 | `tectum_content` | linear | 0.738 [0.63, 0.83] | 1.000 [1.00, 1.00] | 0.947 [0.89, 0.99] | 0.065 [0.00, 0.16] | 0.793 [0.59, 0.97] |
| 43 | `tectum_content` | PCA+MLP | 0.952 [0.90, 0.99] | 0.988 [0.96, 1.00] | 0.974 [0.93, 1.00] | 0.152 [0.05, 0.27] | 0.772 [0.56, 0.96] |
| 44 | `obs_map` | linear | 0.988 [0.96, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 0.141 [0.00, 0.32] | 0.783 [0.56, 0.98] |
| 44 | `obs_map` | PCA+MLP | 0.976 [0.94, 1.00] | 0.988 [0.96, 1.00] | 1.000 [1.00, 1.00] | 0.174 [0.02, 0.36] | 0.489 [0.27, 0.72] |
| 44 | `z_state` | linear | 0.893 [0.82, 0.96] | 0.917 [0.85, 0.97] | 0.803 [0.69, 0.89] | 0.098 [0.00, 0.26] | 0.772 [0.56, 0.97] |
| 44 | `z_state` | PCA+MLP | 0.655 [0.54, 0.75] | 0.845 [0.76, 0.92] | 0.605 [0.46, 0.74] | 0.098 [0.01, 0.26] | 0.359 [0.23, 0.52] |
| 44 | `tectum_content` | linear | 0.226 [0.11, 0.35] | 0.512 [0.41, 0.62] | 0.092 [0.02, 0.18] | 0.054 [0.01, 0.11] | 0.283 [0.15, 0.44] |
| 44 | `tectum_content` | PCA+MLP | 0.440 [0.32, 0.55] | 0.869 [0.78, 0.94] | 0.526 [0.38, 0.67] | 0.120 [0.03, 0.24] | 0.750 [0.63, 0.86] |

- **Prediction V3 held.** Identity on raw `obs_map` is 1.000 at all three models with both probes. It is lower for `z_state` and
  `tectum_content`, which also depend on the recurrent state.
- **Prediction V4 held.** Shape under R3 is at or below chance on all three streams. Colour generalises better than shape. With
  the linear probe, colour on unseen stimuli is 0.783 to 0.978 on `obs_map` and 0.772 to 0.978 on `z_state`.
- **Model 44 is weaker downstream.** Its `tectum_content` gives 0.226 and 0.440 for shape on R1 and 0.092 for identity with the
  linear probe.

## Raw readings, sensory noise 8

| Model | Stream | Probe | R1 shape | R1 colour | R2 identity | R3 shape | R3 colour |
|---|---|---|---|---|---|---|---|
| 42 | `obs_map` | linear | 0.226 [0.11, 0.35] | 0.810 [0.71, 0.89] | 0.461 [0.32, 0.60] | 0.098 [0.02, 0.18] | 0.554 [0.39, 0.70] |
| 42 | `obs_map` | PCA+MLP | 0.190 [0.09, 0.29] | 0.345 [0.23, 0.46] | 0.066 [0.01, 0.13] | 0.141 [0.05, 0.24] | 0.326 [0.22, 0.43] |
| 42 | `z_state` | linear | 0.190 [0.11, 0.27] | 0.738 [0.62, 0.84] | 0.368 [0.24, 0.51] | 0.054 [0.02, 0.10] | 0.652 [0.50, 0.79] |
| 42 | `z_state` | PCA+MLP | 0.167 [0.08, 0.26] | 0.500 [0.39, 0.60] | 0.079 [0.03, 0.14] | 0.087 [0.04, 0.14] | 0.283 [0.18, 0.38] |
| 42 | `tectum_content` | linear | 0.190 [0.11, 0.29] | 0.690 [0.57, 0.79] | 0.171 [0.09, 0.27] | 0.087 [0.03, 0.14] | 0.598 [0.47, 0.72] |
| 42 | `tectum_content` | PCA+MLP | 0.262 [0.17, 0.35] | 0.583 [0.46, 0.70] | 0.105 [0.04, 0.19] | 0.065 [0.02, 0.11] | 0.446 [0.31, 0.59] |
| 43 | `obs_map` | linear | 0.262 [0.17, 0.36] | 0.893 [0.81, 0.96] | 0.566 [0.42, 0.70] | 0.087 [0.02, 0.16] | 0.652 [0.53, 0.78] |
| 43 | `obs_map` | PCA+MLP | 0.274 [0.17, 0.37] | 0.560 [0.43, 0.69] | 0.039 [0.00, 0.09] | 0.163 [0.09, 0.23] | 0.424 [0.29, 0.55] |
| 43 | `z_state` | linear | 0.321 [0.23, 0.43] | 0.833 [0.73, 0.92] | 0.487 [0.35, 0.62] | 0.109 [0.05, 0.17] | 0.576 [0.44, 0.71] |
| 43 | `z_state` | PCA+MLP | 0.286 [0.19, 0.38] | 0.476 [0.37, 0.57] | 0.079 [0.01, 0.15] | 0.163 [0.09, 0.23] | 0.359 [0.24, 0.48] |
| 43 | `tectum_content` | linear | 0.214 [0.13, 0.30] | 0.524 [0.41, 0.63] | 0.066 [0.01, 0.12] | 0.076 [0.02, 0.15] | 0.435 [0.30, 0.59] |
| 43 | `tectum_content` | PCA+MLP | 0.226 [0.14, 0.31] | 0.440 [0.31, 0.56] | 0.092 [0.03, 0.16] | 0.076 [0.02, 0.13] | 0.337 [0.23, 0.46] |
| 44 | `obs_map` | linear | 0.226 [0.13, 0.32] | 0.798 [0.70, 0.89] | 0.487 [0.33, 0.63] | 0.098 [0.04, 0.17] | 0.630 [0.48, 0.78] |
| 44 | `obs_map` | PCA+MLP | 0.274 [0.18, 0.37] | 0.440 [0.32, 0.55] | 0.105 [0.03, 0.19] | 0.098 [0.03, 0.17] | 0.391 [0.31, 0.47] |
| 44 | `z_state` | linear | 0.250 [0.15, 0.36] | 0.607 [0.49, 0.73] | 0.263 [0.16, 0.37] | 0.098 [0.05, 0.17] | 0.435 [0.29, 0.60] |
| 44 | `z_state` | PCA+MLP | 0.131 [0.07, 0.21] | 0.333 [0.22, 0.44] | 0.066 [0.02, 0.12] | 0.141 [0.08, 0.22] | 0.250 [0.17, 0.33] |
| 44 | `tectum_content` | linear | 0.155 [0.07, 0.25] | 0.393 [0.28, 0.49] | 0.053 [0.01, 0.10] | 0.065 [0.01, 0.14] | 0.207 [0.10, 0.32] |
| 44 | `tectum_content` | PCA+MLP | 0.262 [0.17, 0.35] | 0.452 [0.34, 0.57] | 0.092 [0.04, 0.15] | 0.120 [0.04, 0.20] | 0.359 [0.23, 0.49] |

- **Prediction V5 FAILED.** Shape on R1 is 0.131 to 0.321 everywhere. The order of the readings also changes, because colour and
  identity keep part of their value while shape keeps none. Linear colour on `obs_map` is 0.798 to 0.893. Linear identity on
  `obs_map` is 0.461 to 0.566, against 1.000 without noise.
- **What this does and does not show.** The project's probe, at its documented size (196 training trials, 16384 dimensions for
  `obs_map`), does not read shape once two presentations of a stimulus differ by pixel noise of standard deviation 8. It does not
  show that shape information is absent from the noisy vectors. I did not test more trials, a lower noise level, or a decoder that
  fits few samples in many dimensions. The models were trained without sensory noise.

## The channel, on the project's reading (R1 shape), full64 encoding

| Model | Stream | Probe | Raw | Clean | T=16 | T=64 | T=256 | T=1024 | T=4096 | T90 | T95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | `obs_map` | linear | 0.988 | 0.988 [0.96, 1.00] | 0.857 | 0.988 | 0.988 | 0.988 | 0.988 | 64 | 64 |
| 42 | `obs_map` | PCA+MLP | 0.976 | 0.988 [0.96, 1.00] | 0.952 | 0.988 | 0.988 | 0.988 | 0.988 | 16 | 16 |
| 42 | `z_state` | linear | 0.988 | 0.988 [0.96, 1.00] | 0.726 | 0.833 | 0.988 | 0.988 | 0.988 | 256 | 256 |
| 42 | `z_state` | PCA+MLP | 0.750 | 0.988 [0.96, 1.00] | 0.893 | 0.976 | 0.988 | 0.988 | 0.988 | 64 | 64 |
| 42 | `tectum_content` | linear | 0.512 | 0.500 [0.38, 0.61] | 0.345 | 0.464 | 0.524 | 0.500 | 0.500 | 256 | 256 |
| 42 | `tectum_content` | PCA+MLP | 0.821 | 0.762 [0.67, 0.85] | 0.583 | 0.750 | 0.774 | 0.821 | 0.762 | 64 | 64 |
| 43 | `obs_map` | linear | 0.988 | 0.988 [0.96, 1.00] | 0.845 | 0.988 | 0.988 | 0.988 | 0.988 | 64 | 64 |
| 43 | `obs_map` | PCA+MLP | 0.976 | 0.988 [0.96, 1.00] | 0.964 | 0.988 | 0.976 | 0.988 | 0.988 | 16 | 16 |
| 43 | `z_state` | linear | 0.988 | 0.988 [0.96, 1.00] | 0.821 | 0.964 | 0.988 | 0.988 | 0.988 | 64 | 64 |
| 43 | `z_state` | PCA+MLP | 0.845 | 0.988 [0.96, 1.00] | 0.917 | 0.988 | 0.988 | 0.976 | 0.988 | 16 | 64 |
| 43 | `tectum_content` | linear | 0.738 | 0.774 [0.67, 0.86] | 0.536 | 0.798 | 0.738 | 0.774 | 0.798 | 64 | 64 |
| 43 | `tectum_content` | PCA+MLP | 0.952 | 0.917 [0.85, 0.97] | 0.786 | 0.917 | 0.940 | 0.929 | 0.917 | 64 | 64 |
| 44 | `obs_map` | linear | 0.988 | 0.988 [0.96, 1.00] | 0.798 | 0.976 | 0.988 | 0.988 | 0.988 | 64 | 64 |
| 44 | `obs_map` | PCA+MLP | 0.976 | 0.988 [0.96, 1.00] | 0.929 | 0.988 | 0.988 | 0.988 | 0.988 | 16 | 64 |
| 44 | `z_state` | linear | 0.893 | 0.988 [0.96, 1.00] | 0.357 | 0.536 | 0.690 | 0.917 | 0.964 | 1024 | 4096 |
| 44 | `z_state` | PCA+MLP | 0.655 | 0.988 [0.96, 1.00] | 0.571 | 0.762 | 0.929 | 0.976 | 0.988 | 256 | 1024 |
| 44 | `tectum_content` | linear | 0.226 | 0.429 [0.29, 0.57] | 0.250 | 0.262 | 0.298 | 0.202 | 0.357 | none | none |
| 44 | `tectum_content` | PCA+MLP | 0.440 | 0.738 [0.61, 0.86] | 0.190 | 0.381 | 0.440 | 0.405 | 0.607 | none | none |

- **Prediction V2 held in part.** `obs_map` reaches T90 at 64 samples per axis with the linear probe and at 16 with PCA+MLP, at all
  three models. `z_state` needs more at two of three models with the linear probe (256, 64 and 1024).
- `tectum_content` reaches T90 at 64 to 256 for models 42 and 43. For model 44 it does not reach it at any tested dose.
- **An encoding effect that is not a channel effect.** With no channel, the full64 encoding raises the PCA+MLP shape decode of
  `z_state` from 0.750, 0.845 and 0.655 on raw input to 0.988 at all three models. The axes are fitted on training rows whose
  stimuli also occur in the test rows, so this belongs to the known-stimulus reading. It is one split per model.

## The same reading with the plain encoding

| Model | Stream | Probe | Raw | Clean | T=16 | T=64 | T=256 | T=1024 | T=4096 | T90 | T95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | `obs_map` | linear | 0.988 | 0.988 [0.96, 1.00] | 0.595 | 0.738 | 0.952 | 0.976 | 0.988 | 256 | 256 |
| 42 | `obs_map` | PCA+MLP | 0.976 | 0.976 [0.93, 1.00] | 0.726 | 0.821 | 0.964 | 0.976 | 0.976 | 256 | 256 |
| 42 | `z_state` | linear | 0.988 | 0.988 [0.96, 1.00] | not run | 0.619 | not run | 0.976 | not run | 1024 | 1024 |
| 42 | `z_state` | PCA+MLP | 0.750 | 0.750 [0.64, 0.85] | 0.524 | 0.512 | 0.679 | 0.714 | 0.738 | 1024 | 4096 |
| 42 | `tectum_content` | linear | 0.512 | 0.512 [0.37, 0.65] | 0.286 | 0.357 | 0.381 | 0.500 | 0.488 | 1024 | 1024 |
| 42 | `tectum_content` | PCA+MLP | 0.821 | 0.786 [0.68, 0.87] | 0.405 | 0.512 | 0.643 | 0.774 | 0.774 | 1024 | 1024 |
| 43 | `obs_map` | linear | 0.988 | 0.988 [0.96, 1.00] | 0.631 | 0.738 | 0.952 | 0.988 | 0.988 | 256 | 256 |
| 43 | `obs_map` | PCA+MLP | 0.976 | 0.976 [0.94, 1.00] | 0.702 | 0.905 | 0.976 | 0.964 | 0.976 | 64 | 256 |
| 43 | `z_state` | linear | 0.988 | 0.988 [0.96, 1.00] | not run | 0.750 | not run | 0.988 | not run | 1024 | 1024 |
| 43 | `z_state` | PCA+MLP | 0.845 | 0.881 [0.80, 0.94] | 0.548 | 0.774 | 0.810 | 0.857 | 0.869 | 256 | 1024 |
| 43 | `tectum_content` | linear | 0.738 | 0.750 [0.64, 0.84] | 0.226 | 0.357 | 0.452 | 0.536 | 0.726 | 4096 | 4096 |
| 43 | `tectum_content` | PCA+MLP | 0.952 | 0.940 [0.88, 0.99] | 0.381 | 0.714 | 0.857 | 0.917 | 0.964 | 1024 | 1024 |
| 44 | `obs_map` | linear | 0.988 | 0.988 [0.96, 1.00] | 0.643 | 0.774 | 0.940 | 0.988 | 0.988 | 256 | 1024 |
| 44 | `obs_map` | PCA+MLP | 0.976 | 0.976 [0.94, 1.00] | 0.738 | 0.952 | 0.964 | 0.964 | 0.976 | 64 | 64 |
| 44 | `z_state` | linear | 0.893 | 0.821 [0.73, 0.90] | not run | 0.333 | not run | 0.369 | not run | none | none |
| 44 | `z_state` | PCA+MLP | 0.655 | 0.595 [0.47, 0.71] | 0.202 | 0.179 | 0.250 | 0.333 | 0.381 | none | none |
| 44 | `tectum_content` | linear | 0.226 | 0.143 [0.06, 0.24] | 0.179 | 0.202 | 0.155 | 0.143 | 0.167 | none | none |
| 44 | `tectum_content` | PCA+MLP | 0.440 | 0.488 [0.36, 0.60] | 0.226 | 0.155 | 0.143 | 0.214 | 0.214 | none | none |

For `z_state` with the linear probe, only the clean arm and doses 64 and 1024 were run. That is the cost limit set before the
run. For model 44 the plain encoding does not reach T90 on `z_state` or `tectum_content` at any tested dose.

## Events per decision

| Encoding | Stream | Dimensions sent | Events per decision at T=16 | T=64 | T=256 | T=1024 | T=4096 |
|---|---|---|---|---|---|---|---|
| plain | `obs_map` | 16384 | 262144 | 1048576 | 4194304 | 16777216 | 67108864 |
| plain | `z_state` | 262144 | 4194304 | 16777216 | 67108864 | 268435456 | 1073741824 |
| plain | `tectum_content` | 256 | 4096 | 16384 | 65536 | 262144 | 1048576 |
| full64 | every stream | 64 | 1024 | 4096 | 16384 | 65536 | 262144 |

For `obs_map` shape with the linear probe, the plain encoding reaches T90 at T=256 (4194304 events) and full64 reaches it at T=64
(4096 events), at all three models. The ratio is 1024. This count leaves out the cost of the transform that makes the 64 axes
(`docs/results/thermodynamic_transform_cost_2026_10.md`).

## The channel, on known-stimulus identity (R2), full64 encoding

| Model | Stream | Probe | Raw | Clean | T=16 | T=64 | T=256 | T=1024 | T=4096 | T90 | T95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | `obs_map` | linear | 1.000 | 1.000 [1.00, 1.00] | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 16 | 16 |
| 42 | `obs_map` | PCA+MLP | 1.000 | 1.000 [1.00, 1.00] | 0.987 | 1.000 | 1.000 | 1.000 | 1.000 | 16 | 16 |
| 42 | `z_state` | linear | 0.987 | 1.000 [1.00, 1.00] | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 16 | 16 |
| 42 | `z_state` | PCA+MLP | 0.803 | 1.000 [1.00, 1.00] | 0.987 | 1.000 | 1.000 | 1.000 | 1.000 | 16 | 16 |
| 42 | `tectum_content` | linear | 0.829 | 0.987 [0.96, 1.00] | 0.961 | 1.000 | 0.987 | 0.987 | 0.987 | 16 | 16 |
| 42 | `tectum_content` | PCA+MLP | 0.763 | 0.934 [0.86, 0.99] | 0.816 | 0.908 | 0.934 | 0.947 | 0.947 | 64 | 64 |
| 43 | `obs_map` | linear | 1.000 | 1.000 [1.00, 1.00] | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 16 | 16 |
| 43 | `obs_map` | PCA+MLP | 1.000 | 1.000 [1.00, 1.00] | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 16 | 16 |
| 43 | `z_state` | linear | 1.000 | 1.000 [1.00, 1.00] | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 16 | 16 |
| 43 | `z_state` | PCA+MLP | 0.882 | 1.000 [1.00, 1.00] | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 16 | 16 |
| 43 | `tectum_content` | linear | 0.947 | 1.000 [1.00, 1.00] | 0.974 | 1.000 | 1.000 | 1.000 | 1.000 | 16 | 16 |
| 43 | `tectum_content` | PCA+MLP | 0.974 | 1.000 [1.00, 1.00] | 0.921 | 1.000 | 0.974 | 0.987 | 1.000 | 16 | 64 |
| 44 | `obs_map` | linear | 1.000 | 1.000 [1.00, 1.00] | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 16 | 16 |
| 44 | `obs_map` | PCA+MLP | 1.000 | 1.000 [1.00, 1.00] | 0.987 | 1.000 | 1.000 | 1.000 | 1.000 | 16 | 16 |
| 44 | `z_state` | linear | 0.803 | 1.000 [1.00, 1.00] | 0.855 | 0.961 | 0.987 | 1.000 | 1.000 | 64 | 64 |
| 44 | `z_state` | PCA+MLP | 0.605 | 1.000 [1.00, 1.00] | 0.605 | 0.921 | 0.987 | 1.000 | 1.000 | 64 | 256 |
| 44 | `tectum_content` | linear | 0.092 | 0.947 [0.87, 1.00] | 0.250 | 0.461 | 0.684 | 0.816 | 0.934 | 4096 | 4096 |
| 44 | `tectum_content` | PCA+MLP | 0.526 | 0.882 [0.77, 0.97] | 0.158 | 0.250 | 0.579 | 0.711 | 0.776 | none | none |

Identity is the reading the match-to-sample task needs. On `obs_map` it is at 0.987 or above at T=16 with both probes at all
three models. On `z_state` it reaches T90 at 16, 16 and 64. On `tectum_content` it reaches T90 at 16 for models 42 and 43 with the
linear probe and only at 4096 for model 44.

## Colour, full64 encoding

The reading of the project.

| Model | Stream | Probe | Raw | Clean | T=16 | T=64 | T=256 | T=1024 | T=4096 | T90 | T95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | `obs_map` | linear | 1.000 | 0.976 [0.92, 1.00] | 0.964 | 0.976 | 0.976 | 0.976 | 0.976 | 16 | 16 |
| 42 | `obs_map` | PCA+MLP | 0.988 | 0.952 [0.89, 1.00] | 0.964 | 0.952 | 0.952 | 0.964 | 0.952 | 16 | 16 |
| 42 | `z_state` | linear | 1.000 | 0.964 [0.90, 1.00] | 0.940 | 0.964 | 0.988 | 0.964 | 0.964 | 16 | 16 |
| 42 | `z_state` | PCA+MLP | 0.881 | 0.952 [0.89, 1.00] | 0.917 | 0.964 | 0.964 | 0.940 | 0.940 | 16 | 16 |
| 42 | `tectum_content` | linear | 0.988 | 0.988 [0.96, 1.00] | 0.905 | 0.988 | 0.988 | 0.976 | 0.988 | 64 | 64 |
| 42 | `tectum_content` | PCA+MLP | 0.952 | 0.952 [0.89, 1.00] | 0.881 | 0.940 | 0.976 | 0.952 | 0.952 | 16 | 64 |
| 43 | `obs_map` | linear | 1.000 | 1.000 [1.00, 1.00] | 0.952 | 0.988 | 0.988 | 1.000 | 0.988 | 16 | 64 |
| 43 | `obs_map` | PCA+MLP | 0.988 | 0.976 [0.94, 1.00] | 0.988 | 0.988 | 0.988 | 0.988 | 0.988 | 16 | 16 |
| 43 | `z_state` | linear | 1.000 | 1.000 [1.00, 1.00] | 0.988 | 1.000 | 1.000 | 1.000 | 1.000 | 16 | 16 |
| 43 | `z_state` | PCA+MLP | 0.940 | 0.952 [0.89, 1.00] | 0.964 | 1.000 | 0.964 | 0.988 | 0.940 | 16 | 16 |
| 43 | `tectum_content` | linear | 1.000 | 1.000 [1.00, 1.00] | 0.940 | 1.000 | 1.000 | 1.000 | 1.000 | 16 | 64 |
| 43 | `tectum_content` | PCA+MLP | 0.988 | 0.988 [0.96, 1.00] | 0.952 | 0.964 | 0.976 | 0.976 | 0.988 | 16 | 16 |
| 44 | `obs_map` | linear | 1.000 | 1.000 [1.00, 1.00] | 0.976 | 1.000 | 1.000 | 1.000 | 1.000 | 16 | 16 |
| 44 | `obs_map` | PCA+MLP | 0.988 | 0.988 [0.96, 1.00] | 0.964 | 0.976 | 0.964 | 0.964 | 0.952 | 16 | 16 |
| 44 | `z_state` | linear | 0.917 | 1.000 [1.00, 1.00] | 0.917 | 0.988 | 0.976 | 1.000 | 1.000 | 16 | 64 |
| 44 | `z_state` | PCA+MLP | 0.845 | 0.964 [0.92, 1.00] | 0.893 | 0.952 | 0.952 | 0.964 | 0.964 | 16 | 64 |
| 44 | `tectum_content` | linear | 0.512 | 0.964 [0.92, 1.00] | 0.500 | 0.726 | 0.833 | 0.940 | 0.893 | 1024 | 1024 |
| 44 | `tectum_content` | PCA+MLP | 0.869 | 0.929 [0.86, 0.99] | 0.429 | 0.667 | 0.833 | 0.845 | 0.893 | 4096 | 4096 |

Unseen stimuli.

| Model | Stream | Probe | Raw | Clean | T=16 | T=64 | T=256 | T=1024 | T=4096 | T90 | T95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | `obs_map` | linear | 0.913 | 0.891 [0.71, 1.00] | 0.783 | 0.783 | 0.717 | 0.554 | 0.391 | none | none |
| 42 | `obs_map` | PCA+MLP | 0.380 | 0.467 [0.24, 0.70] | 0.446 | 0.435 | 0.522 | 0.239 | 0.283 | 16 | 256 |
| 42 | `z_state` | linear | 0.946 | 0.576 [0.34, 0.82] | 0.565 | 0.804 | 0.804 | 0.750 | 0.793 | 16 | 16 |
| 42 | `z_state` | PCA+MLP | 0.696 | 0.054 [0.00, 0.14] | 0.554 | 0.783 | 0.511 | 0.522 | 0.359 | none | none |
| 42 | `tectum_content` | linear | 0.761 | 0.717 [0.51, 0.89] | 0.663 | 0.793 | 0.739 | 0.717 | 0.728 | 16 | 64 |
| 42 | `tectum_content` | PCA+MLP | 0.609 | 0.587 [0.42, 0.76] | 0.554 | 0.630 | 0.696 | 0.565 | 0.522 | 16 | 64 |
| 43 | `obs_map` | linear | 0.978 | 0.772 [0.55, 1.00] | 0.609 | 0.652 | 0.543 | 0.543 | 0.304 | none | none |
| 43 | `obs_map` | PCA+MLP | 0.728 | 0.565 [0.33, 0.79] | 0.424 | 0.543 | 0.511 | 0.370 | 0.239 | 64 | none |
| 43 | `z_state` | linear | 0.978 | 0.793 [0.59, 0.97] | 0.717 | 0.946 | 0.859 | 0.837 | 0.891 | 64 | 64 |
| 43 | `z_state` | PCA+MLP | 0.620 | 0.380 [0.14, 0.62] | 0.663 | 0.739 | 0.391 | 0.522 | 0.663 | none | none |
| 43 | `tectum_content` | linear | 0.793 | 0.576 [0.38, 0.79] | 0.565 | 0.543 | 0.587 | 0.598 | 0.576 | 16 | 16 |
| 43 | `tectum_content` | PCA+MLP | 0.772 | 0.598 [0.41, 0.80] | 0.489 | 0.565 | 0.446 | 0.630 | 0.467 | 64 | 1024 |
| 44 | `obs_map` | linear | 0.783 | 0.772 [0.56, 0.94] | 0.543 | 0.772 | 0.696 | 0.576 | 0.304 | 64 | 64 |
| 44 | `obs_map` | PCA+MLP | 0.489 | 0.489 [0.27, 0.73] | 0.620 | 0.467 | 0.348 | 0.304 | 0.283 | 16 | 16 |
| 44 | `z_state` | linear | 0.772 | 0.848 [0.67, 0.98] | 0.554 | 0.793 | 0.685 | 0.696 | 0.772 | 64 | none |
| 44 | `z_state` | PCA+MLP | 0.359 | 0.391 [0.19, 0.61] | 0.533 | 0.565 | 0.641 | 0.391 | 0.337 | none | none |
| 44 | `tectum_content` | linear | 0.283 | 0.576 [0.39, 0.75] | 0.424 | 0.489 | 0.696 | 0.620 | 0.565 | 256 | 256 |
| 44 | `tectum_content` | PCA+MLP | 0.750 | 0.283 [0.14, 0.45] | 0.380 | 0.380 | 0.620 | 0.565 | 0.500 | none | none |

**An anomaly I did not resolve.** For colour on unseen stimuli on `obs_map`, the full64 curve FALLS as the dose rises (linear
probe, 0.891 clean and 0.391 at T=4096 for model 42, with the same direction at models 43 and 44). A higher dose should approach
the clean value. One candidate cause is that the training rows of this split hold fewer than 64 distinct vectors, so some of the
64 axes have no training variance. I did not test this. The full64 column of the unseen-colour table must not be used until it is
explained. The plain encoding does not show it.

| Model | Stream | Probe | Raw | Clean | T=16 | T=64 | T=256 | T=1024 | T=4096 | T90 | T95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | `obs_map` | linear | 0.913 | 0.946 [0.85, 1.00] | 0.978 | 0.946 | 0.946 | 0.946 | 0.946 | 16 | 16 |
| 42 | `obs_map` | PCA+MLP | 0.380 | 0.587 [0.36, 0.81] | 0.630 | 0.511 | 0.652 | 0.554 | 0.587 | 16 | 16 |
| 42 | `z_state` | linear | 0.946 | not run | | | | | | | |
| 42 | `z_state` | PCA+MLP | 0.696 | 0.609 [0.40, 0.79] | 0.565 | 0.554 | 0.554 | 0.598 | 0.598 | 16 | 1024 |
| 42 | `tectum_content` | linear | 0.761 | 0.761 [0.57, 0.91] | 0.609 | 0.717 | 0.739 | 0.772 | 0.750 | 64 | 256 |
| 42 | `tectum_content` | PCA+MLP | 0.609 | 0.630 [0.41, 0.81] | 0.478 | 0.587 | 0.587 | 0.630 | 0.630 | 64 | 1024 |
| 43 | `obs_map` | linear | 0.978 | 0.978 [0.92, 1.00] | 0.978 | 0.967 | 0.978 | 0.978 | 0.978 | 16 | 16 |
| 43 | `obs_map` | PCA+MLP | 0.728 | 0.750 [0.52, 0.94] | 0.696 | 0.717 | 0.728 | 0.674 | 0.696 | 16 | 256 |
| 43 | `z_state` | linear | 0.978 | not run | | | | | | | |
| 43 | `z_state` | PCA+MLP | 0.620 | 0.793 [0.60, 0.95] | 0.826 | 0.783 | 0.772 | 0.804 | 0.793 | 16 | 16 |
| 43 | `tectum_content` | linear | 0.793 | 0.804 [0.60, 0.97] | 0.750 | 0.772 | 0.772 | 0.783 | 0.783 | 16 | 1024 |
| 43 | `tectum_content` | PCA+MLP | 0.772 | 0.728 [0.53, 0.91] | 0.696 | 0.728 | 0.728 | 0.696 | 0.761 | 16 | 64 |
| 44 | `obs_map` | linear | 0.783 | 0.783 [0.56, 0.98] | 0.870 | 0.772 | 0.783 | 0.783 | 0.783 | 16 | 16 |
| 44 | `obs_map` | PCA+MLP | 0.489 | 0.370 [0.16, 0.61] | 0.250 | 0.424 | 0.402 | 0.435 | 0.446 | none | none |
| 44 | `z_state` | linear | 0.772 | not run | | | | | | | |
| 44 | `z_state` | PCA+MLP | 0.359 | 0.413 [0.27, 0.58] | 0.207 | 0.250 | 0.315 | 0.359 | 0.402 | 4096 | 4096 |
| 44 | `tectum_content` | linear | 0.283 | 0.304 [0.16, 0.47] | 0.261 | 0.185 | 0.185 | 0.196 | 0.337 | none | none |
| 44 | `tectum_content` | PCA+MLP | 0.750 | 0.685 [0.53, 0.81] | 0.098 | 0.163 | 0.185 | 0.217 | 0.315 | none | none |

## The channel under sensory noise 8, full64 encoding

Shape is near chance under noise, so the tables show identity and colour.

| Model | Stream | Probe | Raw | Clean | T=16 | T=64 | T=256 | T=1024 | T=4096 | T90 | T95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | `obs_map` | linear | 0.461 | 0.461 [0.32, 0.61] | 0.289 | 0.461 | 0.447 | 0.434 | 0.461 | 64 | 64 |
| 42 | `obs_map` | PCA+MLP | 0.066 | 0.316 [0.20, 0.44] | 0.184 | 0.197 | 0.276 | 0.276 | 0.316 | 4096 | 4096 |
| 42 | `z_state` | linear | 0.368 | 0.342 [0.23, 0.47] | 0.289 | 0.316 | 0.303 | 0.342 | 0.355 | 64 | 1024 |
| 42 | `z_state` | PCA+MLP | 0.079 | 0.250 [0.15, 0.36] | 0.118 | 0.224 | 0.289 | 0.224 | 0.263 | 256 | 256 |
| 42 | `tectum_content` | linear | 0.171 | 0.171 [0.09, 0.27] | 0.158 | 0.171 | 0.158 | 0.171 | 0.171 | 16 | 64 |
| 42 | `tectum_content` | PCA+MLP | 0.105 | 0.145 [0.06, 0.25] | 0.053 | 0.145 | 0.158 | 0.171 | 0.092 | none | none |
| 43 | `obs_map` | linear | 0.566 | 0.461 [0.31, 0.62] | 0.263 | 0.421 | 0.434 | 0.461 | 0.421 | 64 | 1024 |
| 43 | `obs_map` | PCA+MLP | 0.039 | 0.342 [0.23, 0.48] | 0.132 | 0.289 | 0.342 | 0.342 | 0.263 | 256 | 256 |
| 43 | `z_state` | linear | 0.487 | 0.303 [0.19, 0.43] | 0.224 | 0.276 | 0.263 | 0.289 | 0.316 | 64 | 1024 |
| 43 | `z_state` | PCA+MLP | 0.079 | 0.158 [0.07, 0.25] | 0.092 | 0.184 | 0.158 | 0.171 | 0.184 | 64 | 64 |
| 43 | `tectum_content` | linear | 0.066 | 0.145 [0.05, 0.25] | 0.105 | 0.171 | 0.145 | 0.158 | 0.171 | none | none |
| 43 | `tectum_content` | PCA+MLP | 0.092 | 0.132 [0.06, 0.21] | 0.118 | 0.145 | 0.132 | 0.092 | 0.079 | none | none |
| 44 | `obs_map` | linear | 0.487 | 0.474 [0.34, 0.61] | 0.316 | 0.395 | 0.474 | 0.487 | 0.474 | 256 | 256 |
| 44 | `obs_map` | PCA+MLP | 0.105 | 0.342 [0.22, 0.46] | 0.316 | 0.303 | 0.316 | 0.329 | 0.368 | 16 | 1024 |
| 44 | `z_state` | linear | 0.263 | 0.329 [0.21, 0.45] | 0.250 | 0.303 | 0.329 | 0.316 | 0.316 | 64 | 256 |
| 44 | `z_state` | PCA+MLP | 0.066 | 0.211 [0.11, 0.32] | 0.145 | 0.184 | 0.158 | 0.197 | 0.250 | 1024 | 4096 |
| 44 | `tectum_content` | linear | 0.053 | 0.224 [0.12, 0.33] | 0.000 | 0.092 | 0.158 | 0.197 | 0.211 | 4096 | none |
| 44 | `tectum_content` | PCA+MLP | 0.092 | 0.132 [0.06, 0.22] | 0.026 | 0.118 | 0.066 | 0.066 | 0.118 | none | none |

| Model | Stream | Probe | Raw | Clean | T=16 | T=64 | T=256 | T=1024 | T=4096 | T90 | T95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | `obs_map` | linear | 0.810 | 0.786 [0.67, 0.88] | 0.655 | 0.738 | 0.738 | 0.798 | 0.786 | 64 | 1024 |
| 42 | `obs_map` | PCA+MLP | 0.345 | 0.750 [0.63, 0.85] | 0.631 | 0.738 | 0.726 | 0.726 | 0.726 | 64 | 64 |
| 42 | `z_state` | linear | 0.738 | 0.679 [0.54, 0.80] | 0.500 | 0.643 | 0.667 | 0.655 | 0.679 | 64 | 256 |
| 42 | `z_state` | PCA+MLP | 0.500 | 0.607 [0.47, 0.73] | 0.548 | 0.583 | 0.607 | 0.619 | 0.690 | 64 | 256 |
| 42 | `tectum_content` | linear | 0.690 | 0.690 [0.57, 0.80] | 0.548 | 0.619 | 0.631 | 0.714 | 0.667 | 1024 | 1024 |
| 42 | `tectum_content` | PCA+MLP | 0.583 | 0.619 [0.51, 0.73] | 0.500 | 0.595 | 0.643 | 0.607 | 0.583 | 64 | 256 |
| 43 | `obs_map` | linear | 0.893 | 0.798 [0.71, 0.88] | 0.560 | 0.690 | 0.774 | 0.798 | 0.798 | 256 | 256 |
| 43 | `obs_map` | PCA+MLP | 0.560 | 0.738 [0.64, 0.83] | 0.560 | 0.679 | 0.643 | 0.750 | 0.774 | 1024 | 1024 |
| 43 | `z_state` | linear | 0.833 | 0.726 [0.61, 0.84] | 0.560 | 0.714 | 0.714 | 0.714 | 0.738 | 64 | 64 |
| 43 | `z_state` | PCA+MLP | 0.476 | 0.667 [0.56, 0.77] | 0.524 | 0.679 | 0.631 | 0.690 | 0.679 | 64 | 64 |
| 43 | `tectum_content` | linear | 0.524 | 0.476 [0.36, 0.59] | 0.429 | 0.524 | 0.464 | 0.488 | 0.512 | 64 | 64 |
| 43 | `tectum_content` | PCA+MLP | 0.440 | 0.464 [0.33, 0.60] | 0.405 | 0.381 | 0.440 | 0.476 | 0.536 | 256 | 1024 |
| 44 | `obs_map` | linear | 0.798 | 0.714 [0.60, 0.82] | 0.524 | 0.631 | 0.702 | 0.726 | 0.714 | 256 | 256 |
| 44 | `obs_map` | PCA+MLP | 0.440 | 0.690 [0.58, 0.79] | 0.548 | 0.583 | 0.750 | 0.738 | 0.667 | 256 | 256 |
| 44 | `z_state` | linear | 0.607 | 0.595 [0.48, 0.71] | 0.524 | 0.619 | 0.560 | 0.619 | 0.595 | 64 | 64 |
| 44 | `z_state` | PCA+MLP | 0.333 | 0.595 [0.49, 0.71] | 0.452 | 0.631 | 0.619 | 0.607 | 0.571 | 64 | 64 |
| 44 | `tectum_content` | linear | 0.393 | 0.464 [0.34, 0.58] | 0.429 | 0.298 | 0.440 | 0.405 | 0.488 | 256 | 4096 |
| 44 | `tectum_content` | PCA+MLP | 0.452 | 0.452 [0.33, 0.57] | 0.333 | 0.345 | 0.417 | 0.429 | 0.500 | 1024 | 4096 |

Where a reading is above chance under noise, the channel does not remove it at the higher doses. The clean values are low, so
the T90 and T95 values of these two tables rest on small differences.

## Predictions, as written before the run

| Prediction | Outcome |
|---|---|
| V1. R1 on raw input reproduces the documented numbers for model 42 | Held in Step 1 (exact). In this run, within two test trials. |
| V2. R1 survives for `obs_map` at a moderate dose with full64, and `z_state` needs more | Held for `obs_map`. Held for `z_state` at two of three models with the linear probe. |
| V3. R2 is near 1.0 on raw `obs_map` and lower downstream | Held. |
| V4. R3 shape is at or below chance, colour may generalise better | Held. |
| V5. Sensory noise lowers every reading a little and keeps their order | FAILED. Shape falls to chance. |

Of the three outcomes I named as evidence against my expectations, none occurred. R1 did not collapse at every dose, R2 is 1.000
on raw `obs_map`, and R3 shape is not above chance.

## Limits

- One probe seed and one split per model. The three models are three training seeds. The intervals cover the choice of test
  stimuli and not the choice of split.
- 280 trials, 84 or fewer test rows. One test trial is 0.012.
- The log has 51 convergence warnings (45 from the logistic solver, 6 from the MLP). These are the project's own settings. I did
  not record which fits they belong to.
- The plain encoding was not run on the noise recordings, and part of the `z_state` linear grid was not run. Both limits were set
  before the run.
- `workspace_broadcast` is not in this probe. It was the one stream that generalised in the closed-stimulus study.
- The channel is the exact binomial law of independent p-bits. No device, no coupling and no drift is modelled.

## What follows

- The question "do the verified properties survive stochastic transduction, measured the same way" has a direct answer for three
  streams. They survive, at the budgets above, and the property is known-stimulus discrimination.
- The sensory-noise result bears on the project's decodability findings and not only on this track. It is recorded here. No other
  project document was changed.
- Open, each needs its own pre-stated plan. Why shape is lost under noise 8 (noise level sweep, more trials). The full64 anomaly
  on unseen colour. The same three readings for `workspace_broadcast`.
