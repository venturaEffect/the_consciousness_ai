# Binding stage 3: adaptation of the map drive

Date: 2026-10-14. Status: review for the owner, before any build. The owner asked for
this review on the same day, after the salience form of option B.

## The problem that this answers

Source: `docs/results/binding_map_stage3_2026_10.md`. With the salience drive the map
formed two or more groups at most steps, and the binding changed the leader at 0.05 or
more of the steps at 3 of 3 seeds. But the selected group was rarely the light. At the
related steps it held one of the 4 centre cells at 0.696 to 0.922 of the steps, and at
seed 137 it was mostly floor (0.603 of its cells) and little light (0.151). The agent's
own ring is drawn on the centre cells in every frame. It never changes, so its contrast
is the same at every step.

The review of the drive stated this risk before the runs. No candidate had adaptation.

## Project documents read

| Document | What it adds |
|---|---|
| `docs/sensory_selection_evidence.md` | Knudsen 2018, as read there. Adaptation lowers the response to a repeated stimulus, so the winner of the midbrain map changes over time |
| `docs/decisions/2026_09_16_precision_weighted_bids.md` | The cited sources say salience habituates and reliability does not. The vision gain of `--bid-precision gain` excludes the agent's own disc by a fixed mask, 0.25 of the frame width from the centre (`models/core/precision_weighting.py`) |
| `docs/results/dark_room_senses_2026_09.md` | The audio bid of `--audio-salience surprise` is the running z-score of a prediction error, so it habituates already |
| `docs/results/reentrant_select_adapt_2026_10.md` | The workspace already has an adaptation of the winning sense, with rate 0.1 and depth 0.5 (assumed). It is not reset between episodes. Gate R1 passed with it |
| `docs/decisions/2026_10_14_map_binding_drive.md` | The salience drive and its sources |

## Sources added

| Source | Read | Statement used here |
|---|---|---|
| Dutta and Gutfreund, Front. Integr. Neurosci. 8, 1 (2014), review | Main text and figure legends, through a tool summary (about two thirds of the page text; the cut fell in the references) | Salience in the optic tectum has three parts. A target differs from its surround, it differs from its own recent history, and it is relevant to the task. Adaptation in the tectum is specific to the stimulus and to several features. Its memory is long. In Netser and colleagues (2011), one short sound lowered the response to a sound 10 to 60 s later. The review proposes a model in words. The network learns the statistics of the background, the response follows the mismatch, and each stimulus then updates the background |
| Boehnke, Berg, Marino, Baldi, Itti and Munoz, Eur. J. Neurosci. 34, 766 (2011) | Methods, results and discussion, through a tool summary (about three quarters of the page text) | Monkey superior colliculus, 98 neurons, seven flashes at one place. The peak response fell by about 50 percent in superficial transient cells, about 35 percent in sustained cells and more than 50 percent in visuomotor cells. Most of the fall came at the second flash. Shorter intervals gave a larger fall. A brighter or dimmer odd flash changed the response. A model of Bayesian surprise with two learning rates fitted the data. Position was not varied |
| Dutta, Wagner and Gutfreund, J. Neurosci. 36, 4876 (2016) | Abstract, introduction, methods and the jitter results, from the text of the article | Barn owl optic tectum. A bar of odd orientation among bars of one orientation gave a stronger response only after the tectum had adapted to the common orientation. The adaptation has two parts, one specific to position and one specific to orientation. It was present at the first odd stimulus. Bars were shown for 400 ms with an interval of 1 s |
| Reches and Gutfreund, J. Neurosci. 28, 1523 (2008) | Abstract, through a search summary | Auditory neurons of the owl tectum respond more strongly to a rare sound than to the same sound when it is frequent |

Summary of the sources. The tectum responds less to a stimulus that stays the same at
one place, and the fall is large but not complete (about 35 to 50 percent). The
adaptation is specific to the place and to the feature. It builds up quickly and lasts
long. It occurs for vision and for hearing.

## Designs compared

All designs keep the salience drive, the half maximum threshold and the grouping of the
salience form. Values marked assumed are not fitted to data.

| Design | What changes | Fit with the sources | Assumed values | Effect on the agent's ring |
|---|---|---|---|---|
| A. Adaptation for each cell and each sense | Each cell keeps a running mean of its own drive, for vision and for audio. The drive becomes drive times (1 minus depth times that mean) | Specific to place (Dutta 2016). A bounded fall, depth 0.5 against the measured 35 to 50 percent (Boehnke 2011). Vision and hearing (Reches 2008). The same form as the workspace rule of Gate R1 | rate 0.1, as in Gate R1. Depth 0.5 lies in the measured range | The ring keeps about half of its drive. A moving light keeps more |
| B. Mismatch of the features at each cell | Each cell keeps a running mean of its feature vector. The visual drive is the distance of the current features from that mean | Specific to place and to feature (Dutta 2016). The background model of Dutta and Gutfreund 2014 | rate, and a rule for the audio side | The ring gets a drive near zero. A light that stays still also falls to near zero, which is a larger fall than the 35 to 50 percent measured |
| C. Bayesian surprise for each cell | A Poisson and Gamma belief for each cell, two learning rates, drive is the surprise | The model that Boehnke 2011 fitted | two rates, a noise level, and a mapping of features to rates | As B. It is the most work, and the fit was to flashes, not to a moving scene |
| D. Mask of the agent's disc | The cells of the agent's disc get no drive, as the vision gain already does | None for adaptation. A fixed rule from the project, not from the sources | none (the existing mask) | Removed. Other constant parts of the view, such as a wall in the same place, keep their full drive |

## Recommendation for the owner

A. It is the closest match to the measured facts that this review read. The fall is
bounded and of the measured size, it is specific to place, it applies to both senses,
and it has the same form as the workspace rule that passed Gate R1. It adds one assumed
value, and that value is the one the project already uses.

B is the better match for the feature part of the adaptation. Its unbounded fall does
not agree with the measured size. D is a fix for this test bed only.

## Test bed and gate for the next runs

The owner decided on 2026-10-14 that the next gate uses 40 episodes per run, because B1
voided both forms of option B at one seed of three (53 and 54 related steps in 20
episodes). The checks and the PASS, FAIL and UNDECIDED rule of option B stay as they
are. The seeds and the full gate will be stated in the result document before the runs.

## What this review does not establish

- No design was tried on the agent.
- The agent has no clock. One step is not a known time, so the rate cannot be taken
  from the sources. The sources show a fast build-up and a long memory. One rate gives
  the same speed for both.
- The sources measured flashes and bars at fixed places. The agent's view moves with the
  agent, so a light can stay on the same cells for many steps when the agent stands
  still, and then it adapts too.
- Two of the four sources were read through tool summaries, and one at abstract level
  only.
