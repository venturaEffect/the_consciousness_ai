# Pretrained model candidates and the compute cost of the running agent

Date: 2026-10-13, measurements added 2026-10-14. Second version. The first version
held what was read and counted. This version adds the cost of 9 models measured on this
machine and one content check for the vision models.

## Result, limits first

1. **One model FAILED to run.** `HuggingFaceTB/SmolVLM2-256M-Video-Instruct` stopped
   with an import error at all 3 settings, because the package `num2words` is not
   installed. It has no measured number. The other 9 models ran.
2. **Cost is measured, quality is not.** Each number is one run per setting at batch
   size 1 on a laptop GPU. No quality score is given for the text, image-text and
   embedding models. For vision there is one decode check on the DMTS frames of the
   project. Its two predictions hold.
3. **The training loop loads no pretrained model.** The agent behind every gate result
   has 7,791,175 parameters, which is 31.2 MB at 32-bit floats. A change of pretrained
   model changes no result in `docs/results/` and does not lower the compute of that
   agent.
4. **The four pretrained models sit in optional modules.** For each role there is a
   candidate with an Apache 2.0 or MIT declaration that has 3 to 32 times fewer
   parameters. One model, Qwen3.5-0.8B, can take two roles.
5. **The target hardware does not run these models.** The p-bit chip that the
   thermodynamic track emulates samples a sparse Ising model. Its vendor plans early
   access for 2027. A transformer or a vision transformer does not map to it. The parts
   of this repo that map are the p-bit emulator and the spiking bridge.
6. **The biology the project follows points away from a deep vision transformer for
   the tectum.** The optic tectum takes direct input from the retina and codes
   location, motion, size and looming. The review used here assigns object
   recognition to the retino-thalamic pathway.

## What runs today

Built from the recorded configuration of a Gate B7 run (`config.json`, dark room,
audio on) with `init_components`, on Python 3.8.3. Every torch module reachable from
the components was counted once.

| Component | Parameters |
|---|---|
| SensoryTectum (includes the convolutional vision encoder and the RSSM) | 6,315,804 |
| AuditorySpecialist | 467,147 |
| SemanticPathway (no input in the loop, so it bids 0) | 393,984 |
| ConsciousnessGate | 267,045 |
| ActionSelectionCore | 223,244 |
| RNDCuriosity | 98,816 |
| Reward predictor | 16,513 |
| EmotionalRewardShaper | 8,577 |
| GlobalWorkspace | 45 |
| **Total** | **7,791,175** |

`transformers` was not loaded by this build. On the default path the vision encoder is
`RetinotopicConvStack`.

Cost of one run of this agent, measured on the result machine (RTX 4090 Laptop GPU,
16 GB):

| Quantity | Value | How |
|---|---|---|
| Wall time for 2000 steps with learning and session recording | 230 s, so at most 0.115 s per step | start and end time of one Gate B7 control run, start-up included |
| GPU board power during the run | mean 20.3 W, range 7.2 to 40.1 W | `nvidia-smi`, one sample every 5 s, 25 samples in the second half of the run |
| GPU board power with no run | 4.1 to 6.6 W | 4 samples |
| GPU memory in use on the board during the run | 1893 MiB | the same samples, all processes |

CPU power was not measured. These numbers describe a laptop GPU. They are not an
estimate for any other device.

## The four roles and the candidates

Records read from huggingface.co on 2026-10-13. "Licence" is the declaration in the
model card. "File" says if the repository also holds a licence file.

### Spatial vision (optional backbone of the retinotopic encoder)

| Model | Parameters | Weights | Licence | File | Note |
|---|---|---|---|---|---|
| Convolutional stack, in repo (runs today) | part of the tectum count | none | project licence | | no download |
| `facebook/dinov2-base` (current option) | 86,580,480 | 346 MB | apache-2.0 | no | patch 14, 16x16 grid at 224 px |
| `facebook/dinov2-with-registers-small` | 22,058,112 | 88 MB | apache-2.0 | no | same patch size, 3.9 times fewer parameters |
| `facebook/dinov2-small` | 22,056,576 | 88 MB | apache-2.0 | no | |
| `timm/vit_pe_core_small_patch16_384.fb` | 23,783,168 | 95 MB | apache-2.0 | no | image and text contrastive training |
| `timm/vit_pe_core_tiny_patch16_384.fb` | 6,141,056 | 25 MB | apache-2.0 | no | 14 times fewer parameters |
| `timm/vit_pe_spatial_base_patch16_512.fb` | 86,433,792 | 346 MB | apache-2.0 | no | same size as the current option |

Not accepted under the licence rule: `facebook/dinov3-vits16-pretrain-lvd1689m`
(21,596,544 parameters) and `facebook/dinov3-convnext-tiny-pretrain-lvd1689m`
(27,820,128). Their licence is the DINOv3 licence, a vendor licence, and access is
gated. `nvidia/C-RADIOv3-B` (98,254,857) uses the NVIDIA open model licence.

### Semantic vision (optional, feeds a 1536-D embedding to the semantic pathway)

| Model | Parameters | Weights | Licence | File | Note |
|---|---|---|---|---|---|
| `Qwen/Qwen2-VL-7B-Instruct` (current option) | 8,291,375,616 | 16,583 MB | apache-2.0 | yes | no caller in the repo |
| `Qwen/Qwen3-VL-2B-Instruct` | 2,127,532,032 | 4,255 MB | apache-2.0 | no | 3.9 times fewer parameters |
| `Qwen/Qwen3-VL-Embedding-2B` | 2,127,532,032 | 4,255 MB | apache-2.0 | no | built to give an embedding, which is what the pathway takes |
| `Qwen/Qwen3.5-2B` | 2,274,069,824 | 4,548 MB | apache-2.0 | yes | text and image |
| `Qwen/Qwen3.5-0.8B` | 873,438,784 | 1,747 MB | apache-2.0 | yes | text and image, 9.5 times fewer parameters |
| `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | 507,482,304 | 2,030 MB | apache-2.0 | no | |
| `HuggingFaceTB/SmolVLM2-256M-Video-Instruct` | 256,484,928 | 1,026 MB | apache-2.0 | no | 32 times fewer parameters |
| `microsoft/Florence-2-base` | 231,567,705 | 463 MB | mit | yes | captioning and grounding, not a chat model |

Not accepted under the licence rule: `LiquidAI/LFM2-VL-450M` (LFM Open License v1.0).
Too large for this purpose: `google/gemma-4-E2B-it` has an apache-2.0 declaration and
5,123,178,051 parameters in the record.

### Narrative text (optional, `models/narrative/narrative_engine.py`)

| Model | Parameters | Weights | Licence | File | Note |
|---|---|---|---|---|---|
| `Qwen/Qwen2.5-0.5B` (current default) | 494,032,768 | 988 MB | apache-2.0 | yes | |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | 361,821,120 | 724 MB | apache-2.0 | no | |
| `HuggingFaceTB/SmolLM2-135M-Instruct` | 134,515,008 | 269 MB | apache-2.0 | no | 3.7 times fewer parameters |
| `Qwen/Qwen3-0.6B` | 751,632,384 | 1,503 MB | apache-2.0 | yes | larger than the current default |
| `Qwen/Qwen3.5-0.8B` | 873,438,784 | 1,747 MB | apache-2.0 | yes | larger, and it can also take the semantic vision role |

Not accepted under the licence rule: `google/gemma-3-270m` (Gemma terms, gated),
`LiquidAI/LFM2.5-230M` and `LiquidAI/LFM2.5-350M` (LFM Open License v1.0).

### Memory embedding (optional, `models/memory/emotional_memory_core.py`)

| Model | Parameters | Weights | Licence | File | Note |
|---|---|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` (current) | 22,713,728 | 91 MB | apache-2.0 | no | |
| `minishlab/potion-base-8M` | 7,559,168 | 30 MB | mit | no | static table, no transformer pass at run time |
| `minishlab/potion-base-32M` | 32,302,592 | 129 MB | mit | no | static table |
| `ibm-granite/granite-embedding-30m-english` | 30,295,296 | 61 MB | apache-2.0 | no | |
| `MongoDB/mdbr-leaf-ir` | 22,565,376 | 91 MB | apache-2.0 | no | same size as the current model |
| `Snowflake/snowflake-arctic-embed-xs` | 22,565,376 | 90 MB | Apache 2 stated in the card text, no licence field | no | |

Not accepted under the licence rule: `google/embeddinggemma-300m` (Gemma terms, gated).
Much larger: `Qwen/Qwen3-Embedding-0.6B` (595,776,512) and
`microsoft/harrier-oss-v1-270m` (268,098,176).

## Measured on this machine

Run of 2026-10-13, 23:52 to 00:00, with
`scripts/analysis/benchmark_pretrained_models.py` at revision `031f071`. Python 3.12.10,
torch 2.14.1+cu126, transformers 5.19.0, sentence-transformers 6.1.0, timm 1.0.30,
model2vec 0.10.0, scikit-learn 1.9.1, RTX 4090 Laptop GPU. The weights were in the
local cache and the run was offline. The predictions of the vision check were pushed
in that revision before any model ran. Source of every number below:
`runs/model_benchmark/benchmark.json`.

**FAILED to run.** `HuggingFaceTB/SmolVLM2-256M-Video-Instruct`, at all 3 settings. The
error text is "ImportError: Package `num2words` is required to run SmolVLM processor".

**How to read the cost tables.** One load per setting. Each call is timed after 2
warm-up calls, for at least 8 seconds and at least 3 calls. A time is the median, with
the range over the calls in brackets. The batch size is 1. GPU power is the board power
from `nvidia-smi` during the timed calls, and the board read 7.6 W with no load. CPU
power was not measured. GPU memory is the largest amount that torch allocated. Each
pair of values is float32 then float16.

### Cost of the vision models, one image, one forward pass

| Model | Parameters | Input | CPU float32 | GPU float32 | GPU float16 | GPU memory | GPU power |
|---|---|---|---|---|---|---|---|
| `facebook/dinov2-base` (current option) | 86,580,480 | 224 px | 140.4 ms (134.8 to 145.4) | 9.5 ms (5.2 to 22.7) | 12.7 ms (5.5 to 27.6) | 366 / 196 MB | 105.5 / 40.9 W |
| `facebook/dinov2-with-registers-small` | 22,058,112 | 224 px | 69.9 ms (63.7 to 90.2) | 12.8 ms (5.7 to 30.5) | 11.1 ms (5.4 to 23.8) | 104 / 59 MB | 38.8 / 27.3 W |
| `timm/vit_pe_core_tiny_patch16_384.fb` | 6,141,056 | 384 px | 94.5 ms (75.2 to 141.7) | 16.2 ms (8.4 to 36.5) | 15.9 ms (8.0 to 29.7) | 40 / 25 MB | 37.1 / 24.8 W |

- On the CPU the small DINOv2 needs half the time of the base model.
- On the GPU the three medians are between 9.5 and 16.2 ms and the ranges overlap. At
  batch size 1 the GPU time does not separate the models.
- The PE model has the fewest parameters and is not the fastest. Its input is 384 px,
  which is 576 patches against 256. I did not test if that is the reason.
- The 105.5 W value belongs to the first GPU load of the run and was not repeated. It
  is 2.7 times the next vision value. I do not use it.

### Cost of the text models, one prompt, 32 new tokens, greedy

| Model | Parameters | CPU float32 | GPU float32 | GPU float16 | GPU memory | GPU power |
|---|---|---|---|---|---|---|
| `Qwen/Qwen2.5-0.5B` (current default) | 494,032,768 | 3246 ms (3200 to 3324) | 1683 ms (1617 to 1732) | 1542 ms (1181 to 1670) | 1992 / 1010 MB | 27.4 / 26.9 W |
| `HuggingFaceTB/SmolLM2-135M-Instruct` | 134,515,008 | 2294 ms (2264 to 2524) | 1722 ms (1522 to 1840) | 1762 ms (1396 to 1814) | 560 / 281 MB | 30.1 / 24.5 W |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | 361,821,120 | 3166 ms (3112 to 3286) | 1506 ms (1196 to 1581) | 1515 ms (1388 to 1602) | 1468 / 751 MB | 29.3 / 29.1 W |
| `Qwen/Qwen3.5-0.8B`, loaded as a text model | 752,393,024 | 6078 ms (6010 to 6193) | 2056 ms (1762 to 2463) | 2058 ms (1919 to 2390) | 3052 / 1570 MB | 28.5 / 27.2 W |

### Cost of the image and text models, one 224 px image with a prompt, 32 new tokens

| Model | Parameters | CPU float32 | GPU float32 | GPU float16 | GPU memory | GPU power |
|---|---|---|---|---|---|---|
| `Qwen/Qwen3.5-0.8B` | 852,985,920 | 6863 ms (6813 to 6874) | 2099 ms (2039 to 2264) | 2365 ms (2076 to 2505) | 3467 / 1788 MB | 31.6 / 27.2 W |
| `HuggingFaceTB/SmolVLM2-256M-Video-Instruct` | FAILED | FAILED | FAILED | FAILED | | |

- Each row of these two tables has 3 to 6 timed calls.
- On the GPU every text model needs 1.5 to 2.1 s for 32 tokens. The smallest model is
  not faster there (1722 ms against 1683 ms for the current default).
- On the CPU the smallest model is 1.4 times faster than the current default.
- The clear difference is memory. SmolLM2-135M allocates 281 MB at float16 and the
  current default 1010 MB.
- Qwen3.5 ran on slower code. The library reported that two optional kernel packages
  (`flash-linear-attention` and `causal_conv1d`) are not installed and that its
  replacement code is correct but much slower. So the Qwen3.5 times are upper limits.
- The parameter count of Qwen3.5-0.8B is 752,393,024 as a text model and 852,985,920
  as an image and text model. The public record says 873,438,784. I did not find the
  reason for the difference.

### Cost of the embedding models, one sentence

| Model | Parameters | CPU float32 | GPU float32 | GPU float16 | GPU power |
|---|---|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` (current) | 22,713,216 | 17.3 ms (13.8 to 28.7) | 7.1 ms (5.2 to 19.5) | 7.2 ms (5.2 to 18.7) | 20.2 / 14.6 W |
| `minishlab/potion-base-8M` | 7,559,168 | 0.38 ms (0.3 to 1.9) | 1.0 ms (0.5 to 9.6) | 0.8 ms (0.5 to 9.4) | 9.8 / 10.0 W |
| `ibm-granite/granite-embedding-30m-english` | 30,295,296 | 27.8 ms (18.8 to 44.4) | 6.9 ms (5.1 to 20.1) | 7.6 ms (5.3 to 24.5) | 12.5 / 10.1 W |

- The static table model on the CPU is 45 times faster than the current model on the
  CPU, and faster than every GPU row.
- GPU memory is not reported for these three models. The float16 reading is above the
  float32 reading for each of them, so the reading includes memory of an earlier row.

**Two readings that are not reported.** The growth of process memory is negative at 6
rows, because all models ran in one process. The load time of the first model includes
the import of the libraries.

### Content check for the vision models. Both predictions hold

280 DMTS trials, 67 distinct stimuli, 6 shapes and 6 colours, so chance is 0.167. The
linear probe reads the label from the patch grid averaged to 1x1 and to 4x4 cells.
Whole stimuli are held out, 5 splits, 30 percent of the stimuli on the test side.

| Prediction, written before the run | Result |
|---|---|
| V1. For the two DINOv2 models, shape from the 4x4 grid is above 0.50 at 5 of 5 splits | HOLDS. Lowest value 0.829 (base) and 0.855 (small) |
| V2. For the same two models, colour from the 4x4 grid is above 0.50 at 5 of 5 splits | HOLDS. Lowest value 0.750 (base) and 0.579 (small) |

| Features | Grid | Dimensions | Shape, 5 splits | Colour, 5 splits |
|---|---|---|---|---|
| Pixels of the frame, 16x16 | 1x1 | 3 | 0.114, 0.053, 0.127, 0.000, 0.000 | 1.000 at 5 of 5 |
| Pixels of the frame, 16x16 | 4x4 | 48 | 0.114, 0.053, 0.191, 0.000, 0.000 | 1.000 at 5 of 5 |
| `dinov2-base` | 1x1 | 768 | 1.000 at 5 of 5 | 0.977, 0.961, 1.000, 0.976, 0.711 |
| `dinov2-base` | 4x4 | 12,288 | 1.000, 0.829, 1.000, 1.000, 1.000 | 0.955, 1.000, 0.964, 0.812, 0.750 |
| `dinov2-with-registers-small` | 1x1 | 384 | 1.000 at 5 of 5 | 0.864, 0.934, 1.000, 0.906, 0.539 |
| `dinov2-with-registers-small` | 4x4 | 6,144 | 1.000, 0.855, 1.000, 1.000, 1.000 | 0.864, 0.934, 1.000, 0.788, 0.579 |
| `vit_pe_core_tiny_patch16_384.fb` | 1x1 | 192 | 1.000, 0.961, 0.891, 0.788, 1.000 | 1.000, 1.000, 1.000, 1.000, 0.921 |
| `vit_pe_core_tiny_patch16_384.fb` | 4x4 | 3,072 | 1.000, 0.961, 0.891, 0.882, 1.000 | 1.000, 1.000, 1.000, 0.976, 0.882 |

- The averaged pixels give the colour at every split and do not give the shape.
- All three pretrained models give the shape of a stimulus that the probe did not see
  in training, at every split.
- The two DINOv2 models give the colour less well than the averaged pixels. The lowest
  split is 0.539.
- No prediction was made for the PE model and the pixels. Those rows are description.
- For reference, `docs/results/thermodynamic_survival_2026_10.md` reports that the
  trained tectum of this project reads the shape of an unseen stimulus at 0.163 from
  `obs_map` with a linear probe. That number comes from another probe and one other
  split. The two were not measured side by side, so this document makes no comparison.
- This is a decode on a closed task with 67 stimuli. It is not evidence about
  consciousness, and it moves no indicator.

## What the sources say

Each item says how much of the source was read in this study.

**Biology of the tectum (review, main text read).** Isa, Marquez-Legorreta, Grillner
and Scott, Current Biology 31, R741 (2021). The superficial layers of the tectum take
their main input from retinal ganglion cells. Tectal cells respond to location, motion,
direction, size and looming. The review assigns object recognition mainly to the
retino-thalamic pathway and not to the tectum. Consequence for this project. The
encoder of the tectum stream should give a retinotopic map of simple features. A deep
vision transformer trained for object semantics fits the role that the project gives to
the optional semantic pathway, not the tectum. The project's own design record
(`docs/isomorphic_visual_mapping_research.md`) already notes that DINOv2 features are
semantic and spatial at once and not purely retinotopic.

**Brain alignment of vision models (abstracts read only).** Raugel and colleagues
(arXiv 2508.18226) report that larger DINOv3 models with more training are more
similar to human brain recordings, and that similarity to early sensory cortex comes
first in training. Höfling and colleagues (arXiv 2604.21780, ICLR 2026) report that
common alignment scores separate models weakly and that top-ranked models often do not
reproduce the pattern of relations between brain regions. Deb and colleagues
(arXiv 2501.16396) report a topographic loss for vision and language models. None of
these gives a score for the small models in the tables above. So brain alignment is
not used to rank the candidates here.

**What p-bit hardware runs (paper and vendor pages read).** Jelinčič and colleagues
(arXiv 2510.23972) propose a probabilistic computer for denoising models built from
sparse Boltzmann machines with binary variables on a grid. The results come from a GPU
simulation. The energy figure is a model (about 2 fJ per cell update), and the paper
describes its claim of about 10,000 times less energy than a GPU as a projection for a
small image benchmark. The vendor page for the Z1 chip states 269,568 p-bits, 16
neighbours per p-bit, sampling above 50 MHz and less than 1 W per die, with early
access planned for 2027. The vendor's post of 2026-09-04 describes a sparse
transformer family for this chip. In its estimate the chip runs the sparse layers and
a companion processor runs the rest, and the companion processor takes more than 95
percent of the estimated energy. All of these are estimates and not measurements on a
complete device.

**Small satellite computers (table read from the source).** NASA, Small Spacecraft
Avionics, chapter 8 of the State-of-the-Art report, 2026 edition, Table 8-4. A Jetson
Orin Nano board is listed at 6 to 11 W in active mode with a total dose figure of 30
krad, and the orbit flown is given as to be determined. A Jetson Xavier NX board is listed at 7 W and flown in low
Earth orbit. A Jetson Orin system is listed at 130 W peak and less than 15 W idle. The
running agent of this project used a mean of 20.3 W of GPU board power on a laptop GPU,
with no effort to lower it.

## Recommendation for each role

These are proposals. They are not selections. No model id in the code changes with this
document.

| Role | Proposal | What the measurement adds |
|---|---|---|
| Spatial vision | Keep the convolutional stack as the encoder of the training loop. If a pretrained option stays in the code, change it to `facebook/dinov2-with-registers-small` | The small model needs half the CPU time and 3.5 times less GPU memory than the current option. On the shape check it has the same range. On the colour check its lowest split is 0.579 against 0.750. The stack is what the results use, and by my reading of the review it fits the tectum role as a shallow map of simple features |
| Semantic vision | No proposal yet | `Qwen/Qwen3.5-0.8B` ran with 1788 MB at float16 and about 2.1 s for 32 tokens on the GPU. The smaller candidate FAILED to run. The current option has no caller, so the first decision is if the role stays |
| Narrative text | No proposal yet | A smaller model saves memory and CPU time. It saves no GPU time at batch size 1. The quality of the text was not measured, and the role is not used by the training loop |
| Memory embedding | Measure the retrieval quality of `minishlab/potion-base-8M` on the memory use of the project before a change | It is 45 times faster on the CPU and needs no GPU. Its quality for this role was not measured |

## Limits

- The quality of the text, image-text and embedding models was not measured, so
  nothing here says that a smaller model is good enough for its role.
- One run per setting, batch size 1, one machine. The ranges are over calls of one
  load. They are not ranges over repeated runs.
- The times and the power values describe a laptop GPU with Python overhead. They are
  not an estimate for a satellite computer or for a p-bit device.
- GPU power is the board power. It includes the 7.6 W that the board reads with no
  load, and it excludes the CPU.
- The licence column is the declaration of the publisher. It is not legal advice. Four
  of the accepted candidates have a licence file in the repository, and the others have
  only the declaration.
- The parameter count of the agent covers modules reachable within 4 attribute levels
  of the built components.
- The search for candidates used the public model index and four web searches. A
  model that neither found is missing.
- The project licence badge says non-commercial. The licence rule here applies to the
  models only.

## Downloads made, and what is open

The owner approved the download list on 2026-10-13. The weights are in the Hugging Face
cache of the machine.

| File set from huggingface.co | Size on disk |
|---|---|
| `facebook/dinov2-base`, `facebook/dinov2-with-registers-small`, `timm/vit_pe_core_tiny_patch16_384.fb` | 346 + 88 + 25 MB |
| `Qwen/Qwen2.5-0.5B`, `HuggingFaceTB/SmolLM2-135M-Instruct`, `HuggingFaceTB/SmolLM2-360M-Instruct` | 1000 + 273 + 727 MB |
| `Qwen/Qwen3.5-0.8B`, `HuggingFaceTB/SmolVLM2-256M-Video-Instruct` | 1770 + 1031 MB |
| `minishlab/potion-base-8M`, `ibm-granite/granite-embedding-30m-english` | 31 + 63 MB |

The script also needs the packages `torchvision`, `timm`, `model2vec` and
`sentence-transformers`.

Open items, each for the owner.

1. SmolVLM2 needs the package `num2words`. With it the 3 failed rows can be measured.
2. The trained tectum and the pretrained features were not measured on the same splits.
   A read-only probe can do that. The project rule against a non-biological control
   encoder stays in force, so this would be a measurement and not a change of encoder.
3. The quality of a smaller text model and of the static embedding model for their
   roles in this project.
