# Which sensory signal reaches action selection when two senses compete

*Compiled 2026-10-13. An evidence review for one open decision, the default of
`--broadcast-merge`. It reads the project documents first and then six sources.
Paraphrase and citation only. No code changes with this document, and no run was made
for it.*

## The conclusion, stated first

1. No source read here describes a brain that sends the losing signal to the system
   that selects the action. The default merge (`legacy`) does that at more than 90
   percent of the active steps. It has no support in these sources.
2. The sources agree that the selected signal is amplified and routed on, and that the
   others are suppressed. That is the behaviour of `top_winner`.
3. A larger difference sits under both options. In the midbrain, vision and hearing do
   not compete with each other. They add salience to places on one map of space, and
   the places compete. In this project's workspace, a vision module and an audio module
   compete, and place has no part in the competition. So neither option reproduces the
   midbrain selection. The measured collapse of the competition under `top_winner`
   (vision at 0.990 to 1.000 of the active steps at 6 of 6 seeds) is a property of that
   design.
4. This review does not select an option. It gives the owner three, with the evidence
   for each, at the end.

## The decision and what was measured before

The workspace admits every module whose bound bid is at least 0.8 of the ignition
threshold. Three behaviours exist in the code for what the policy then receives.

| Setting | What the policy receives |
|---|---|
| `--broadcast-merge legacy` (default) | The vector of the weakest admitted module |
| `--broadcast-merge top_winner` | The vector of the strongest admitted module |
| `broadcast_mode attention_weighted` | A weighted sum of the vectors of all admitted modules |

Measured facts, from `docs/results/broadcast_carrier_2026_10.md`. Two modules are
admitted at 0.905 to 0.975 of the active steps. With the default, the policy receives
the vector of the module with the highest bid at 0.025 to 0.095 of the active steps.
With `top_winner`, vision has the highest bid at 0.990 to 1.000 of the active steps.

One more fact from the code, relevant below. Hearing enters the agent twice. The
auditory specialist gives a spatial map to the tectum, where it is fused with the
visual map by inverse effectiveness (`models/core/sensory_tectum.py`,
`_fuse_inverse_effectiveness`, and `auditory_specialist.get_spatial_for_tectum()` in
the training loop). The tectum output is the payload of the module named `vision`. The
auditory specialist also competes as the module named `audio` with its own vector.

## What the project documents say

- `docs/biological_neural_architecture_research.md` names the isthmus nuclei as the
  attention mechanism of the tectum. It says they enhance some inputs and suppress
  others, and that the sensory maps are in register with motor maps.
- `docs/preregistered_predictions.md` describes the default as a winner-take-all
  broadcast that carries the payload of the winning module. With two admitted modules
  the code does not do that. The broadcast carrier result measured it.
- `docs/consciousness_indicators_butlin.md`, indicator GWT-2, asks for a workspace of
  limited capacity with a selection mechanism. Its row notes that the dark room gates
  judge bids.

So the project documents already point to selection with suppression of the losers.
They do not answer which vector the policy must receive, because they assume that one
module wins.

## The sources

Each entry says how much was read. All were read through a text extract of the page.

### 1. Knudsen (2018), review, main text read
"Neural Circuits That Mediate Selective Attention: A Comparative Perspective," *Trends
in Neurosciences* 41, 789-805, doi:10.1016/j.tins.2018.06.006.

- A midbrain network selects by place. Its hub is the optic tectum (the superior
  colliculus in mammals), with the isthmic nuclei.
- Priority at a place combines physical salience (size, luminance, motion, novelty)
  with task relevance from the forebrain. Other senses add strength, motion and novelty
  at a place. They do not add identity.
- One isthmic nucleus (Imc) sends inhibition to all other places. The strongest place
  suppresses the rest, so the map holds the winner in a categorical way.
- Two cholinergic nuclei (Ipc, SLu) raise the gain at the selected place and
  synchronise its activity at about 25 to 60 Hz. The review calls this a tag for the
  selected stimulus. The selected signal goes on to the thalamus and the forebrain.
- The midbrain network selects across space only. Selection across senses, features,
  objects and memory is assigned to forebrain networks.
- Adaptation lowers the response to a repeated stimulus, so the winner changes with
  time.

### 2. Mysore and Knudsen (2013), experiment, author manuscript read
"A shared inhibitory circuit for both exogenous and endogenous control of stimulus
selection," *Nature Neuroscience* 16, 473-478, doi:10.1038/nn.3352.

- Barn owls, not anaesthetised. Recording in the intermediate and deep tectum. Visual
  looming dots and auditory noise bursts.
- A distant competitor suppresses the response to a stimulus in the receptive field.
- A local block of Imc at the place of the competitor removed that suppression in 16
  of 18 units. It did so for visual and for auditory competitors. Responses to a single
  stimulus stayed about the same.
- Limits stated by the authors. The path of the top-down signal to Imc is open. About
  40 percent of the experiments succeeded. A test in behaving animals was still needed.

### 3. Isa, Marquez-Legorreta, Grillner and Scott (2021), review, first two thirds read
"The tectum/superior colliculus as the vertebrate solution for spatial sensory
integration and action," *Current Biology* 31, R741-R762,
doi:10.1016/j.cub.2021.04.001.

- Visual input reaches the top layer, other senses reach deeper layers, and a motor map
  in register lies in the deepest layer.
- When two salient stimuli compete, the more prominent one drives the gaze shift.
- Output of the basal ganglia holds the deep layers under tonic inhibition, and a pause
  of that inhibition releases the movement.
- In cats and primates, a visual and an auditory stimulus from the same place help each
  other, and stimuli from different places inhibit each other.

### 4. Stanford, Quessy and Stein (2005), experiment, summary of the full text read
"Evaluating the operations underlying multisensory integration in the cat superior
colliculus," *Journal of Neuroscience* 25, 6499-6508.

- Most combined visual and auditory responses equal the sum of the two single
  responses. They exceed the sum when both stimuli are weak, and they fall below it in
  rare cases with strong stimuli. This is inverse effectiveness.

### 5. Pluta, Rowland, Stanford and Stein (2011), experiment, summary of the full text read
"Alterations to multisensory and unisensory integration by stimulus competition,"
*Journal of Neurophysiology* 106, 3091-3101.

- Cats under anaesthesia, 51 visual and auditory neurons of the deep colliculus, with a
  competitor in the other half of space.
- Responses to a combined visual and auditory stimulus resisted the competitor much
  more than responses to one sense alone. Two visual stimuli together did not.
- The authors read this as support for a priority or winner-take-all process for the
  orienting target. They state that the source of the suppression is not established.

### 6. Mashour, Roelfsema, Changeux and Dehaene (2020), review, first third read
"Conscious Processing and the Global Neuronal Workspace Hypothesis," *Neuron* 105,
776-798.

- Ignition is the sudden and exclusive activation of one subset of workspace neurons,
  and the rest of the workspace is inhibited.
- The selected content becomes available to memory, report and motor systems.
- Content that is not selected is still processed in local circuits and can have a
  limited effect on behaviour.

## What follows for the three settings

| Setting | Agreement with the sources |
|---|---|
| `legacy` | None. In sources 1, 2, 3 and 6 the signal that lost is suppressed. No source describes it as the input to action |
| `top_winner` | Agrees with the routing in sources 1, 3 and 6. The selected signal is amplified and sent on. It does not reproduce what competes, see below |
| `attention_weighted` | Agrees with sources 4 and 5 for two signals from the same place, where the responses add. It does not agree with them for signals from different places, which suppress each other |

**What competes.** In sources 1 to 5 the senses are not the competitors. Places are.
A sound and a light at one place form one stronger candidate, and a sound at another
place is a rival candidate. In this project the tectum already holds one fused map
with vision and hearing, and that fused map is the payload of the module named
`vision`. The module named `audio` then competes against it with a vector that has no
place code in the competition. By my reading this is closer to the forebrain kind of
selection in source 1 (across senses) than to the midbrain kind (across places). The
project's thesis puts the selection in the midbrain.

**Why the winner can change.** Source 1 gives adaptation as the mechanism. The
response to a repeated stimulus falls, and a new or stronger stimulus at another place
takes the map. A light that stays in view is a repeated stimulus. A sound from a place
with no light is a new one. This is relevant to the open question of why the hearing
bid stays below the vision bid under `top_winner`.

## Options for the owner

| Option | What it does | Evidence for | Cost and risk |
|---|---|---|---|
| A. Keep `legacy` as the default for now | No change. Every text keeps the statement that the gates judge bids | No biological support. It keeps every earlier run comparable | The policy keeps receiving the weaker vector |
| B. Make `top_winner` the default | The policy receives the strongest module's vector | Sources 1, 3 and 6 on routing | Under it the competition collapses to vision at 6 of 6 seeds, so the gates B to B8 do not carry over. Needs 3 seeds or more and a statement of what the new baseline is |
| C. Move the competition to places | The fused tectum map gives a priority per place. Places compete with global inhibition and adaptation. The content of the selected place goes to the policy. The separate audio competitor is removed or kept as a forebrain route | Sources 1 to 5 | A new mechanism. It needs a design document, a default-off flag and a pre-stated gate before any run |

My recommendation, for the owner to accept or reject. Keep option A as the default
until a diagnosis has run. Run that diagnosis under `top_winner`, because it is the
routing the sources support, and test adaptation of the vision bid as the first
candidate cause. Treat option C as the design that the sources describe, and decide on
it after the diagnosis.

## Limits of this review

- Six sources. Four are reviews or summaries read in part. The page extract cut
  source 1 after its main text, source 3 after about two thirds and source 6 after about
  one third.
- The experiments are in barn owls and in cats under anaesthesia. Sources 1 and 3
  compare species, and they state that parts of the mammal circuit are not resolved.
- The step from "module" in this project to "place" in the sources is my reading. No
  source discusses an artificial workspace.
- Nothing here is a measurement on the agent, and no indicator changes.
