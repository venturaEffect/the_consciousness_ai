# Artificial Consciousness Module (ACM)

[![Image frame from Blade Runner the producer of memories](./repo_images/NIQM2NDZ._for_github.png)](https://theconsciousness.ai)

## Overview

The **Artificial Consciousness Module (ACM)** is our ambitious project to explore synthetic awareness in AI systems. We're combining modern AI technologies with virtual reality environments and emotional reinforcement learning to investigate the potential emergence of consciousness-like behaviors in non-biological systems.

Our design centers around the hypothesis that consciousness-like properties are not programmed directly but **emerge** from the complex, high-order interactions between multiple specialized AI systems. We orchestrate advanced models for perception (VideoLLaMA3, Whisper), memory ([`EmotionalMemoryCore`](models/memory/emotional_memory_core.py)), emotion ([`EmotionalProcessingCore`](models/emotion/emotional_processing.py)), world modeling (DreamerV3), and narrative reasoning (LLaMA 3.3) within dynamic VR simulations (Unreal Engine 5). The [`ConsciousnessCore`](models/core/consciousness_core.py) acts as a central hub, integrating these streams of information and guiding the agent's attention and behavior based on simulated emotional states and goals. The goal is to develop AI agents that form and store emotional memories through these orchestrated experiences, leading to emergent self-awareness and potentially empathetic responses, all while adhering to Asimov's Three Laws as ethical guidelines.

[![The Consciousness AI Module](./repo_images/acm_thumbnail_1.png)](https://theconsciousness.ai)

## Core Architecture

```python
# Core components hierarchy (within models/ directory)
models/
├── memory/
│   ├── emotional_memory_core.py     # Emotional indexing and storage
│   ├── temporal_coherence.py        # Experience sequencing
│   └── consolidation.py             # Memory optimization
├── emotion/
│   ├── emotional_processing.py      # Affect handling, emotional state generation
│   ├── meta_emotional.py            # Learning related to emotions
│   └── tgnn/
│       └── emotional_graph.py       # Potential Emotional Graph Neural Network
├── core/
│   ├── consciousness_core.py        # Central orchestration, decision-making, includes AsimovComplianceFilter logic
│   ├── consciousness_gating.py      # Attention control
│   # └── asimov_compliance_filter.py  # Integrated ethical filter (logic within consciousness_core.py)
└── self_model/
    └── self_representation_core.py  # Self-representation, internal state model
# ... other model categories like integration/, evaluation/, narrative/, ace_core/
```

## Core Features

1. **Consciousness Development Through Survival**

   The design includes VR-based survival scenarios to activate attention mechanisms, inspired by how stress creates lasting memories in biological systems. We're building systems for emotional memory formation during simulated high-intensity moments, with dynamic adaptation handled via `emotional_memory_core.py`.

   ```python
   from models.memory.emotional_memory_core import EmotionalMemoryCore
   from models.core.consciousness_gating import ConsciousnessGating

   memory = EmotionalMemoryCore(config)
   consciousness = ConsciousnessGating(config)
   ```

2. **Emotional Intelligence & Empathetic Learning**

   Our emotional processing system ([`EmotionalProcessingCore`](models/emotion/emotional_processing.py), potentially using [`EmotionalGraphNetwork`](models/emotion/tgnn/emotional_graph.py)) analyzes multimodal inputs to generate internal emotional states for the agent. Crucially, these self-generated emotions are not just passive states; they directly influence the agent's learning and behavior through **emotionally-grounded reinforcement learning**.
   *   **RL Integration:** We integrate these emotional states with the DreamerV3 world model. The agent learns to predict not only environmental outcomes but also its own likely emotional responses.
   *   **Reward Shaping for Empathy:** The RL reward signal ([`EmotionalRewardShaper`](models/evaluation/consciousness_development.py)) is shaped by these internal emotional states. Positive simulated emotions (e.g., from successful social interaction, achieving goals safely) act as positive reinforcement, while negative emotions (e.g., from simulated harm, failure, social rejection) act as negative reinforcement or penalties.
   *   **Emergent Empathy Hypothesis:** By learning to navigate complex scenarios ([`docs/simulation_guide.md`](docs/simulation_guide.md)) to maximize positive internal emotional states and minimize negative ones, particularly in social contexts (using datasets like MELD, RAMAS), we hypothesize the agent will develop behaviors interpretable as understanding and responding appropriately to the emotions of others – a foundation for empathy. Meta-learning capabilities aim to accelerate adaptation to new emotional contexts.

3. **Memory Architecture**

   The memory system design includes emotional indexing using Pinecone v2, temporal coherence maintenance, and experience consolidation through the `consolidation.py` module. We've designed consciousness-weighted storage to prioritize potentially impactful memories.

   ```python
   from models.memory.consolidation import MemoryConsolidationManager
   consolidation = MemoryConsolidationManager(config)
   ```

4. **Ethical Framework & Safety**

   The system is designed to operate within the ethical boundaries defined by Asimov's Three Laws:
   1. No harm to humans through action or inaction.
   2. Obey human orders unless conflicting with First Law.
   3. Self-preservation unless conflicting with First/Second Laws.

   **Implementation:** Adherence is enforced primarily within the [`ConsciousnessCore`](models/core/consciousness_core.py) module via an integrated `AsimovComplianceFilter` logic. This filter intercepts and evaluates potential actions against the laws based on predicted outcomes and current state *before* execution. Actions violating the prioritized laws are blocked, and a safe fallback is initiated. Continuous monitoring ([`SelfAwarenessEvaluator`](models/evaluation/self_awareness_evaluation.py)) also checks for ethical alignment. *(See [`docs/ethics_framework.md`](docs/ethics_framework.md) for detailed implementation strategy)*.

5. **Narrative Foundation**

   The project uses LLaMA 3.3 for consciousness development, with plans for dynamic fine-tuning through LoRA and controlled adaptation mechanisms.

6. **Enhanced Memory Systems**

   We're designing for Pinecone as our primary vector store due to its managed nature and scalability. For applications requiring lower latency or more control, FAISS or Milvus are alternatives being considered.

### Memory Indexing

Pinecone serves as the primary vector store in our design due to its scalability and managed nature. For applications requiring lower latency or finer control over infrastructure, FAISS or Milvus are being explored as alternatives.

### World Modeling and Reinforcement Learning

The architecture incorporates DreamerV3 with emotional context weighting for world modeling. MuZero or PlaNet are being evaluated as alternatives for scenarios with specific latency or scaling requirements.

## Technologies

- **Core AI:** LLaMA 3.3, **palme (open-source PaLM-E)**, Whisper v3
- **Animation & Expression:** NVIDIA ACE, Audio2Face
- **Memory Systems:** Pinecone v2, Temporal Graph Neural Networks
- **Emotion Processing:** GoEmotions, MELD, HEU Emotion
- **Simulation:** Unreal Engine 5 with real-time physics
- **Learning:** DreamerV3, PEFT, RLHF

## Available Datasets for Training

The ACM leverages a variety of datasets for training its components. Below are some key examples. Please refer to `docs/datasets.md` (Note: This document is planned and needs to be created) for a comprehensive list, licensing details, and setup instructions.

### First-Person Interaction & Simulation Datasets

These datasets provide rich visual and interaction data, often from an egocentric perspective, suitable for training agents in simulated or real-world scenarios.

#### 1. Ego4D Dataset

- **Description**: Large-scale dataset containing 3,670+ hours of first-person video from 74 worldwide locations.
- **Features**: Daily activities, social interactions, episodic memory. Crucial for understanding human behavior from a first-person view.
- **License**: Ego4D License Agreement (Requires application).
- **Access**: Use the `ego4d` CLI tool.
  ```bash
  pip install ego4d
  ego4d --output_directory="~/ego4d_data" --datasets full_scale annotations --metadata
  ```

#### 2. Charades-Ego Dataset

- **Description**: 68,000+ video clips of daily activities, often with paired third/first person views.
- **Features**: Object/people interactions, action recognition. Useful for learning common human actions.
- **License**: Check website for terms (likely research-focused).
- **Access**: [Charades-Ego Dataset](https://allenai.org/plato/charades/)

#### 3. GTEA Gaze+ Dataset

- **Description**: First-person videos with gaze tracking during cooking activities.
- **Features**: Object manipulation, attention mapping, fine-grained action understanding. Useful for linking gaze/attention to actions.
- **License**: Check website for terms.
- **Access**: [GTEA Gaze+ Portal](http://cbs.ic.gatech.edu/fpv/)

#### 4. AI Habitat Datasets (Gibson, Matterport3D)

- **Description**: Photorealistic 3D scans of real-world indoor environments for use with the AI Habitat simulator.
- **Features**: Realistic navigation, physics-based interaction, visual understanding in 3D.
- **License**: Various (often research-focused).
- **Access**: [AI Habitat Datasets](https://aihabitat.org/datasets/)

#### 5. VirtualHome

- **Description**: Simulates daily activities in household environments with action scripts.
- **Features**: Task planning, complex object interactions, understanding household routines.
- **License**: MIT License.
- **Access**: [VirtualHome Project](http://virtual-home.org/)

### Emotion Recognition & Social Interaction Datasets

These datasets are vital for training the emotional intelligence components of the ACM.

#### 6. GoEmotions

- **Description**: Large-scale text dataset for fine-grained emotion classification.
- **Features**: Text-based emotion analysis.
- **License**: Apache 2.0.
- **Access**: [GoEmotions GitHub](https://github.com/google-research/google-research/tree/master/goemotions)

#### 7. MELD (Multimodal EmotionLines Dataset)

- **Description**: Multimodal (audio, visual, text) dialogues annotated for emotions.
- **Features**: Emotion in conversational context.
- **License**: Check website for terms.
- **Access**: [MELD Dataset GitHub](https://github.com/declare-lab/MELD)

#### 8. RAMAS (Real-world Affective Measurement...)

- **Description**: Multimodal recordings (video, audio, physiology) of natural small group social interactions.
- **Features**: Group dynamics, empathy, social signal processing.
- **License**: Requires application/agreement.
- **Access**: [RAMAS Project](https://ramas-project.github.io/)

#### 9. MSP-IMPROV

- **Description**: Multimodal dataset of acted dyadic emotional interactions.
- **Features**: Dyadic interaction analysis, emotion expression.
- **License**: Check website for terms.
- **Access**: [MSP-IMPROV Dataset](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Improv.html)

#### 10. IEMOCAP

- **Description**: Widely used multimodal dataset of dyadic emotional interactions.
- **Features**: Benchmark for multimodal emotion recognition.
- **License**: Requires license agreement.
- **Access**: [IEMOCAP Dataset](https://sail.usc.edu/iemocap/)

### Dataset Usage

- Detailed setup instructions in `docs/datasets.md` (Note: This document is planned and needs to be created)
- Data preprocessing guidelines in `docs/preprocessing.md`
- Example notebooks potentially in `notebooks/dataset_examples/` (if created)

## Real-Time Integration with VideoLLaMA3

### Overview

The design incorporates VideoLLaMA3 for processing live video or frames from simulations in real time, enabling AI agents to interpret their environment dynamically, especially in Unreal Engine simulations. (Detailed setup in `docs/installation.md` or a dedicated integration guide is recommended).

### Requirements

- High-performance GPU (e.g., NVIDIA RTX 40-series) or TPU for low-latency inference.
- (Optional) Tools like NVIDIA TensorRT or TorchServe for additional optimization.

### Implementation Steps

1. **Frame Streaming**  
   Capture frames in real time (e.g., from Unreal Engine) and send them to your Python process via sockets or shared memory.

2. **VideoLLaMA3 Processing**  
   In Python, use the methods in `VideoLLaMA3Integration` (e.g., `process_stream_frame`) to process each frame:

   ```python
   frame = ...  # Captured from simulation
   context = video_llama3_integration.process_stream_frame(frame)
   ```

3. **Emotional Memory & Consciousness Updates**

The output can be stored in `EmotionalMemoryCore` or forwarded to `ConsciousnessCore` to trigger reinforcement learning or consciousness updates.

4. **Performance Considerations**

- Use smaller resolutions or frame skipping for higher FPS.
- Keep total inference latency under ~100ms for near real-time interaction.

- **Latency Mitigation:**
  - Lower frame resolution or implement frame skipping.
  - Leverage GPU optimizations such as NVIDIA TensorRT.
  - Monitor total inference latency and aim for below ~100 ms.

**Example**

```python
# Inside your simulation loop
while simulation_running:
    frame = unreal_engine_capture()  # Or another method
    output = video_llama3_integration.process_stream_frame(frame)
    consciousness_core.update_state(output)
```

## Measuring Consciousness Evolution

The project includes a framework for analyzing the consciousness development over time:

1.  ConsciousnessMonitor
    Designed to track internal metrics at runtime to compute a consciousness score.

2.  ConsciousnessMetrics
    Implements functions like `evaluate_emotional_awareness`, `evaluate_memory_coherence`, and `evaluate_learning_progress` to assess different dimensions of AI consciousness.

The system architecture allows for logging and plotting these metrics over time to evaluate the development of synthetic awareness.

## Self-Awareness and Self-Representation

The ACM implements self-awareness and self-representation mechanisms based on several cognitive theories:

### Core Components

1. **Self-Representation Core**  
   A dynamic model that maintains and updates the system's representation of itself, including emotional states, attention focus, and confidence levels.

2. **Meta-Learning System**  
   Enables the system to adapt its learning processes based on experience, emotional states, and consciousness development.

3. **Self-Awareness Evaluation**  
   Tracks development of self-awareness across multiple dimensions, including emotional recognition, behavioral consistency, and metacognitive accuracy.

### Theoretical Foundations

- **Global Workspace Theory**: Implements a central workspace where information becomes globally available to all cognitive processes
- **Attention Schema Theory**: Maintains dynamic models of the system's own attention processes
- **Higher-Order Metacognition**: Enables reflection on the system's own knowledge and beliefs

### Usage Example

```python
# Initialize self-representation core
from models.self_model.self_representation_core import SelfRepresentationCore

self_model = SelfRepresentationCore(config)

# Update self-model with new experience
update_result = self_model.update_self_model(
    current_state=perception_state,
    attention_level=attention_score,
    timestamp=current_time
)

# Evaluate self-awareness development
from models.evaluation.self_awareness_evaluation import SelfAwarenessEvaluator

evaluator = SelfAwarenessEvaluator(config)
metrics = evaluator.evaluate_self_awareness(
    self_model_state=self_model.get_current_state(),
    interaction_history=recent_interactions,
    emotional_context=emotional_state
)
```

## Visualizing Consciousness Development

The project includes a ConsciousnessDashboard module to track and display consciousness development metrics in real time. This Flask server displays metrics (e.g., integrated information, PCI, emotional awareness) as a line graph that updates over time:

```bash
python models/evaluation/consciousness_dashboard.py
```

The dashboard will be accessible at [http://localhost:5000](http://localhost:5000) to visualize the ACM's performance metrics.

### 1. Clone the Repository

```bash
git clone https://github.com/venturaEffect/the_consciousness_ai.git
cd the_consciousness_ai
```

### 2. Set Up a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

### 3. Install Dependencies

Run the provided installation script: (See `docs/installation.md` for more details and model-specific setup)

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Unreal Engine Setup

Install or build Unreal Engine 5 manually from the Epic Games launcher or source code (not from PyPI).
Refer to Unreal Engine Docs [Unreal Engine Docs](https://docs.unrealengine.com/) for additional details.

## Folder Structure

- `data/`: Datasets for emotions and simulations.
- `docs/`: Documentation for architecture, installation, datasets, and the roadmap.
  - Includes `datasets.md` (Note: This document is planned) and `preprocessing.md` for dataset-related details.
- `models/`: Pre-trained and fine-tuned AI models.
- `scripts/`: Utility scripts for setup, training, and testing.
- `simulations/`: VR environments and APIs for agent interactions.
- `tests/`: Unit and integration tests.

### Usage

Refer to the subdirectories (`/docs/` and `/models/`) for more detailed instructions. (Detailed model setup and usage can be found in `docs/installation.md` and specific model integration guides).

### Contributing

We welcome contributions. Please see `docs/contributing.md`.

### License

This project is licensed under the terms of the `LICENSE` file.

### Download and Preprocess Datasets

Datasets are hosted externally and need to be downloaded and preprocessed locally:

1. Refer to `/docs/datasets.md` (Note: This document is planned) for dataset details and download links.
2. Follow the preprocessing instructions in `/docs/preprocessing.md` to prepare datasets for use.

Example:

```bash
python scripts/utils/preprocess_emotions.py --input /path/to/raw/data --output /path/to/processed/data
```

### Authenticate with Hugging Face

LLaMA 3.3 is not distributed via pip. You need to download model weights from Hugging Face.  
Sign up or log in at [Hugging Face](https://huggingface.co/settings/tokens) to obtain a token.

```bash
huggingface-cli login
```

Follow the prompts to enter your token.

### Download the LLaMA 3.3 Model

The model weights download automatically on first use. Alternatively, manually download:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "meta-llama/Llama-3.3-70B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_auth_token=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_auth_token=True
)
```

### GPU Support

LLaMA 3.3 is large and requires a GPU (16 GB VRAM recommended) and CUDA installed.

### bitsandbytes Library

Install bitsandbytes for reduced memory usage:

```bash
pip install bitsandbytes
```

### Unreal Engine Prerequisites

Install Unreal Engine 5 and its prerequisites.

**Linux example:**

```bash
sudo apt-get update
sudo apt-get install -y build-essential clang
```

For Windows and macOS, refer to [Unreal Engine Docs](https://docs.unrealengine.com/).

### Setting Up Other Models

(For detailed setup of these and other models, please refer to `docs/installation.md` or specific integration guides.)

**PaLM-E Integration:**

```bash
pip install palme
```

**Whisper v3 Integration:**

```bash
pip install whisper-v3
```

### Integration with VideoLLaMA3

(This section might be better placed in `docs/installation.md` or a dedicated integration guide.)

To integrate VideoLLaMA3 into the ACM project, follow these steps:

1. **Clone the VideoLLaMA3 Repository:**
   ```bash
   git clone https://github.com/DAMO-NLP-SG/VideoLlaMA3.git
   ```

# ACE Integration with ACM

## Overview

The project now integrates NVIDIA's Avatar and Chat Engine (ACE) with our Artificial Consciousness Module (ACM) to enable:

- Realistic avatar animation driven by emotional states
- Natural conversational interactions
- Dynamic facial expressions and gestures

## Components

### ACE Integration

- Audio2Face real-time facial animation
- Emotion-driven animation control
- Natural language processing for conversations

### ACM Components

- Consciousness Core for high-level cognition
- Emotional Memory for experience accumulation
- Consciousness Gating for meta-awareness (attention control)
- World Model for predictive processing

## Configuration

See [`ace_integration/`](ace_integration/) for setup and configuration files.

### Running the Project

Activate your virtual environment and start the narrative engine:

```bash
python models/narrative/narrative_engine.py
```

## Usage

Detailed usage instructions for each module are in their respective directories and documentation files.

## Contributing

Contributions are welcome. Please see `docs/contributing.md` for details on contributing new datasets, features, or fixes.

## License

This project is licensed under the terms of the `LICENSE` file.

## Acknowledgments

- **Meta AI** for the LLaMA model
- **Google AI** for PaLM-E and DreamerV3
- **OpenAI** for Whisper
- **Contributors** for suggesting and integrating datasets

## Based on research in:

- MANN architecture
- Holonic consciousness
- Emotional memory formation
- Survival-driven attention

## Citation & Credits

If you use ACM in your research, please cite:

```bibtex
@software{acm2024consciousness,
    title = {Artificial Consciousness Module (ACM)},
    author = {The Consciousness AI Team},
    year = {2024},
    publisher = {GitHub},
    url = {https://github.com/venturaEffect/the_consciousness_ai},
    version = {1.0.0}
}
```

## License

<img alt="License: Custom MIT Non-Commercial" src="https://img.shields.io/badge/License-MIT_NC-blue.svg">

This project is licensed under a modified MIT License with non-commercial use restrictions.

**Usage Requirements:**

When using this code, you must include the following:

This project includes code from the Artificial Consciousness Module (ACM) Copyright (c) 2024 The Consciousness AI [`https://github.com/venturaEffect/the_consciousness_ai`](https://github.com/venturaEffect/the_consciousness_ai)

Licensed under Modified MIT License (non-commercial)

For commercial licensing inquiries, please contact: <a href="mailto:info@theconsciousness.ai">info@theconsciousness.ai</a>
