# Artificial Consciousness Module (ACM)

## Overview

The **Artificial Consciousness Module (ACM)** attempts to create synthetic awareness in AI systems by combining the latest AI technologies, virtual reality (VR) environments, and emotional processing. This project explores the possibility of replicating human-like consciousness in non-biological systems. By fostering an emotional connection between an ACM-equipped AI agent and humans, to reinforce adherence to **Asimov’s Three Laws of Robotics**.

[![The Consciousness AI Module](./repo_images/acm_thumbnail_1.png)](https://theconsciousness.ai)

## Core Features

1. **VR Simulations:** Realistic VR environments built with Unreal Engine 5.
2. **Multimodal Integration:** Combines vision, speech, and text models for rich understanding.
3. **Emotional Memory Core:** Processes and stores past emotional experiences.
4. **Narrative Construction:** Maintains a self-consistent internal narrative using large language models.
5. **Adaptive Learning:** Employs self-modifying code for continuous improvement.
6. **Dataset Integration:** Leverages high-quality, licensed datasets (e.g., GoEmotions, MELD) for emotion recognition and simulation tasks.

## Technologies

- **Game Engines:** Unreal Engine 5
- **AI Models:** Llama 3.3, GPT-4V, PaLI-2, Whisper
- **Vector Storage:** Pinecone, Chroma
- **Emotion Detection:** Temporal Graph Neural Networks, GoEmotions, MELD
- **Learning Frameworks:** LoRA, PEFT, RLHF

## Folder Structure

- `data/`: Datasets for emotions and simulations.
- `docs/`: Documentation for architecture, installation, datasets, and the roadmap.
  - Includes `datasets.md` and `preprocessing.md` for dataset-related details.
- `models/`: Pre-trained and fine-tuned AI models.
- `scripts/`: Utility scripts for setup, training, and testing.
- `simulations/`: VR environments and APIs for agent interactions.
- `tests/`: Unit and integration tests.

## Getting Started

### Prerequisites

- **Python 3.8 or higher**
- **CUDA Toolkit** (for GPU support)
- **Unreal Engine 5**
- **Git**

### 1. Clone the Repository

```bash
git clone https://github.com/venturaEffect/the_consciousness_ai.git
cd the_consciousness_ai
```

### Set Up a Virtual Environment

It’s recommended to use a Python virtual environment to manage dependencies.

**Linux/MacOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**

```bash
python -m venv venv
.\venv\Scripts\activate
```

### Install Dependencies

Run the provided installation script:

```bash
bash scripts/setup/install_dependencies.sh
```

Or install manually:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Download and Preprocess Datasets

Datasets are hosted externally and need to be downloaded and preprocessed locally:

1. Refer to `/docs/datasets.md` for dataset details and download links.
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

**PaLM-E Integration:**

```bash
pip install palm-e
```

**Whisper v3 Integration:**

```bash
pip install whisper-v3
```

### Running the Project

Activate your virtual environment and start the narrative engine:

```bash
python models/narrative/narrative_engine.py
```

## Usage

Detailed usage instructions for each module are in their respective directories and documentation files.

## Contributing

Contributions are welcome. Please see `docs/CONTRIBUTING.md` for details on contributing new datasets, features, or fixes.

## License

This project is licensed under the terms of the `LICENSE` file.

## Acknowledgments

- **Meta AI** for the LLaMA model
- **Google AI** for PaLM-E
- **OpenAI** for Whisper
- **Contributors** for suggesting and integrating datasets
