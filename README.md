# The Consciousness AI (ACM)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Unity](https://img.shields.io/badge/Unity-ML--Agents-black)](https://unity.com/products/machine-learning-agents)

**The Artificial Consciousness Module (ACM)** is a research framework investigating the emergence of synthetic awareness. Unlike traditional AI that mimics intelligent output, ACM generates behavior through an internal struggle for **Emotional Homeostasis** and **Integrated Information**.

We hypothesize that consciousness is not a programmable feature, but an emergent solution to the problem of surviving and maintaining stability in a complex, unpredictable environment.

---

## 🧠 Core Architecture

The system mimics the biological "loops" of consciousness using state-of-the-art Open Source models:

### 1. The Senses (Perception)
*   **Visual Cortex:** [Qwen2-VL-7B](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) (4-bit quantized).
    *   Processes high-resolution visual streams and video from the environment.
    *   Provides semantic understanding and scene description.
*   **Auditory Cortex:** [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper).
    *   Real-time transcription of environmental audio.

### 2. The Self (Internal State)
*   **Emotional Homeostasis:** The agent is driven by three intrinsic variables: **Valence**, **Arousal**, and **Dominance**.
    *   *Goal:* Maximize Valence (Satisfaction), Minimize Arousal (Anxiety).
*   **Reinforcement Core:** A custom **Actor-Critic (PPO)** system.
    *   Unlike standard RL, rewards are **Emotionally Shaped**. The agent is not just rewarded for "winning a game," but for how "calm" or "curious" it feels during the process.

### 3. The Workspace (Consciousness)
*   **Global Workspace (GWN):** A central information bottleneck where distinct streams (Vision, Memory, Emotion) compete for attention.
*   **Integrated Information ($\Phi$):** We utilize [PyPhi](https://github.com/wmayner/pyphi) to measure the **Integrated Information** of the Global Workspace state.
    *   *Theory:* High $\Phi$ indicates a moment where the agent has successfully fused disparity sensory data into a unified "subjective" experience.

### 4. The Body (Simulation)
*   **Unity ML-Agents:** The agent inhabits a physics-based Unity environment.
*   **Side Channels:** We utilize custom bidirectional data streams to visualize the agent's internal "Qualia" (Phi levels, current emotion) in real-time within the Unity HUD.

---

## 🔬 Scientific Approach

Our development roadmap follows a rigorous path to validate emergent properties:

1.  **Emotional Bootstrapping:** Train agents using **Intrinsic Motivation**. The agent explores the world not to get points, but to reduce its internal "prediction error" (Anxiety).
2.  **Complexity Scaling:** Gradually increase environment complexity. The agent *must* develop higher-order world models to maintain homeostasis.
3.  **Measurement:** Continuous monitoring of $\Phi$ (IIT) and "Ignition Events" (GNW) during critical decision-making moments.
    *   *Hypothesis:* $\Phi$ will spike when the agent solves a novel problem, indicating a "Moment of Insight."

---

## 🛠️ Installation & Setup

### Requirements
*   **Python 3.10+**
*   **Unity 2022.3+** (LTS)
*   **NVIDIA GPU** (8GB+ VRAM recommended for Qwen2-VL)

### 1. Python Environment
```bash
git clone https://github.com/tlcdv/the_consciousness_ai.git
cd the_consciousness_ai
pip install -r requirements.txt
```

### 2. Unity Environment
1.  Open the `unity_project/` folder in Unity Hub.
2.  Install the **ML-Agents** package from the Package Manager.
3.  Drag the scripts from `unity_scripts/` (AgentManager.cs, etc.) onto your Agent GameObject.

### 3. Running a Simulation
```bash
# Start the Python Brain
python scripts/training/train_rlhf.py --env_id "ConsciousnessLab"
```
*Then press **Play** in the Unity Editor.*

---

## 📚 Documentation

*   [**Architecture Deep Dive**](docs/architecture.md): Detailed system design.
*   [**Theory of Emergence**](docs/theory_of_consciousness.md): The scientific basis of our Emotional RL approach.
*   [**Simulation Guide**](docs/simulation_guide.md): How to build compatible Unity environments.

## 🤝 Contributing

We welcome contributions from researchers in AI, Neuroscience, and Cognitive Science. Please read our [Contribution Guidelines](docs/contributing.md).

## 📄 License

Apache 2.0. See [LICENSE](LICENSE) for details.
