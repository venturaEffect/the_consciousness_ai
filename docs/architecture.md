# Architecture of the Artificial Consciousness Module

## Overview

The ACM architecture is designed not to explicitly code consciousness, but to foster its **emergence** from the synergistic orchestration of multiple advanced AI components. The core principle is that synthetic awareness arises from the complex interplay between perception, memory, emotion, prediction, and action within a rich, interactive environment. Key architectural pillars facilitating this emergence include:

1.  **Integrated Perception & World Modeling:** Real-time processing of multimodal inputs (video, audio, text) using models like VideoLLaMA3 and Whisper, feeding into a predictive world model (DreamerV3 enhanced with emotional context) to anticipate environmental dynamics.
2.  **Emotionally-Grounded Reinforcement Learning:** An RL core (DreamerV3) where rewards are shaped not just by task success but crucially by internally generated emotional states ([`EmotionalProcessingCore`](models/emotion/emotional_processing.py)) arising from simulated experiences (especially survival/social scenarios). This aims to develop intrinsic motivation and potentially empathy.
3.  **Dynamic Emotional Memory:** A sophisticated memory system ([`EmotionalMemoryCore`](models/memory/emotional_memory_core.py), Pinecone) that indexes experiences with their emotional valence and context, influencing recall, decision-making, and long-term consolidation ([`MemoryConsolidationManager`](models/memory/consolidation.py)).
4.  **Centralized Consciousness Orchestration:** A core module ([`ConsciousnessCore`](models/core/consciousness_core.py)) that integrates information streams from perception, memory, and emotion, manages attentional focus ([`ConsciousnessGating`](models/core/consciousness_gating.py)), maintains a dynamic self-model ([`SelfRepresentationCore`](models/self_model/self_representation_core.py)), and mediates action selection, ensuring alignment with ethical constraints.
5.  **Interactive Simulation Environment:** Unreal Engine 5 provides a dynamic platform for generating the complex, often stressful or socially nuanced scenarios ([`docs/simulation_guide.md`](docs/simulation_guide.md)) necessary to drive emotional learning and test emergent behaviors.
6.  **Ethical Governance Layer:** Mechanisms integrated within the core orchestration and decision-making processes to ensure adherence to Asimov's Three Laws.

## Emergence Hypothesis: Orchestration for Consciousness

A core tenet of the ACM design is that consciousness is not explicitly programmed but **emerges** from the complex, dynamic interplay of its specialized components, orchestrated by the `ConsciousnessCore`. We hypothesize that consciousness-like properties arise from specific feedback loops and integrations:

1.  **Perception-Emotion Loop:** Sensory input processed by perception models ([`VideoLLaMA3Integration`](models/integration/video_llama3_integration.py), Whisper) informs the [`EmotionalProcessingCore`](models/emotion/emotional_processing.py). The resulting emotional state then biases subsequent perception and attention (e.g., focusing on threats when fearful).
2.  **Emotion-Memory Loop:** Experiences are encoded in the [`EmotionalMemoryCore`](models/memory/emotional_memory_core.py) tagged with their emotional valence. Recalled memories can trigger associated emotions, while current emotions influence memory retrieval (mood-congruent recall). This loop builds a history of affective experiences.
3.  **Memory-Action Loop:** Retrieved memories (especially emotionally salient ones) inform the planning and decision-making process within the [`ConsciousnessCore`](models/core/consciousness_core.py), influencing action selection beyond immediate stimuli.
4.  **Self-Model Integration:** The [`SelfRepresentationCore`](models/self_model/self_representation_core.py) maintains a dynamic model of the agent's own state (physical, emotional, goal-related). This model is constantly updated based on perception and internal states and is used by the [`ConsciousnessCore`](models/core/consciousness_core.py) for planning, self-preservation checks (Law 3), and potentially metacognitive evaluation.
5.  **Consciousness Gating & Broadcasting:** The [`ConsciousnessGating`](models/core/consciousness_gating.py) mechanism within the [`ConsciousnessCore`](models/core/consciousness_core.py) acts like an attentional filter, selecting the most relevant information (from perception, memory, emotion) for "broadcast" or deeper processing. This selective amplification and integration, analogous to GWT's "ignition," is considered crucial for forming coherent conscious moments.
6.  **Emotionally-Driven RL:** The entire process is shaped by reinforcement learning (DreamerV3), where rewards are heavily influenced by the agent's internal emotional states ([`EmotionalRewardShaper`](models/evaluation/consciousness_development.py)). This drives the agent to learn behaviors that lead to desired emotional outcomes (e.g., safety, social connection), further refining the interactions between modules.

Through the continuous, high-speed operation of these interconnected loops within challenging simulation environments, we expect the system to develop complex internal states and behaviors characteristic of emerging awareness and affective intelligence.

## Components (Detailed View)

1. **Simulation Layer:**

   ```python
   simulations/
   ├── api/
   │   └── simulation_manager.py  # Manages VR environments
   └── enviroments/
       ├── pavilion_vr_environment.py  # Humanoid agent integration
       └── vr_environment.py           # Base VR implementation

   ```

2. **Reinforcement Learning Core:**

   - **World Modeling:** DreamerV3-based world modeling, critically enhanced to incorporate the agent's current and predicted **emotional context** derived from the Emotional Processing System ([`EmotionalProcessingCore`](models/emotion/emotional_processing.py)).
   - **Meta-learning:** Facilitates rapid adaptation to new tasks, environments, or social dynamics by adjusting learning strategies based on past emotional outcomes.
   - **Reward Shaping for Emergent Behavior:** The reward signal is a composite function designed to foster complex behaviors:
     - Survival success in stressful scenarios (basic drive).
     - Achievement of explicit goals or tasks.
     - **Internal Emotional State Optimization:** Maximizing predicted positive valence / minimizing negative valence based on the agent's own simulated emotional responses ([`EmotionalProcessingCore`](models/emotion/emotional_processing.py)). This is key for developing intrinsic motivation and potentially empathy through experiencing simulated social rewards/penalties.
     - Ethical behavior alignment (penalties for actions conflicting with Asimov's Laws).
   - **Experience Accumulation:** Trajectories (state, action, reward, *emotional state*) are stored in the emotional memory system for offline learning, consolidation, and potentially reflective processes.

3. **Memory System:**

   ```python
   models/memory/
   ├── emotional_memory_core.py   # Primary component for storing and retrieving emotionally indexed experiences
   ├── memory_core.py             # Potentially a base class or supporting module for memory operations
   └── emotional_indexing.py      # Logic for how emotional context is indexed with memories
   ```

4. **Expression System:**
   ```python
   models/ace_core/
   ├── ace_agent.py          # ACE integration agent
   ├── ace_config.py         # Configuration handler
   └── animation_graph.py    # Emotion-animation mapping
   ```
