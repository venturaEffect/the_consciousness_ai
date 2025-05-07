# Simulation Guide for Building ACM Training Scenarios on Unreal Engine 5

This guide outlines the approach for setting up simulation scenarios in Unreal Engine that are specifically designed to train the Artificial Consciousness Module (ACM) using emotional reinforcement learning. The ACM project is based on principles observed in organic consciousness development in mammals. A stratified accumulation of ancestral experiences, captured as emotional and instinctive responses.

---

## 1. Overview

### Organic Consciousness Principles

- **Stratified Experiences:** Consciousness in mammals evolves from layered, inherited experiences. Our ACM emulates this by accumulating emotional rewards from interactions in simulations. Each interaction represents a building block of ancestral memory.
- **Instinct & Emotional Response:** The agent’s behavior evolves through basic instinctive responses at early simulation stages, eventually developing into complex social behaviors (e.g., love, sadness, self-awareness).

### Goal

- **Incremental Complexity:** Begin with simple survival or interaction scenarios and gradually increase the complexity to include social interactions, communication, and self-reflection.
- **Meta-Memory Storage:** Emotional rewards are treated as meta-memory that not only guide the reinforcement learning process but also serve as feedback to fine-tune foundational models (and act as LoRAs for image/video generation) during simulated reflective thinking.

---

## 2. Setting Up the Unreal Engine Environment

### Environment Configuration

- **Install Unreal Engine 5:** Follow the official installation guide for Unreal Engine 5.
- **Project Setup:** Create a new VR or simulation project. Organize your project structure with folders for maps, assets, and blueprints.
- **AI Tools Integration:**
  - **Behavior Trees and AI Controllers:** Utilize Unreal's Behavior Trees for decision making and AI Controllers to manage agents.
  - **Dynamic Event Systems:** Set up event triggers (using Blueprints or C++ code) that capture simulation events (e.g., danger, social encounters).
  - **Integration Plugins:** Look for plugins that support face recognition, emotion detection, or real-time analytics, such as:
    - **FaceFX** – for facial animation and emotion mapping.
    - **Live Link Face** – to stream facial capture data into Unreal.
    - **AI-Assisted Tools:** Consider third-party AI copilot tools that integrate within Unreal Editor to assist in simulation development.
  - **Unreal Engine Marketplace:** Actively search the Unreal Engine Marketplace for plugins that facilitate communication with external applications (e.g., via TCP/IP, UDP, gRPC, REST APIs) or offer direct Python scripting enhancements for runtime interactions. Many community-developed and official plugins can simplify the integration of external AI modules like the ACM.
  - **Custom C++ or Blueprint Solutions:** For bespoke communication needs, develop custom C++ modules or Blueprint scripts within Unreal Engine to handle data exchange with your Python-based ACM. This could involve setting up socket listeners, gRPC services, or HTTP endpoints.

---

## 3. Simulation Scenarios Development

### Stage 1: Simple Interactions

- **Basic Survival:** Create scenarios where an agent navigates an environment with basic challenges (obstacles, limited resources).
- **Instinctive Responses:** Trigger simple emotional cues (fear, hunger) which will produce initial emotional rewards.
- **Code Example:**

  ```python
  # python: simulations/enviroments/simple_survival.py
  def step(self, action: Dict) -> tuple:
      # Basic environment step
      next_state, reward, done, info = super().step(action)
      # Capture basic instinctual reaction
      # This info is then processed by EmotionalProcessingCore
      info['raw_emotional_cues'] = {'state_features': next_state} 
      # info['emotional_context'] = simple_emotion_calculation(next_state) # old
      return next_state, reward, done, info
  ```

### Stage 2: Advanced Social Interactions

- **Complex Scenarios:** Develop environments where multiple agents interact. Introduce social cues such as cooperation, competition, expressions of empathy (love, sadness).
- **Emotional Feedback Loop:** Integrate face recognition/emotion detection using Unreal plugins:
  - Capture facial expressions.
  - Map expressions to emotional states.
- **Code Example with Unreal Integration:**

  ```python
  # python: simulations/enviroments/advanced_social.py
  def step(self, action: Dict) -> tuple:
      next_state, reward, done, info = super().step(action)
      # Obtain real-time facial emotion from Unreal's face recognition plugin
      # This info is then processed by EmotionalProcessingCore
      info['raw_emotional_cues'] = {'state_features': next_state}
      if self.face_recognition:
          facial_emotion = self.face_recognition.detect_emotion()
          info['raw_emotional_cues']['facial_emotion'] = facial_emotion
      # Update contextual emotional state
      # info['emotional_context'] = compute_emotion(next_state, facial_emotion) # old
      return next_state, reward, done, info
  ```

---

## 4. Reinforcement Learning Loop with Emotional Rewards

### Simulation Manager Responsibilities

- **Interaction Episodes:** Manage episodes where the agent interacts with the environment.
- **Reward Shaping:** Utilize the `EmotionalRewardShaper` to combine environmental rewards with emotional signals to calculate a composite emotional reward.
- **Memory and Meta-Memory:** Store each interaction’s emotional reward as meta-memory guiding both present behavior and future fine-tuning of models.

### Sample Interaction Loop:

```python
# python: simulations/api/simulation_manager.py
# Assuming self.emotional_processing_core and self.emotional_reward_shaper are initialized
# and self.rl_agent represents the core RL algorithm (e.g., DreamerV3)

def run_interaction_episode(self, agent, environment) -> Dict[str, Any]:
    episode_data = []
    state = environment.reset()
    # Initial emotional context from EmotionalProcessingCore based on initial state
    current_emotional_context = self.emotional_processing_core.get_emotional_state(state, agent.current_internal_state())

    done = False
    step_count = 0 # Renamed from 'step' to avoid conflict with environment.step
    while not done:
        action = agent.select_action(state, current_emotional_context) # Agent might use emotion for action selection
        next_state, base_reward, done, info = environment.step(action)

        # Update emotional context based on new state and info from environment
        # The EmotionalProcessingCore would interpret info['raw_emotional_cues']
        next_emotional_context = self.emotional_processing_core.update_emotional_state(
            next_state, 
            info.get('raw_emotional_cues'), 
            agent.current_internal_state()
        )

        # Compute composite emotional reward using EmotionalRewardShaper
        composite_emotional_reward = self.emotional_reward_shaper.compute_reward(
            base_reward=base_reward,
            current_emotional_state=current_emotional_context,
            next_emotional_state=next_emotional_context, # Rewarding transitions to better emotional states
            narrative_coherence=agent.check_narrative_coherence(state, action, next_state), # Example
            ethical_compliance=agent.check_ethical_compliance(action) # Example
        )

        # Store experience in memory for meta-learning
        self.memory.store_experience({
            "state": state,
            "action": action,
            "reward": composite_emotional_reward, # Use the composite reward
            "next_state": next_state,
            "emotion_context_before_action": current_emotional_context,
            "emotion_context_after_action": next_emotional_context,
            "narrative_elements": agent.get_narrative_elements(), # Example
            "done": done
        })

        # Update RL agent based on emotional feedback
        self.rl_agent.update(
            state=state,
            action=action,
            reward=composite_emotional_reward, # Use the composite reward
            next_state=next_state,
            done=done,
            emotional_context=current_emotional_context # Provide emotion as part of state if agent uses it
        )

        episode_data.append({
            "step": step_count,
            "emotion": current_emotional_context, # Log emotion at the time of action
            "reward": composite_emotional_reward
        })
        state = next_state
        current_emotional_context = next_emotional_context
        step_count += 1

    return {"episode_data": episode_data}
```

## 5. Using Emotional Rewards for Model Fine-Tuning and Self-Awareness

### Integration of Foundational Models and LoRAs

- **Fine-Tuning:** Use stored emotional rewards and interaction meta-data to fine-tune the foundational language model.
- **LoRA for Image & Video:** Leverage emotional states as cues for generating imagery or video scenarios that reflect the agent’s internal state.
- **Self-Reflection Module:** Implement a module that allows the agent to "self-reflect" by processing its own recorded interactions and emotional responses. This self-reflection is key to developing self-awareness and an internal model of identity.

**Example Integration:**

```python
# python: models/narrative/narrative_engine.py
def generate_self_reflection(self, interaction_log: List[Dict]) -> str:
    """
    Generate a reflective narrative based on past emotional rewards and interactions.
    """
    # Use foundational model fine-tuned with emotional meta-memory to generate narrative
    refined_log = prepare_data(interaction_log)
    narrative = self.foundational_model.generate(
        prompt="Reflect on the following experiences: " + refined_log,
        parameters={"temperature": 0.8, "max_length": 512}
    )
    return narrative
```

## 6. Recommended AI Tools for Unreal Engine Development

To assist developers in building these robust simulations, consider the following AI tools that integrate well with Unreal Engine 5:

- **Unreal Engine’s AI Perception System:** For real-time event and emotion recognition.
- **Live Link Face:** For streaming live capture of facial expressions.
- **Behavior Trees & AI Controllers:** Native to Unreal for developing complex agent behaviors.
- **External AI Copilot Tools:** Tools that assist in real-time debugging and simulation adjustments, such as those integrated with VS Code.

## 7. Conclusion

This guide provides an initial framework for developing simulation scenarios in Unreal Engine that align with the goals of the ACM project. Creating a robust system that evolves from basic survival instincts to complex, self-aware social interactions, driven by emotional reinforcement learning.
