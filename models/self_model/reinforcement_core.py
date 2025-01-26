import torch
import numpy as np
from collections import deque
from typing import Dict, Any

from models.predictive.dreamerv3_wrapper import DreamerV3
from models.memory.memory_core import MemoryCore
from models.narrative.narrative_engine import NarrativeEngine
from models.self_model.emotion_context_tracker import EmotionContextTracker
from models.self_model.belief_system import BeliefSystem
from models.self_model.meta_learner import MetaLearner


class ReinforcementCore:
    def __init__(self, config: Dict):
        """
        Core reinforcement module integrating DreamerV3, memory,
        emotion context, and meta-learning.

        Args:
            config: Dictionary of parameters. Expected keys:
                - 'dreamerV3': sub-config for DreamerV3
                - 'emotional_scale': float
                - 'positive_emotion_bonus': float
                - 'meta_config': sub-config for meta-learning (optional)
                - 'memory_capacity': optional capacity for MemoryCore
        """
        # Initialize memory; use config capacity if present.
        capacity = config.get('memory_capacity', 100000)
        self.memory = MemoryCore(capacity=capacity)

        # Initialize DreamerV3 with matching key from the config.
        if 'dreamerV3' in config:
            self.dreamer = DreamerV3(config['dreamerV3'])
        else:
            self.dreamer = DreamerV3({})  # Fallback if missing.

        self.narrative = NarrativeEngine()
        self.emotion_tracker = EmotionContextTracker()
        self.belief_system = BeliefSystem()

        # Top-level config references.
        self.config = config
        self.emotional_scale = config.get('emotional_scale', 2.0)
        self.positive_emotion_bonus = config.get('positive_emotion_bonus', 0.5)

        # Meta-learning setup.
        self.meta_learning = False
        self.adaptation_steps = 0
        self.inner_lr = 0.0

        if 'meta_config' in config:
            meta_cfg = config['meta_config']
            self.meta_learning = meta_cfg.get('enabled', False)
            self.adaptation_steps = meta_cfg.get('adaptation_steps', 5)
            self.inner_lr = meta_cfg.get('inner_learning_rate', 0.01)

        # Initialize meta-learner if needed.
        self.meta_learner = MetaLearner(config) if self.meta_learning else None
        self.current_task_params = None

        # Metrics storage. For example:
        self.metrics = {
            'reward_history': deque(maxlen=10000)
        }

    def adapt_to_scenario(self, scenario_data: Dict) -> Dict:
        """
        Adapt to a new scenario using meta-learning, if enabled.
        Args:
            scenario_data: Info about the new scenario/task.
        Returns:
            A dict containing adaptation results.
        """
        if not self.meta_learner:
            return {}

        adaptation_result = self.meta_learner.adapt_to_task(scenario_data)
        self.current_task_params = adaptation_result.get('adapted_params', {})
        return adaptation_result

    def compute_reward(self, state, emotion_values, action_info):
        base_reward = self._get_base_reward(state, action_info)
        shaped_reward = self.emotion_shaper.compute_emotional_reward(emotion_values, base_reward)
        # ...
        return shaped_reward

    def compute_reward(
        self,
        state: Any,
        emotion_values: Dict[str, float],
        action_info: Dict[str, Any]
    ) -> float:
        """
        Compute the reward based on emotional response and state.

        Args:
            state: Current environment state (tensor or array).
            emotion_values: Dict of emotion measurements (e.g., valence, arousal).
            action_info: Information about the taken action.
        """
        # Get emotional valence from the emotion tracker.
        emotional_reward = self.emotion_tracker.get_emotional_value(emotion_values)

        # Scale by any scenario/task-specific parameter from meta-learning.
        if self.current_task_params is not None:
            scale_factor = self.current_task_params.get('emotional_scale', 1.0)
            emotional_reward *= scale_factor

        # Apply top-level emotion scaling.
        scaled_reward = emotional_reward * self.emotional_scale

        # Add bonus for positive emotions.
        if emotional_reward > 0:
            scaled_reward += self.positive_emotion_bonus

        # Build experience for memory.
        experience = {
            'state': state,
            'emotion': emotion_values,
            'action': action_info,
            'reward': scaled_reward,
            'narrative': self.narrative.generate_experience_narrative(
                state=state, emotion=emotion_values, reward=scaled_reward
            ),
            'task_params': self.current_task_params
        }
        self.memory.store_experience(experience)

        # Optionally track rewards for progress or debugging.
        self.metrics['reward_history'].append(scaled_reward)
        return scaled_reward

    def update(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool,
        emotion_context: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Update the model using DreamerV3 with emotional context.
        """
        # Create a batch for the world model.
        world_model_batch = self.dreamer.create_batch(
            state, action, reward, next_state, done,
            additional_context=emotion_context
        )

        # Update the world model with emotional context.
        world_model_loss = self.dreamer.update_world_model(
            world_model_batch,
            emotion_context=emotion_context
        )

        # Update actor-critic with emotional weighting.
        actor_loss, critic_loss = self.dreamer.update_actor_critic(
            world_model_batch,
            emotion_scale=self.emotional_scale
        )

        # Update belief system with new experience.
        belief_update = self.belief_system.update(
            state, action, reward, emotion_context
        )

        # Generate a narrative describing the update.
        # Access the last emotion from emotion_tracker if needed.
        current_emotion = self.emotion_tracker.current_emotion
        narrative = self.narrative.generate_experience_narrative(
            state=state,
            action=action,
            reward=reward,
            emotion=current_emotion,
            belief_update=belief_update
        )

        return {
            'world_model_loss': world_model_loss,
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'narrative': narrative,
            'belief_update': belief_update
        }

    def meta_adapt(self, task: Dict[str, Any]) -> None:
        """
        Perform meta-adaptation using MAML-style update, if enabled.
        """
        if not self.meta_learning or not self.meta_learner:
            return

        # Retrieve experiences relevant to the new task.
        task_experiences = self.memory.get_relevant_experiences(task)

        # Perform quick adaptation steps.
        for _ in range(self.adaptation_steps):
            batch = self.memory.sample_batch(task_experiences)
            self.dreamer.inner_update(batch, self.inner_lr)

    def get_learning_stats(self) -> Dict[str, float]:
        """
        Return some learning statistics, e.g., average reward.
        """
        reward_hist = list(self.metrics['reward_history'])
        avg_reward = float(np.mean(reward_hist)) if reward_hist else 0.0
        return {
            'avg_reward': avg_reward,
            'recent_rewards': reward_hist[-10:]
        }
