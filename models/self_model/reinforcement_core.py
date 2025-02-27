import torch
import torch.nn as nn
import numpy as np
from collections import deque
from typing import Dict, Any

from models.predictive.dreamerv3_wrapper import DreamerV3
from models.memory.memory_core import MemoryCore
from models.narrative.narrative_engine import NarrativeEngine
from models.self_model.emotion_context_tracker import EmotionContextTracker
from models.self_model.belief_system import BeliefSystem
from models.self_model.meta_learner import MetaLearner
from models.emotion.reward_shaping import EmotionalRewardShaper


class ReinforcementCore:
    """
    Core RL module that integrates emotional rewards into policy updates.
    """

    def __init__(self, config: Dict[str, Any], emotion_shaper: EmotionalRewardShaper):
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
            emotion_shaper: instance of EmotionalRewardShaper
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
        self.emotion_shaper = emotion_shaper
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

        self.gamma = config.get("gamma", 0.99)
        self.q_network = self._init_q_network(config)
        self.optimizer = self._init_optimizer()

    def _init_q_network(self, config: Dict[str, Any]) -> nn.Module:
        # Stub: Replace with actual Q-network initialization
        return nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 4))

    def _init_optimizer(self):
        return torch.optim.Adam(self.q_network.parameters(), lr=self.config.get("learning_rate", 1e-4))

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

    def compute_reward(self, state: Any, action: int, emotion_values: Dict[str, float], base_reward: float) -> float:
        """
        Computes the reward by modulating the base reward with emotional feedback.
        """
        return self.emotion_shaper.compute_emotional_reward(emotion_values, base_reward)

    def update_policy(self, transition: Dict[str, Any]) -> None:
        """
        Applies a Q-learning update with emotional reward shaping.
        """
        state = transition["state"]
        action = transition["action"]
        reward = transition["reward"]
        next_state = transition["next_state"]

        # Compute Q-values and target
        q_values = self.q_network(torch.tensor(state, dtype=torch.float32))
        next_q_values = self.q_network(torch.tensor(next_state, dtype=torch.float32))
        target = reward + self.gamma * torch.max(next_q_values)

        loss = (q_values[action] - target) ** 2
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

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

    def update(self, state, action, reward, next_state, done, emotion_context=None):
        """
        Update the reinforcement learning core with experience
        """
        # Create embeddings for current state
        state_embedding = self.state_encoder(state)
        
        # Retrieve relevant past experiences based on emotional similarity
        if self.memory and emotion_context:
            relevant_memories = self.memory.retrieve_relevant(
                emotion_context=emotion_context,
                k=self.config.get('memory_context_size', 5)
            )
            
            # Incorporate memory influence into world model updates
            memory_embeddings = [self.state_encoder(memory['state']) for memory in relevant_memories]
            if memory_embeddings:
                memory_context = torch.mean(torch.stack(memory_embeddings), dim=0)
                # Include memory context in world model update
                world_model_info = self.dreamer.update_world_model(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    memory_context=memory_context,
                    emotion_context=emotion_context
                )
            else:
                world_model_info = self.dreamer.update_world_model(
                    state=state, action=action, reward=reward, 
                    next_state=next_state, done=done,
                    emotion_context=emotion_context
                )
        else:
            world_model_info = self.dreamer.update_world_model(
                state=state, action=action, reward=reward,
                next_state=next_state, done=done,
                emotion_context=emotion_context
            )
        
        # Update policy with integrated emotional information
        policy_info = self.dreamer.update_actor_critic(
            state_embedding, 
            emotion_scale=self.config.get('emotional_scale', 1.0),
            emotion_values=emotion_context
        )
        
        return {
            'world_model_loss': world_model_info.get('loss', 0),
            'policy_loss': policy_info.get('actor_loss', 0),
            'value_loss': policy_info.get('critic_loss', 0),
            'adaptation_score': self._calculate_adaptation_score(emotion_context)
        }
