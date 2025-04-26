# models/evaluation/consciousness_development.py
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import torch
import time
import logging # Added for logging warnings
import numpy as np  # Added for numpy functions

# Assuming reward_shaping is directly under emotion
from ..emotion.reward_shaping import EmotionalRewardShaper
# Assuming memory_core is directly under memory
from ..memory.memory_core import MemoryCore
# Assuming consciousness_metrics is in the same directory (evaluation)
from .consciousness_metrics import ConsciousnessMetrics
# Assuming dreamer_emotional_wrapper is directly under predictive
from ..predictive.dreamer_emotional_wrapper import DreamerEmotionalWrapper
# Assuming self_representation_core.py is directly in the 'models' directory
# Go up one level (..) to models/, then import the file directly
from ..self_model.self_representation_core import SelfRepresentationCore
# Removed import for non-existent social_learning_pipeline

@dataclass
class DevelopmentMetrics:
    """Tracks consciousness development metrics"""
    emotional_awareness: float = 0.0
    memory_coherence: float = 0.0
    attention_level: float = 0.0
    behavioral_adaptation: float = 0.0
    survival_success: float = 0.0
    last_updated: float = field(default_factory=time.time)

class ConsciousnessDevelopment:
    """
    Manages and evaluates consciousness development through:
    1. Survival-driven attention mechanisms
    2. Emotional reinforcement learning
    3. Memory formation and coherence
    4. Behavioral adaptation
    """

    def __init__(self, config: Dict):
        """
        Initializes the ConsciousnessDevelopment module.

        Args:
            config: A dictionary containing configuration for all sub-modules.
        """
        self.config = config

        # --- Core components ---
        self.dreamer = DreamerEmotionalWrapper(config.get('dreamer_config', {}))
        self.reward_shaper = EmotionalRewardShaper(config.get('reward_config', {}))
        self.memory = MemoryCore(config.get('memory_config', {}))

        # Ensure ConsciousnessMetrics gets access to core/memory if needed by sub-calculators
        # This assumes ConsciousnessCore and MemoryCore instances are created elsewhere
        # and passed in or accessible globally/via context, which might need refactoring.
        # For now, initialize with None, highlighting the dependency.
        core_module_instance = None # Placeholder: Needs actual instance
        memory_instance = self.memory # Use the instance created here
        self.consciousness_metrics = ConsciousnessMetrics(
            config.get('metrics_config', {}),
            core_module=core_module_instance,
            memory_system=memory_instance
        )
        self.self_model = SelfRepresentationCore(config.get('self_model_config', {}))
        # Removed self.social_learning initialization

        # Initialize metrics tracking
        self.metrics = DevelopmentMetrics()


    def process_experience(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        emotion_values: Dict[str, float],
        attention_level: float,
        done: bool,
        timestamp: float = None,
        # Removed social_context argument as the related module doesn't exist
    ) -> Dict:
        """
        Processes a single step of experience, updates models, shapes reward, and calculates metrics.
        """
        if timestamp is None:
            timestamp = time.time()

        # 1. Shape the reward using emotional context
        shaped_reward = self.reward_shaper.shape_reward(
            raw_reward=reward,
            emotion_values=emotion_values,
            state=state,
            action=action,
            next_state=next_state
        )

        # 2. Store experience for RL/memory
        experience = {
            "state": state, "action": action, "raw_reward": reward,
            "shaped_reward": shaped_reward, "next_state": next_state,
            "emotion_values": emotion_values, "attention_level": attention_level,
            "done": done, "timestamp": timestamp
            # Removed social_context from experience dict
        }
        self.store_experience(**experience)

        # 3. Update RL model (Dreamer) - Requires Dreamer instance and its API
        learning_info = {} # Default empty info
        if hasattr(self.dreamer, 'update') and callable(self.dreamer.update):
             try:
                  # Assuming dreamer.update takes the experience dict and returns learning info
                  learning_info = self.dreamer.update(experience)
                  if learning_info is None: learning_info = {} # Ensure it's a dict
             except Exception as e:
                  logging.error(f"Error calling dreamer.update: {e}", exc_info=True)
                  learning_info = {"error": str(e)}
        else:
             logging.warning("Dreamer update method not found or not callable.")
             learning_info = {"loss": torch.rand(1).item()} # Placeholder if no update method

        # 4. Update internal development metrics
        self.update_metrics(emotion_values, attention_level, learning_info)

        # 5. Return relevant info
        return {"learning_info": learning_info, "current_metrics": self.get_metrics()}


    def store_experience(self, **kwargs):
        """Stores experience in memory and potentially RL buffer."""
        self.memory.store(timestamp=kwargs['timestamp'], data=kwargs)


    def update_metrics(
        self,
        emotion_values: Dict[str, float],
        attention_level: float,
        learning_info: Dict
    ):
        """Updates the internal DevelopmentMetrics based on recent experience."""
        self.metrics.emotional_awareness = self._calculate_emotional_awareness(emotion_values)
        # Ensure MemoryCore has check_coherence or similar method
        if hasattr(self.memory, 'check_coherence') and callable(self.memory.check_coherence):
             try:
                  self.metrics.memory_coherence = self.memory.check_coherence()
             except Exception as e:
                  logging.error(f"Error calling memory.check_coherence: {e}", exc_info=True)
                  self.metrics.memory_coherence = 0.0 # Default on error
        else:
             logging.warning("MemoryCore.check_coherence method not found.")
             self.metrics.memory_coherence = 0.0

        self.metrics.attention_level = attention_level # Assumes attention is passed in
        self.metrics.behavioral_adaptation = self._calculate_behavioral_adaptation(learning_info)
        # Survival success likely updated less frequently, e.g., in evaluate_development
        self.metrics.last_updated = time.time()


    def _calculate_emotional_awareness(self, emotion_values: Dict[str, float]) -> float:
        """Placeholder: Calculate score based on emotion complexity/differentiation."""
        if not emotion_values: return 0.0
        try:
             # Example: Use variance or entropy of emotion values if available
             variance = np.var(list(emotion_values.values()))
             # Normalize or scale variance appropriately
             awareness_score = np.clip(variance * 10, 0, 1) # Example scaling/clipping
             return awareness_score
        except Exception as e:
             logging.error(f"Error calculating emotional awareness: {e}", exc_info=True)
             return 0.0

    def _calculate_behavioral_adaptation(self, learning_info: Dict) -> float:
        """Placeholder: Calculate adaptation based on RL learning progress (e.g., loss reduction)."""
        try:
             loss = learning_info.get("loss")
             if loss is not None:
                  # Simple inverse relationship, maybe track change over time later
                  adaptation_score = np.clip(1.0 / (1.0 + float(loss)), 0, 1) # Normalize loss effect
                  return adaptation_score
             else:
                  logging.warning("No 'loss' found in learning_info for behavioral adaptation calculation.")
                  return 0.0 # Default if no loss info
        except Exception as e:
             logging.error(f"Error calculating behavioral adaptation: {e}", exc_info=True)
             return 0.0

    def calculate_survival_success(self) -> float:
         """Calculates survival success rate based on historical data (e.g., from memory)."""
         if hasattr(self.memory, 'query_survival_rate') and callable(self.memory.query_survival_rate):
              try:
                   # Example: Query last 100 episodes, find % that didn't end in 'failure' state
                   survival_rate = self.memory.query_survival_rate(last_n=100) # Needs MemoryCore method
                   return survival_rate
              except Exception as e:
                   logging.error(f"Error calling memory.query_survival_rate: {e}", exc_info=True)
                   return self.metrics.survival_success # Return previous value on error
         else:
              logging.warning("MemoryCore.query_survival_rate method not found.")
              return self.metrics.survival_success # Return previous value

    def get_metrics(self) -> Dict:
         """Returns the current development metrics as a dictionary."""
         # Optionally update less frequent metrics here
         self.metrics.survival_success = self.calculate_survival_success()
         return self.metrics.__dict__


    def evaluate_development(
        self,
        current_state: Dict,
        activity_log: Optional[List] = None,
        attention_metrics: Optional[Dict[str, float]] = None
        # Removed social_interactions argument
    ) -> Dict:
        """
        Performs a more comprehensive evaluation of development stage.
        """
        # Use ConsciousnessMetrics for theoretical scores
        # Ensure current_state and activity_log are valid inputs
        if not isinstance(current_state, dict) or not current_state:
             logging.error("Invalid current_state provided to evaluate_development.")
             # Return default/empty report or raise error
             return {"error": "Invalid current_state"}

        phi_approx = self.consciousness_metrics.phi_calculator.calculate_phi_approximation(current_state, activity_log)
        ignition_detected, ignition_details = self.consciousness_metrics.gwt_tracker.detect_ignition(current_state, activity_log)
        pci_approx = self.consciousness_metrics.pci_tester.calculate_pci_approximation(current_state)
        self_awareness_scores = self.consciousness_metrics.self_awareness_monitor.evaluate_self_awareness(current_state)

        current_dev_metrics = self.get_metrics() # Get updated basic metrics

        # Combine all metrics into a development report
        report = {
            "timestamp": current_state.get("timestamp", time.time()),
            "development_metrics": current_dev_metrics,
            "consciousness_metrics": {
                "phi_approx": phi_approx,
                "gwt_ignition": ignition_detected,
                "gwt_details": ignition_details,
                "pci_approx": pci_approx,
                "self_awareness": self_awareness_scores
            },
            "attention_focus": attention_metrics if attention_metrics else {}
        }
        return report