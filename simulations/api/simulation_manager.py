import pandas as pd
from threading import Lock
import subprocess
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np
import torch
import logging

from models.self_model.reinforcement_core import ReinforcementCore
from models.emotion.tgnn.emotional_graph import EmotionalGraphNetwork
from models.narrative.narrative_engine import NarrativeEngine
from models.memory.memory_core import MemoryCore
from models.predictive.dreamerv3_wrapper import DreamerV3
from simulations.enviroments.pavilion_vr_environment import PavilionVREnvironment
from simulations.enviroments.vr_environment import VREnvironment
from models.cognitive.chain_of_thought import ChainOfThought


@dataclass
class SimulationConfig:
    """Configuration for the simulation environment."""
    max_steps: int = 1000
    emotional_scale: float = 2.0
    emotion_threshold: float = 0.6
    memory_capacity: int = 100000
    narrative_max_length: int = 128
    use_pavilion: bool = True
    pavilion_config: Optional[Dict] = None


class SimulationManager:
    """
    Main simulation manager for consciousness development
    through emotional learning.
    """

    def __init__(self, config: SimulationConfig):
        self.lock = Lock()
        self.config = config
        logging.info("Simulation Manager initialized with config: %s", config)

        # Core modules.
        self.rl_core = ReinforcementCore(config)
        self.emotion_network = EmotionalGraphNetwork()
        self.narrative = NarrativeEngine()
        self.memory = MemoryCore(capacity=config.memory_capacity)
        self.chain_processor = ChainOfThought(self.memory)

        # Initialize environment (Pavilion or fallback VR).
        self.env = self._initialize_environment()

        # Tracking metrics.
        self.episode_rewards: List[float] = []
        self.emotion_history: List[Dict[str, float]] = []
        self.current_scenario = None

    def _initialize_environment(self):
        """Initialize the VR environment with optional Pavilion integration."""
        if self.config.use_pavilion:
            return PavilionVREnvironment(
                config=self.config.pavilion_config,
                emotion_network=self.emotion_network
            )
        return VREnvironment()

    def execute_code(self, code: str) -> dict:
        """
        Safely executes dynamically generated Python code.

        Args:
            code (str): Python code to execute.

        Returns:
            dict: Updated globals after execution.

        Raises:
            Exception: If code execution fails.
        """
        try:
            exec_globals = {}
            exec(code, exec_globals)
            logging.info("Code executed successfully.")
            return exec_globals
        except Exception as e:
            logging.error("Code execution error: %s", e)
            raise

    def load_interaction_data(self):
        """
        Load simulation datasets (e.g., INTERACTION, UE-HRI) for environment tasks.
        """
        try:
            interaction_data = pd.read_csv("/data/simulations/interaction_data.csv")
            print("INTERACTION data loaded successfully.")

            ue_hri_data = pd.read_csv("/data/simulations/ue_hri_data.csv")
            print("UE-HRI data loaded successfully.")

            # Extend or store loaded data as needed for simulation tasks.
        except Exception as e:
            print(f"Error loading datasets: {e}")

    def run_interaction_episode(self, agent, environment) -> Dict[str, Any]:
        episode_data = []
        state = environment.reset()
        
        done = False
        step = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = environment.step(action)
            
            # Compute emotional reward
            emotional_reward = self.rl_core.compute_reward(
                state=state,
                emotion_values=info.get('emotional_context'),
                narrative=agent.current_narrative()
            )
            
            # Store experience in memory
            self.memory.store_experience({
                "state": state,
                "action": action,
                "reward": emotional_reward,
                "next_state": next_state,
                "emotion": info.get('emotional_context'),
                "narrative": agent.current_narrative(),
                "done": done
            })
            
            # Update RL core with emotional context
            self.rl_core.update(
                state=state,
                action=action,
                reward=emotional_reward,
                next_state=next_state,
                done=done,
                emotion_context=info.get('emotional_context')
            )
            
            episode_data.append({
                "step": step,
                "emotion": info.get('emotional_context'),
                "reward": emotional_reward
            })
            state = next_state
            step += 1
        
        # After the episode, generate the chain-of-thought narrative and multimodal output.
        thought_data = self.chain_processor.generate_multimodal_thought()
        # Update the agent's narrative state with the introspection output.
        agent.update_narrative(thought_data["chain_text"])
        # Optionally, you can also store the visual output reference or use it further.
        episode_data.append({
            "chain_of_thought": thought_data["chain_text"],
            "visual_output": thought_data["visual_output"]
        })

        return {"episode_data": episode_data}

    def _compute_mean_emotion(self, step: int, episode_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute the mean emotion from the last 'step' entries in episode_data.
        Returns an empty dict if no data is available.
        """
        if step <= 0 or not episode_data:
            return {}
        emotion_vals = [data["emotion"] for data in episode_data]
        # Collect keys in the first entry. Assuming consistent keys.
        keys = emotion_vals[0].keys()
        # Average each key across all steps.
        mean_emotion = {
            k: float(np.mean([entry[k] for entry in emotion_vals]))
            for k in keys
        }
        return mean_emotion

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get current learning and performance metrics.
        """
        mean_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0
        recent_emotions = self.emotion_history[-1000:] if self.emotion_history else []
        emotion_stability = 0.0
        if recent_emotions:
            # Example: compute variance of a single dimension (e.g., valence).
            # If your emotion dict has multiple dims, adapt accordingly.
            valences = [em.get("valence", 0.0) for em in recent_emotions]
            emotion_stability = np.std(valences)

        return {
            "mean_reward": mean_reward,
            "emotion_stability": emotion_stability,
            "memory_usage": self.memory.get_usage_stats(),
            "learning_progress": self.rl_core.get_learning_stats()
        }

    def save_checkpoint(self, path: str):
        """
        Save simulation state and model checkpoints.
        """
        checkpoint = {
            "rl_core": self.rl_core.state_dict(),
            "emotion_network": self.emotion_network.state_dict(),
            "episode_rewards": self.episode_rewards,
            "emotion_history": self.emotion_history,
            "config": self.config
        }
        torch.save(checkpoint, path)


# Example usage
if __name__ == "__main__":
    manager = SimulationManager(config=SimulationConfig())
    manager.execute_code("print('Hello, Unreal Engine!')")
    manager.load_interaction_data()
