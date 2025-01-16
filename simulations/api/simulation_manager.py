import pandas as pd
from threading import Lock
import subprocess
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np
import torch

from models.self_model.reinforcement_core import ReinforcementCore
from models.emotion.tgnn.emotional_graph import EmotionalGraphNetwork
from models.narrative.narrative_engine import NarrativeEngine
from models.memory.memory_core import MemoryCore
from models.predictive.dreamerv3_wrapper import DreamerV3
from simulations.enviroments.pavilion_vr_environment import PavilionVREnvironment
from simulations.enviroments.vr_environment import VREnvironment


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

        # Core modules.
        self.rl_core = ReinforcementCore(config)
        self.emotion_network = EmotionalGraphNetwork()
        self.narrative = NarrativeEngine()
        self.memory = MemoryCore(capacity=config.memory_capacity)

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

    def execute_code(self, script: str):
        """
        Executes the provided Python code within the simulation environment.
        Useful for debugging or dynamic scripting.
        """
        try:
            with self.lock:
                # Save script to a temporary file.
                with open("temp_script.py", "w") as temp_file:
                    temp_file.write(script)

                # Execute the script.
                result = subprocess.run(
                    ["python", "temp_script.py"], capture_output=True, text=True
                )

                # Log the result.
                if result.returncode == 0:
                    print(f"Script executed successfully: {result.stdout}")
                else:
                    print(f"Script execution failed: {result.stderr}")

                return result
        except Exception as e:
            print(f"Error during script execution: {str(e)}")

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
        """
        Run a single interaction episode with emotional reinforcement learning.

        Args:
            agent: The RL agent or policy to act in the environment.
            environment: The VR environment instance.

        Returns:
            A dictionary containing final stats of the episode.
        """
        state = environment.reset()
        total_reward = 0.0
        episode_data = []

        for step in range(self.config.max_steps):
            # Get action from agent's policy.
            action = agent.get_action(state)

            # Environment step.
            next_state, env_reward, done, info = environment.step(action)

            # Process emotional response.
            emotion_values = self.emotion_network.process_interaction(
                state=state,
                action=action,
                next_state=next_state,
                info=info
            )

            # Generate narrative description.
            narrative = self.narrative.generate_experience_narrative(
                state=state,
                action=action,
                emotion=emotion_values,
                include_context=True
            )

            # Compute emotional reward.
            emotional_reward = self.rl_core.compute_reward(
                state=state,
                emotion_values=emotion_values,
                narrative=narrative
            )

            # Store experience in memory.
            self.memory.store_experience({
                "state": state,
                "action": action,
                "reward": emotional_reward,
                "next_state": next_state,
                "emotion": emotion_values,
                "narrative": narrative,
                "done": done
            })

            # Update learning systems.
            update_info = self.rl_core.update(
                state=state,
                action=action,
                reward=emotional_reward,
                next_state=next_state,
                done=done,
                emotion_context=emotion_values
            )

            # Track episode data.
            episode_data.append({
                "step": step,
                "emotion": emotion_values,
                "reward": emotional_reward,
                "narrative": narrative,
                "update_info": update_info
            })

            total_reward += emotional_reward
            state = next_state

            if done:
                break

        # Update tracked metrics.
        self.episode_rewards.append(total_reward)
        # Avoid slicing with 0 if the episode ended immediately.
        steps_in_episode = max(step, 0)  
        if steps_in_episode > 0:
            recent_emotions = [data["emotion"] for data in episode_data]
            self.emotion_history.extend(recent_emotions)

        return {
            "total_reward": total_reward,
            "steps": step + 1,
            "episode_data": episode_data,
            "mean_emotion": self._compute_mean_emotion(step, episode_data),
            "final_narrative": episode_data[-1]["narrative"] if episode_data else ""
        }

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
