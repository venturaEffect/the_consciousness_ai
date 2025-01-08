import pandas as pd
from threading import Lock
import subprocess
import unreal
from models.self_model.reinforcement_core import ReinforcementCore
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np
from models.emotion.tgnn.emotional_graph import EmotionalGraphNetwork
from models.narrative.narrative_engine import NarrativeEngine
from models.memory.memory_core import MemoryCore
from models.predictive.dreamerv3_wrapper import DreamerV3
from simulations.enviroments.pavilion_vr_environment import PavilionVREnvironment
from simulations.enviroments.vr_environment import VREnvironment
import torch

@dataclass
class SimulationConfig:
    """Configuration for simulation environment"""
    max_steps: int = 1000
    emotional_scale: float = 2.0
    emotion_threshold: float = 0.6
    memory_capacity: int = 100000
    narrative_max_length: int = 128
    use_pavilion: bool = True
    pavilion_config: Optional[Dict] = None

class SimulationManager:
    """
    Main simulation manager for consciousness development through emotional learning
    """
    
    def __init__(self, config: SimulationConfig):
        self.lock = Lock()
        self.config = config
        
        # Core components
        self.rl_core = ReinforcementCore(config)
        self.emotion_network = EmotionalGraphNetwork()
        self.narrative = NarrativeEngine()
        self.memory = MemoryCore(capacity=config.memory_capacity)
        
        # Initialize Unreal Engine environment
        self.env = self._initialize_environment()
        
        # Tracking metrics
        self.episode_rewards = []
        self.emotion_history = []
        self.current_scenario = None

    def _initialize_environment(self):
        """Initialize VR environment with Pavilion integration"""
        if self.config.use_pavilion:
            return PavilionVREnvironment(
                config=self.config.pavilion_config,
                emotion_network=self.emotion_network
            )
        return VREnvironment()

    def execute_code(self, script: str):
        """
        Executes the provided Python code within the simulation environment.
        """
        try:
            with self.lock:
                # Save the script to a temporary file
                with open("temp_script.py", "w") as temp_file:
                    temp_file.write(script)
                
                # Execute the script
                result = subprocess.run(["python", "temp_script.py"], capture_output=True, text=True)

                # Log the result
                if result.returncode == 0:
                    print(f"Script executed successfully: {result.stdout}")
                else:
                    print(f"Script execution failed: {result.stderr}")

                return result
        except Exception as e:
            print(f"Error during script execution: {str(e)}")

    def load_interaction_data(self):
        """Load INTERACTION and UE-HRI datasets for simulations."""
        try:
            # Load INTERACTION dataset
            interaction_data = pd.read_csv('/data/simulations/interaction_data.csv')
            print("INTERACTION data loaded successfully.")

            # Load UE-HRI dataset
            ue_hri_data = pd.read_csv('/data/simulations/ue_hri_data.csv')
            print("UE-HRI data loaded successfully.")

        except Exception as e:
            print(f"Error loading datasets: {e}")

    def run_interaction_episode(self, agent, environment) -> Dict[str, Any]:
        """
        Run a single interaction episode with emotional reinforcement learning
        """
        state = environment.reset()
        total_reward = 0
        episode_data = []
        
        for step in range(self.config.max_steps):
            # Get action from agent's policy
            action = agent.get_action(state)
            
            # Take step in environment 
            next_state, env_reward, done, info = environment.step(action)
            
            # Process emotional response
            emotion_values = self.emotion_network.process_interaction(
                state=state,
                action=action,
                next_state=next_state,
                info=info
            )
            
            # Generate narrative description
            narrative = self.narrative.generate_experience_narrative(
                state=state,
                action=action,
                emotion=emotion_values,
                include_context=True
            )
            
            # Compute emotional reward with Pavilion's emotional feedback
            emotional_reward = self.rl_core.compute_reward(
                state=state,
                emotion_values=emotion_values,
                narrative=narrative
            )
            
            # Store experience in memory
            self.memory.store_experience({
                'state': state,
                'action': action,
                'reward': emotional_reward,
                'next_state': next_state,
                'emotion': emotion_values,
                'narrative': narrative,
                'done': done
            })
            
            # Update learning systems
            update_info = self.rl_core.update(
                state=state,
                action=action, 
                reward=emotional_reward,
                next_state=next_state,
                done=done,
                emotion_context=emotion_values
            )
            
            # Track episode data
            episode_data.append({
                'step': step,
                'emotion': emotion_values,
                'reward': emotional_reward,
                'narrative': narrative,
                'update_info': update_info
            })
            
            total_reward += emotional_reward
            state = next_state
            
            if done:
                break
                
        # Update tracking metrics
        self.episode_rewards.append(total_reward)
        self.emotion_history.extend(
            [data['emotion'] for data in episode_data]
        )
        
        return {
            'total_reward': total_reward,
            'steps': step + 1,
            'episode_data': episode_data,
            'mean_emotion': np.mean(self.emotion_history[-step:], axis=0),
            'final_narrative': episode_data[-1]['narrative']
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current learning and performance metrics"""
        return {
            'mean_reward': np.mean(self.episode_rewards[-100:]),
            'emotion_stability': np.std(self.emotion_history[-1000:]),
            'memory_usage': self.memory.get_usage_stats(),
            'learning_progress': self.rl_core.get_learning_stats()
        }

    def save_checkpoint(self, path: str):
        """Save simulation state and model checkpoints"""
        checkpoint = {
            'rl_core': self.rl_core.state_dict(),
            'emotion_network': self.emotion_network.state_dict(),
            'episode_rewards': self.episode_rewards,
            'emotion_history': self.emotion_history,
            'config': self.config
        }
        torch.save(checkpoint, path)

# Example usage
if __name__ == "__main__":
    manager = SimulationManager(config=SimulationConfig())
    manager.execute_code("print('Hello, Unreal Engine!')")
    manager.load_interaction_data()
