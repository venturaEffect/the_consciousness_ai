import pandas as pd
from threading import Lock
import subprocess
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np
import torch
import logging
import asyncio

from models.self_model.reinforcement_core import ReinforcementCore
from models.emotion.tgnn.emotional_graph import EmotionalGraphNetwork
from models.narrative.narrative_engine import NarrativeEngine
from models.memory.memory_core import MemoryCore
from models.predictive.dreamerv3_wrapper import DreamerV3
from simulations.enviroments.vr_environment import VREnvironment
from models.cognitive.chain_of_thought import ChainOfThought
from models.ace_core.ace_agent import ACEConsciousAgent
from models.ace_core.ace_config import ACEConfig
from models.integration.video_llama3_integration import VideoLLaMA3Integration
from models.memory.emotional_memory_core import EmotionalMemoryCore
from models.core.consciousness_core import ConsciousnessCore
from models.predictive.dreamer_emotional_wrapper import DreamerEmotionalWrapper
from models.memory.attention_schema import AttentionSchema


@dataclass
class SimulationConfig:
    """Configuration for the simulation environment."""
    max_steps: int = 1000
    emotional_scale: float = 2.0
    emotion_threshold: float = 0.6
    memory_capacity: int = 100000
    narrative_max_length: int = 128
    # Removed Pavilion-specific flag and config
    # use_pavilion: bool = True
    # pavilion_config: Optional[Dict] = None


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

        # Always use standard VR environment.
        self.env = VREnvironment()

        # Tracking metrics.
        self.episode_rewards: List[float] = []
        self.emotion_history: List[Dict[str, float]] = []
        self.current_scenario = None

        # ACE components
        self.ace_config = ACEConfig()
        self.ace_agent = ACEConsciousAgent(self.ace_config)
        self.video_llama = VideoLLaMA3Integration()
        self.consciousness_core = ConsciousnessCore()
        self.emotional_memory = EmotionalMemoryCore()
        self.world_model = DreamerEmotionalWrapper()
        self.attention_schema = AttentionSchema()

    def execute_code(self, code: str) -> dict:
        """
        Safely executes dynamically generated Python code.
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
        
        thought_data = self.chain_processor.generate_multimodal_thought()
        agent.update_narrative(thought_data["chain_text"])
        episode_data.append({
            "chain_of_thought": thought_data["chain_text"],
            "visual_output": thought_data["visual_output"]
        })

        return {"episode_data": episode_data}

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get current learning and performance metrics.
        """
        mean_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0
        recent_emotions = self.emotion_history[-1000:] if self.emotion_history else []
        emotion_stability = 0.0
        if recent_emotions:
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

    async def initialize_simulation(self):
        """Initialize all components"""
        await self.ace_agent.initialize()
        await self.video_llama.initialize()
        
    async def simulation_step(self, visual_input, audio_input=None, context=None):
        """Execute one simulation step with ACM-ACE integration"""
        llama_perception = await self.video_llama.process_input(
            visual_input, 
            audio_input
        )

        consciousness_state = await self.consciousness_core.process({
            'perception': llama_perception,
            'context': context
        })

        emotional_response = await self.emotional_memory.generate_response(
            consciousness_state
        )

        ace_result = await self.ace_agent.process_interaction(
            visual_input=visual_input,
            audio_input=audio_input,
            context={
                'consciousness_state': consciousness_state,
                'emotional_response': emotional_response,
                'llama_perception': llama_perception
            }
        )

        # Update attention schema with current focus data
        current_focus = {
            'visual': visual_input,
            'audio': audio_input,
        }
        self.attention_schema.update(current_focus)
        cumulative_focus = self.attention_schema.get_overview()
        self.adjust_self_model(cumulative_focus)

        await self.emotional_memory.update(
            consciousness_state,
            emotional_response,
            ace_result['animation_data']
        )

        await self.world_model.update(
            consciousness_state,
            emotional_response,
            ace_result
        )

        return {
            'consciousness_state': consciousness_state,
            'emotional_response': emotional_response,
            'ace_result': ace_result,
            'llama_perception': llama_perception
        }
    
    def adjust_self_model(self, cumulative_focus):
        """
        Dynamically adjust internal state parameters based on the aggregated focus data.
        This is a placeholder function intended to integrate meta-awareness into the self-model.
        """
        # Example implementation: log the focus data and adjust parameters accordingly.
        print("Adjusting self-model with focus data:", cumulative_focus)

    def load_character_blueprint(self):
        """Load ACE-compatible character blueprint"""
        try:
            blueprint_path = self.ace_config.get_blueprint_path()
            return unreal.load_object(None, blueprint_path)
        except Exception as e:
            print(f"Failed to load character blueprint: {e}")
            return None


# Example usage
if __name__ == "__main__":
    manager = SimulationManager(config=SimulationConfig())
    manager.execute_code("print('Hello, Unreal Engine!')")
    manager.load_interaction_data()
