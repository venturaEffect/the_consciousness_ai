import numpy as np
import torch
import logging
import time
import os
from PIL import Image
from tqdm import tqdm

from simulations.environments.simple_visual_env import SimpleVisualEnv
from models.agent.consciousness_agent import ConsciousnessAgent

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_emotional_agent():
    # 1. Configuration
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "vision": {
            "model_name": "Qwen/Qwen2-VL-7B-Instruct",
            "quantization": {"load_in_4bit": True},
            "generation": {"max_new_tokens": 50}
        },
        "workspace": {
            "broadcast_threshold": 0.6,
            "ignition_gain": 5.0,
            "reverberation_alpha": 0.8
        },
        "emotion": {
            "valence_weight": 0.5,
            "arousal_penalty": 1.0
        },
        "reinforcement": {
            "state_dim": 64, # Small for prototype
            "action_dim": 2, # [Move X, Move Y]
            "learning_rate": 1e-3
        },
        "memory": {}
    }
    
    logger.info("Starting 'The Dark Room' Experiment...")
    
    # 2. Initialize Environment (The Body)
    env = SimpleVisualEnv(render_mode="rgb_array", width=224, height=224) # Smaller size for speed
    
    # 3. Initialize Agent (The Brain)
    try:
        agent = ConsciousnessAgent(config)
    except Exception as e:
        logger.error(f"Failed to initialize Agent: {e}")
        return

    # 4. Training Loop
    episodes = 10
    max_steps = 100
    
    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        logger.info(f"--- Episode {episode+1} Started ---")
        
        with tqdm(total=max_steps, desc=f"Ep {episode+1}") as pbar:
            while not done and step < max_steps:
                # Preprocess Observation (Numpy -> PIL)
                # Qwen expects PIL Image for 'image' field usually
                pil_image = Image.fromarray(obs)
                
                # Agent Step (Perceive -> Feel -> Think -> Act)
                action, agent_info = agent.step(pil_image)
                
                # Environment Step
                next_obs, reward, terminated, truncated, env_info = env.step(action)
                done = terminated or truncated
                
                # Learning Step
                # We calculate a 'Functionalist Reward'
                # R = R_ext (Environment) + Delta Valence - Anxiety
                
                # Anxiety is high if 'arousal' is high.
                current_arousal = agent_info["emotion"]["arousal"]
                functional_reward = reward - (current_arousal * 0.5) 
                
                # Feed back to Agent
                loss_metrics = agent.update(
                    state=None, # handled internally via text encoding
                    action=action,
                    reward=functional_reward,
                    next_state=None, 
                    done=done,
                    info=agent_info
                )
                
                # Logging
                phi = agent_info.get("phi", 0.0)
                is_conscious = agent_info.get("is_conscious", False)
                desc = agent_info.get("description", "...")
                
                if step % 10 == 0:
                    tqdm.write(f"Step {step}: P={phi:.2f} | C={is_conscious} | E={current_arousal:.2f} | '{desc}'")
                
                obs = next_obs
                total_reward += functional_reward
                step += 1
                pbar.update(1)
                
                if done:
                    tqdm.write(">>> GOAL REACHED (Battery Depleted or Success) <<<")
        
        logger.info(f"Episode {episode+1} Finished. Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    train_emotional_agent()
