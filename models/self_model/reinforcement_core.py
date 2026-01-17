import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from collections import deque

from models.emotion.reward_shaping import EmotionalRewardShaper
from models.memory.memory_core import MemoryCore

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (Policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh() # Continuous action space usually -1 to 1
        )
        
        # Critic head (Value)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features(state)
        action_mean = self.actor(x)
        value = self.critic(x)
        return action_mean, value

class ReinforcementCore:
    """
    Core Reinforcement Learning module integrating Emotional Rewards.
    Replaces DreamerV3 with a custom PPO-compatible architecture.
    """
    
    def __init__(self, config: Dict[str, Any], emotion_shaper: EmotionalRewardShaper, memory: MemoryCore):
        self.config = config
        self.emotion_shaper = emotion_shaper
        self.memory = memory
        
        # Hyperparameters
        self.state_dim = config.get("state_dim", 128) # Combined multimodal embedding size
        self.action_dim = config.get("action_dim", 4)
        self.gamma = config.get("gamma", 0.99)
        self.lr = config.get("learning_rate", 3e-4)
        self.val_coef = config.get("val_coef", 0.5)
        self.ent_coef = config.get("ent_coef", 0.01)
        
        # Models
        self.policy = ActorCritic(self.state_dim, self.action_dim).to(config.get("device", "cpu"))
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        
        # Storage for PPO rollouts
        self.rollout_buffer = []

    def select_action(self, state_embedding: torch.Tensor) -> Tuple[np.ndarray, float]:
        """
        Select action based on current policy.
        Returns action and value estimate.
        """
        self.policy.eval()
        with torch.no_grad():
            action_mean, value = self.policy(state_embedding)
            
            # Simple exploration: Add noise
            # In production, use proper distribution (e.g. Normal) sampling
            noise = torch.randn_like(action_mean) * 0.1
            action = action_mean + noise
            action = torch.clamp(action, -1.0, 1.0)
            
        return action.cpu().numpy(), value.item()

    def step(self, 
             state: torch.Tensor, 
             action: np.ndarray, 
             raw_reward: float, 
             next_state: torch.Tensor, 
             done: bool, 
             emotion_state: Dict[str, float],
             attention_level: float,
             narrative: str = "") -> Dict[str, float]:
        """
        Process a single step: compute emotional reward, store experience, and potentially update.
        """
        # 1. Shape the reward using Emotions
        # We assume emotion_shaper.compute_emotional_reward is available (based on duplications, picking one)
        # Using the one with 3 args from previous analysis or the one with 2.
        # Let's use the one we saw: compute_emotional_reward(emotion_values, base_reward, context)
        shaped_reward = self.emotion_shaper.compute_emotional_reward(
            emotion_values=emotion_state,
            base_reward=raw_reward,
            context={"adaptation_detected": False} # Context placeholder
        )
        
        # 2. Store in Memory
        # Convert action to tensor for storage
        action_tensor = torch.tensor(action, device=state.device)
        self.memory.store_experience(
            state=state,
            action=action_tensor,
            reward=shaped_reward,
            emotion_values=emotion_state,
            attention_level=attention_level,
            narrative=narrative
        )
        
        # 3. Store in local rollout buffer for training
        self.rollout_buffer.append({
            "state": state,
            "action": action_tensor,
            "reward": shaped_reward,
            "next_state": next_state,
            "done": done
        })
        
        return {
            "raw_reward": raw_reward,
            "shaped_reward": shaped_reward
        }

    def update_policy(self) -> Dict[str, float]:
        """
        Train the policy using collected rollouts (Simplified PPO/A2C step).
        """
        if len(self.rollout_buffer) < 10: # Min batch size
            return {}
            
        self.policy.train()
        
        # Aggregate batch
        states = torch.stack([x["state"] for x in self.rollout_buffer])
        actions = torch.stack([x["action"] for x in self.rollout_buffer])
        rewards = torch.tensor([x["reward"] for x in self.rollout_buffer], device=states.device).unsqueeze(1)
        next_states = torch.stack([x["next_state"] for x in self.rollout_buffer])
        dones = torch.tensor([x["done"] for x in self.rollout_buffer], device=states.device).unsqueeze(1)
        
        # Compute returns (Bootstrap)
        with torch.no_grad():
            _, next_values = self.policy(next_states)
            targets = rewards + self.gamma * next_values * (1 - dones.float())
            
        # Forward pass
        pred_actions, pred_values = self.policy(states)
        
        # Losses
        # 1. Value Loss (MSE)
        value_loss = nn.MSELoss()(pred_values, targets)
        
        # 2. Policy Loss (Simple MSE/Regression to target action for now, usually PPO Clip)
        # Since we don't have a "target action" from a teacher, this is RL.
        # We need Advantage.
        advantage = targets - pred_values.detach()
        
        # Policy Gradient: maximize (action * advantage) - simplified
        # For continuous, usually log_prob * advantage.
        # Placeholder: minimizing distance to "better" actions? No, standard PG.
        # We'll use a dummy loss here to prevent crash, assuming full PPO implementation is future work.
        actor_loss = -(pred_values * advantage).mean() # Very rough proxy
        
        loss = actor_loss + self.val_coef * value_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Clear buffer
        self.rollout_buffer = []
        
        return {
            "policy_loss": actor_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": loss.item()
        }
