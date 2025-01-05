# models/self_model/meta_learner.py

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from models.memory.memory_core import MemoryCore
from models.emotion.tgnn.emotional_graph import EmotionalGraphNetwork
from models.predictive.dreamerv3_wrapper import DreamerV3

class MetaLearner:
    """
    Meta-learning system for adapting to new emotional experiences and scenarios.
    Implements MAML-style meta-learning optimized for emotional reinforcement learning.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.memory = MemoryCore()
        self.emotion_network = EmotionalGraphNetwork()
        self.dreamer = DreamerV3(config.dreamer_config)
        
        # Meta-learning hyperparameters
        self.inner_lr = config.meta_config.inner_learning_rate
        self.meta_batch_size = config.meta_config.meta_batch_size
        self.adaptation_steps = config.meta_config.adaptation_steps
        
        # Initialize meta-parameters
        self.meta_parameters = {}
        self.initialize_meta_parameters()
        
    def initialize_meta_parameters(self):
        """Initialize meta-parameters for fast adaptation"""
        self.meta_parameters = {
            'emotional_scale': torch.nn.Parameter(torch.ones(1) * self.config.emotional_scale),
            'context_weights': torch.nn.Parameter(torch.randn(self.config.memory_config.context_length))
        }
        
    def inner_loop_update(self, task_data: Dict) -> Tuple[float, Dict]:
        """
        Perform inner loop update for task-specific adaptation
        """
        adapted_params = {k: v.clone() for k, v in self.meta_parameters.items()}
        task_loss = 0.0
        
        for step in range(self.adaptation_steps):
            # Sample batch from task data
            batch = self.memory.sample_batch(task_data, batch_size=self.meta_batch_size)
            
            # Compute loss with current parameters
            loss, metrics = self.compute_adaptation_loss(batch, adapted_params)
            
            # Update adapted parameters
            grads = torch.autograd.grad(loss, adapted_params.values())
            adapted_params = {
                k: v - self.inner_lr * g
                for (k, v), g in zip(adapted_params.items(), grads)
            }
            
            task_loss += loss.item()
            
        return task_loss / self.adaptation_steps, adapted_params
    
    def compute_adaptation_loss(
        self, 
        batch: Dict,
        params: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute loss for adaptation using emotional context
        """
        # Get emotional embeddings
        emotional_context = self.emotion_network.get_embeddings(batch['emotion_values'])
        
        # Apply context weights
        weighted_context = emotional_context * params['context_weights']
        
        # Get DreamerV3 predictions
        world_model_loss = self.dreamer.compute_loss(
            batch['states'],
            batch['actions'],
            batch['rewards'] * params['emotional_scale'],
            batch['next_states'],
            weighted_context
        )
        
        metrics = {
            'world_model_loss': world_model_loss.item(),
            'emotional_scale': params['emotional_scale'].item()
        }
        
        return world_model_loss, metrics
    
    def adapt_to_task(self, task_data: Dict) -> Dict:
        """
        Adapt model to new task/scenario
        """
        task_loss, adapted_params = self.inner_loop_update(task_data)
        
        # Store adapted parameters in memory for future use
        self.memory.store_adaptation({
            'task_id': task_data['task_id'],
            'adapted_params': adapted_params,
            'performance': -task_loss  # Higher is better
        })
        
        return {
            'task_loss': task_loss,
            'adapted_params': adapted_params
        }