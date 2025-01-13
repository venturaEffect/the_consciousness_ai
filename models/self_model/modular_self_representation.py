"""
Modular Self Representation System for ACM

This module implements:
1. Core self-model representation and updating
2. Integration of emotional and memory contexts
3. Self-awareness development tracking
4. Modular architecture for self-model components

Dependencies:
- models/emotion/tgnn/emotional_graph.py for emotion processing
- models/memory/emotional_memory_core.py for memory integration
- models/evaluation/consciousness_monitor.py for metrics

Reference: Martinez-Luaces et al. "Using modular neural networks to model self-consciousness 
and self-representation for artificial entities"
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class HolonicState:
    """
    Tracks the holonic state of the entity following the paper's framework
    
    Attributes:
        growth_level: Current developmental stage (0-9)
        state_values: Current state vector across modalities
        self_confidence: Confidence in self-representation (affects learning rates)
        interaction_history: Record of social interactions for observational learning
    """
    growth_level: int = 0
    state_values: Dict[str, float] = None
    self_confidence: float = 0.5
    interaction_history: List[Dict] = None

@dataclass
class SelfModelState:
    """Tracks current self-model state"""
    emotional_state: Dict[str, float]
    attention_focus: float
    memory_context: List[Dict]
    consciousness_level: float

class ModularSelfRepresentation(nn.Module):
    """
    Core MANN implementation for self-representation and consciousness
    
    Features:
    1. Abstract self-representation through modular networks
    2. Direct experience learning through self-interaction
    3. Observational learning from other agents
    4. Dynamic adaptation of self-model
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Core representation networks
        self.feature_encoder = FeatureEncodingNetwork(config)
        self.social_encoder = SocialContextNetwork(config)
        self.self_model = SelfModelNetwork(config)
        
        # Learning modules
        self.direct_learner = ExperienceLearner(config)
        self.observational_learner = SocialLearner(config)
        self.meta_learner = MetaLearningNetwork(config)
        
        # Holonic state
        self.state = HolonicState()
        
        # Initialize adaptation parameters
        self._init_adaptation_params()

        # Initialize core components
        self.emotion_network = EmotionalGraphNN(config)
        self.memory = EmotionalMemoryCore(config)
        self.monitor = ConsciousnessMonitor(config)
        
        # Current state
        self.current_state = SelfModelState(
            emotional_state={},
            attention_focus=0.0,
            memory_context=[],
            consciousness_level=0.0
        )

    def update_self_representation(
        self,
        current_features: torch.Tensor,
        social_feedback: Optional[Dict] = None,
        interaction_data: Optional[Dict] = None
    ) -> Dict:
        """
        Update self-representation through direct and observational learning
        
        Args:
            current_features: Current feature vector
            social_feedback: Optional feedback from other agents
            interaction_data: Optional interaction context
            
        Returns:
            Dict containing updated self-model state and metrics
        """
        # Encode current features
        feature_embedding = self.feature_encoder(current_features)
        
        # Process social context if available
        if social_feedback:
            social_embedding = self.social_encoder(social_feedback)
            self._integrate_social_learning(social_embedding)

        # Update self-model through direct experience
        self_model_update = self.direct_learner(
            feature_embedding=feature_embedding,
            current_state=self.state
        )

        # Integrate observational learning if available
        if interaction_data:
            observational_update = self.observational_learner(
                interaction_data=interaction_data,
                current_model=self.self_model
            )
            self._integrate_observational_learning(observational_update)

        # Meta-learning update
        self.meta_learner.update(
            direct_update=self_model_update,
            observational_update=observational_update if interaction_data else None,
            current_state=self.state
        )

        return {
            'self_model_state': self.get_self_model_state(),
            'learning_metrics': self.get_learning_metrics(),
            'holonic_state': self.state
        }

    def update_self_model(
        self,
        input_state: Dict[str, torch.Tensor],
        emotional_context: Dict[str, float]
    ) -> Tuple[SelfModelState, Dict[str, float]]:
        """Update self-model based on new experience"""
        # Process emotional state
        emotional_features = self.emotion_network.process(
            input_state,
            emotional_context
        )
        
        # Update memory context
        memory_context = self.memory.retrieve_relevant(
            input_state,
            emotional_features,
            k=self.config.memory.context_size
        )
        
        # Update consciousness metrics
        consciousness_metrics = self.monitor.evaluate_state(
            current_state=self.current_state,
            new_emotional_state=emotional_features,
            memory_context=memory_context
        )
        
        # Update current state
        self.current_state = SelfModelState(
            emotional_state=emotional_features,
            attention_focus=consciousness_metrics['attention_level'],
            memory_context=memory_context,
            consciousness_level=consciousness_metrics['consciousness_score']
        )
        
        return self.current_state, consciousness_metrics

    def _integrate_social_learning(self, social_embedding: torch.Tensor):
        """Integrate learning from social interactions"""
        # Update confidence based on social feedback
        confidence_update = self.meta_learner.compute_confidence_update(
            social_embedding=social_embedding,
            current_state=self.state
        )
        self.state.self_confidence = torch.clamp(
            self.state.self_confidence + confidence_update,
            min=self.config['min_confidence'],
            max=self.config['max_confidence']
        )

    def _integrate_observational_learning(self, observational_update: Dict):
        """Integrate learning from observing other agents"""
        # Update self-model weights based on observed interactions
        self.self_model.update_weights(
            observational_update['weight_updates'],
            learning_rate=self.state.self_confidence * self.config['observational_lr']
        )