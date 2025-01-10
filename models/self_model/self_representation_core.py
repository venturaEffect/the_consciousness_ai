"""
Self Representation Core Module

Implements dynamic self-model generation and maintenance through:
1. Direct experience learning
2. Social feedback integration  
3. Meta-memory formation
4. Narrative self-understanding

Based on the research paper's MANN architecture and holon concept.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List
from dataclasses import dataclass

@dataclass 
class SelfModelState:
    """Tracks the current state of self-representation"""
    emotional_state: Dict[str, float] = None
    behavioral_patterns: Dict[str, float] = None
    social_context: Dict[str, float] = None
    belief_system: Dict[str, float] = None
    temporal_coherence: float = 0.0

class SelfRepresentationCore(nn.Module):
    """
    Core module for maintaining and updating agent's self-representation
    through both direct experience and social learning.
    """

    def __init__(self, config: Dict):
        super().__init__()
        
        # Core state representation networks
        self.emotional_network = EmotionalStateNetwork(config)
        self.behavioral_network = BehavioralPatternNetwork(config)
        self.social_network = SocialContextNetwork(config)
        self.belief_network = BeliefSystemNetwork(config)

        # Learning components
        self.meta_learner = MetaLearningModule(config)
        self.experience_buffer = ExperienceBuffer(config)
        self.social_buffer = SocialFeedbackBuffer(config)

        self.state = SelfModelState()
        self.config = config

    def update_self_model(
        self,
        current_state: Dict[str, torch.Tensor],
        social_feedback: Optional[Dict] = None,
        attention_level: float = 0.0
    ) -> Dict:
        """
        Update self-representation through experience and feedback
        
        Args:
            current_state: Current agent state including perceptions/actions
            social_feedback: Optional feedback from other agents/humans
            attention_level: Current attention/consciousness level
        """
        # Process current emotional and behavioral state
        emotional_embedding = self.emotional_network(current_state)
        behavioral_embedding = self.behavioral_network(current_state)

        # Update from social feedback if available
        if social_feedback:
            social_embedding = self.social_network(social_feedback)
            self._integrate_social_feedback(social_embedding)

        # Generate self-model update
        self_model_update = self.meta_learner.get_update(
            emotional_state=emotional_embedding,
            behavioral_state=behavioral_embedding,
            social_context=social_embedding if social_feedback else None,
            attention_level=attention_level
        )

        # Update state if significant
        if attention_level > self.config['update_threshold']:
            self._update_state(self_model_update)
            self._store_experience(
                state=current_state,
                update=self_model_update,
                social_feedback=social_feedback
            )

        return {
            'self_model_state': self.state,
            'update_info': self_model_update,
            'coherence': self._calculate_coherence()
        }

    def _integrate_social_feedback(self, social_embedding: torch.Tensor):
        """Integrate learning from social interactions"""
        self.social_buffer.add(social_embedding)
        social_update = self.belief_network.update(social_embedding)
        self.state.belief_system.update(social_update)

    def _calculate_coherence(self) -> float:
        """Calculate temporal coherence of self-model"""
        return self.meta_learner.evaluate_coherence(
            current_state=self.state,
            experience_buffer=self.experience_buffer
        )