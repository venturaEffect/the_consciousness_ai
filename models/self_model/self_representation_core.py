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
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass 
class SelfModelState:
    """Tracks the current state of the self-model"""
    emotional_state: Dict[str, float]
    attention_focus: float
    memory_context: List[Dict]
    consciousness_level: float
    confidence: float
    goals: List[Dict[str, float]]
    last_update_timestamp: float

class SelfRepresentationCore(nn.Module):
    """
    Core module for self-representation and awareness in ACM
    
    This module implements:
    1. Dynamic self-representation that evolves through experience
    2. Attention-based update mechanisms
    3. Confidence-based learning rate adaptation
    4. Integration with emotional and memory systems
    5. Temporal coherence maintenance
    """

    def __init__(self, config: Dict):
        super(SelfRepresentationCore, self).__init__()
        self.config = config
        
        # Core network components
        self.embedding_dim = config.get("embedding_dim", 256)
        self.state_encoder = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        )
        
        # Direct experience learning network
        self.direct_learner = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        
        # Social learning network for observational learning
        self.social_network = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh()
        )
        
        # Attention and confidence thresholds
        self.attention_threshold = config.get("attention_threshold", 0.5)
        self.update_threshold = config.get("update_threshold", 0.3)
        
        # Initialize self-model state
        self.state = SelfModelState(
            emotional_state={},
            attention_focus=0.0,
            memory_context=[],
            consciousness_level=0.1,
            confidence=0.2,
            goals=[{"survive": 1.0, "learn": 0.8}],
            last_update_timestamp=0.0
        )
        
        # Experience buffer for temporal coherence
        self.experience_buffer = []
        self.buffer_max_size = config.get("buffer_size", 100)

    def update_self_model(
        self,
        current_state: Dict[str, torch.Tensor],
        social_feedback: Optional[Dict] = None,
        attention_level: float = 0.0,
        timestamp: float = None
    ) -> Dict:
        """
        Update self-representation through experience and feedback
        
        Args:
            current_state: Current agent state including perceptions/actions
            social_feedback: Optional feedback from other agents/humans
            attention_level: Current attention/consciousness level
            timestamp: Current time
        """
        # Only update if attention exceeds threshold
        if attention_level < self.attention_threshold:
            return {"updated": False, "reason": "attention_below_threshold"}
        
        # Encode current state
        feature_embedding = self.state_encoder(
            torch.tensor(current_state.get("features", [0] * self.embedding_dim), 
                        dtype=torch.float32)
        )
        
        # Update from direct experience
        self_model_update = self.direct_learner(
            torch.cat([
                feature_embedding, 
                torch.tensor([self.state.consciousness_level, 
                            self.state.attention_focus] + 
                            list(self.state.emotional_state.values()),
                            dtype=torch.float32)
            ])
        )
        
        # Integrate social feedback if available
        if social_feedback:
            social_embedding = self.social_network(
                torch.tensor(social_feedback.get("features", [0] * self.embedding_dim),
                           dtype=torch.float32)
            )
            self._integrate_social_feedback(social_embedding)
        
        # Update internal state
        self._update_internal_state(
            feature_embedding, 
            current_state, 
            attention_level,
            timestamp
        )
        
        # Add experience to buffer for temporal coherence
        self._add_to_experience_buffer(current_state, attention_level)
        
        return {
            "updated": True,
            "attention_level": attention_level,
            "consciousness_level": self.state.consciousness_level,
            "confidence": self.state.confidence
        }

    def _update_internal_state(self, 
                             feature_embedding, 
                             current_state, 
                             attention_level,
                             timestamp=None):
        """Update internal components of self-model"""
        # Update emotional state if emotion data is present
        if "emotion" in current_state:
            self.state.emotional_state = {
                **self.state.emotional_state,
                **current_state["emotion"]
            }
            
        # Update attention focus
        self.state.attention_focus = (
            self.state.attention_focus * 0.7 + attention_level * 0.3
        )
        
        # Update memory context
        if "memory_context" in current_state:
            self.state.memory_context = current_state["memory_context"]
            
        # Adjust consciousness level based on attention and temporal coherence
        coherence = self._calculate_temporal_coherence()
        self.state.consciousness_level = min(1.0, 
            self.state.consciousness_level * 0.8 + 
            (attention_level * 0.1) + 
            (coherence * 0.1)
        )
        
        # Update confidence based on prediction accuracy
        if "prediction_error" in current_state:
            error = current_state["prediction_error"]
            # Lower error increases confidence
            confidence_delta = 0.05 * (1.0 - min(1.0, error))
            self.state.confidence = min(
                1.0, 
                max(0.1, self.state.confidence + confidence_delta)
            )
            
        # Update timestamp
        if timestamp:
            self.state.last_update_timestamp = timestamp

    def _integrate_social_feedback(self, social_embedding: torch.Tensor):
        """Integrate feedback from social interaction"""
        # Convert tensor to list for processing
        social_data = social_embedding.detach().numpy().tolist()
        
        # Social feedback can modify confidence
        confidence_modifier = np.mean(social_data) if len(social_data) > 0 else 0
        self.state.confidence = min(
            1.0, 
            max(0.1, self.state.confidence + (confidence_modifier * 0.1))
        )

    def _add_to_experience_buffer(self, state, attention):
        """Add experience to buffer for temporal coherence"""
        self.experience_buffer.append({
            "state": {k: v for k, v in state.items() if k != "features"}, 
            "attention": attention
        })
        
        # Keep buffer within max size
        if len(self.experience_buffer) > self.buffer_max_size:
            self.experience_buffer.pop(0)

    def _calculate_temporal_coherence(self) -> float:
        """Calculate temporal coherence of recent experiences
        
        Returns:
            Coherence score between 0.0-1.0
        """
        if len(self.experience_buffer) < 2:
            return 0.5
            
        # Simple coherence calculation
        # Higher variance in attention = lower coherence
        recent_attention = [e["attention"] for e in self.experience_buffer[-5:]]
        if len(recent_attention) > 1:
            variance = np.var(recent_attention)
            coherence = 1.0 - min(1.0, variance * 5)
            return coherence
        
        return 0.5

    def get_current_state(self) -> Dict:
        """Get current self-model state
        
        Returns:
            Dict representation of current self-model state
        """
        return {
            "emotional_state": self.state.emotional_state,
            "attention_focus": self.state.attention_focus,
            "consciousness_level": self.state.consciousness_level,
            "confidence": self.state.confidence,
            "goals": self.state.goals,
            "memories": len(self.state.memory_context)
        }