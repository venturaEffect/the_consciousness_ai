# models/memory/emotional_memory_core.py

"""
Emotional memory system implementing:
- Integration with LLaMA 3.3 narrative states
- Meta-memory for experience weighting
- Controlled adaptation mechanisms
- Pattern reinforcement
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from models.memory.memory_store import MemoryStore
from models.emotion.tgnn.emotional_graph import EmotionalGraphNetwork
from models.core.consciousness_gating import ConsciousnessGate
from models.predictive.emotional_predictor import EmotionalPredictor

@dataclass
class EmotionalMemoryState:
    """Track emotional memory state"""
    stability: float = 0.0
    coherence: float = 0.0
    emotional_activation: float = 0.0
    meta_memory_weight: float = 0.0
    narrative_confidence: float = 0.0

@dataclass 
class MemoryMetrics:
    """Track memory system performance"""
    stability: float = 0.0
    coherence: float = 0.0
    pattern_strength: float = 0.0
    adaptation_rate: float = 0.0
    narrative_alignment: float = 0.0

class EmotionalMemoryCore(nn.Module):
    def __init__(self, config):
        """Initialize emotional memory system"""
        super().__init__()
        
        # Memory components
        self.memory_store = MemoryStore(config)
        self.emotional_graph = EmotionalGraphNetwork()
        self.consciousness_gate = ConsciousnessGate(config)
        
        # Meta-memory tracking
        self.meta_memories = {
            'stable_patterns': [],
            'novel_experiences': [],
            'reinforcement_weights': {}
        }
        
        # Control parameters
        self.novelty_threshold = config.memory.novelty_threshold
        self.stability_threshold = config.memory.stability_threshold
        self.initial_weight = 0.1  # Low initial weight
        
        # Metrics tracking
        self.metrics = MemoryMetrics()
        
    def store_experience(
        self,
        experience: torch.Tensor,
        emotional_context: Optional[Dict] = None,
        narrative_state: Optional[Dict] = None
    ) -> str:
        """Store new experience with controlled adaptation"""
        
        # Generate experience embedding
        experience_embedding = self.experience_encoder(experience)
        
        # Calculate stability score
        stability_score = self._calculate_stability(
            experience_embedding,
            emotional_context
        )
        
        # Handle novel experiences with low initial weight
        if stability_score < self.novelty_threshold:
            memory_key = self._store_novel_experience(
                experience_embedding,
                emotional_context,
                narrative_state
            )
            
        # Reinforce existing patterns
        else:
            memory_key = self._reinforce_pattern(
                experience_embedding,
                emotional_context,
                narrative_state
            )
            
        # Update metrics
        self._update_metrics(
            stability_score,
            emotional_context,
            narrative_state
        )
        
        return memory_key

    def process_experience(
        self,
        input_state: Dict[str, torch.Tensor],
        emotional_context: Optional[Dict] = None,
        narrative_context: Optional[Dict] = None
    ) -> Tuple[Dict, EmotionalMemoryState]:
        """Process new experiences through enhanced emotional memory pipeline"""
        
        # Generate emotional embedding
        emotional_embedding = self.emotional_graph(
            input_state,
            self.meta_memories['stable_patterns']
        )
        
        # Gate information based on consciousness state
        gated_state = self.consciousness_gate(
            emotional_embedding,
            narrative_context
        )
        
        # Predict emotional outcomes
        predictions = self.emotional_predictor(
            gated_state,
            emotional_context
        )
        
        # Update meta-memory with controlled adaptation
        stability_score = self._update_meta_memory(
            emotional_embedding,
            predictions,
            narrative_context
        )
        
        # Store experience in memory
        memory_key = self.memory_store.store(
            gated_state,
            emotional_embedding,
            stability_score
        )
        
        # Track current state
        current_state = EmotionalMemoryState(
            stability=stability_score,
            coherence=predictions['coherence_score'],
            emotional_activation=emotional_embedding.mean().item(),
            meta_memory_weight=len(self.meta_memories['stable_patterns']),
            narrative_confidence=narrative_context['confidence'] if narrative_context else 0.0
        )
        
        return {
            'memory_key': memory_key,
            'emotional_embedding': emotional_embedding,
            'predictions': predictions,
            'meta_memory_state': self.meta_memories
        }, current_state
        
    def _update_meta_memory(
        self,
        emotional_embedding: torch.Tensor,
        predictions: Dict,
        narrative_context: Optional[Dict]
    ) -> float:
        """Update meta-memory with controlled adaptation"""
        
        # Calculate stability
        stability_score = self._calculate_stability(
            emotional_embedding,
            predictions,
            narrative_context
        )
        
        # Handle novel experiences with low weight
        if stability_score < self.novelty_threshold:
            self.meta_memories['novel_experiences'].append({
                'embedding': emotional_embedding.detach(),
                'predictions': predictions,
                'weight': 0.1  # Low initial weight
            })
            
        # Reinforce stable patterns
        elif stability_score > self.stability_threshold:
            self._reinforce_pattern(
                emotional_embedding,
                predictions,
                narrative_context
            )
            
        return stability_score