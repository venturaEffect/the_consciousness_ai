"""
Enhanced Memory Integration Module

Implements a holonic memory architecture that combines:
1. Emotional indexing and contextual storage
2. Experience-based learning through high-attention states
3. Dynamic self-representation updates
4. Temporal coherence maintenance

Based on the MANN (Modular Artificial Neural Networks) architecture for self-consciousness.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class MemoryMetrics:
    """Tracks memory system performance and coherence"""
    temporal_coherence: float = 0.0
    emotional_stability: float = 0.0
    retrieval_quality: float = 0.0
    consciousness_relevance: float = 0.0

class MemoryIntegrationCore(nn.Module):
    """
    Core memory system implementing holonic principles from the paper.
    Each memory component acts both independently and as part of the whole.
    """

    def __init__(self, config: Dict):
        super().__init__()
        
        # Core networks
        self.feature_encoder = nn.Sequential(
            nn.Linear(config['input_dim'], config['hidden_dim']),
            nn.LayerNorm(config['hidden_dim']),
            nn.GELU(),
            nn.Linear(config['hidden_dim'], config['embedding_dim'])
        )
        
        self.emotional_encoder = EmotionalContextNetwork(config)
        self.temporal_encoder = TemporalContextNetwork(config)
        self.consciousness_gate = ConsciousnessGating(config)
        
        # Memory systems
        self.episodic_memory = EpisodicMemoryStore(config)
        self.semantic_memory = SemanticMemoryStore(config)
        
        self.metrics = MemoryMetrics()

    def store_experience(
        self,
        experience_data: Dict[str, torch.Tensor],
        emotional_context: Dict[str, float],
        consciousness_level: float,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Store new experience with emotional and consciousness context
        
        Args:
            experience_data: Raw experience tensors
            emotional_context: Emotional state values
            consciousness_level: Current consciousness level
            metadata: Optional additional context
        """
        # Generate embeddings
        feature_embedding = self.feature_encoder(experience_data['state'])
        emotional_embedding = self.emotional_encoder(emotional_context)
        temporal_embedding = self.temporal_encoder(experience_data['timestamp'])
        
        # Gate storage based on consciousness
        if self.consciousness_gate(consciousness_level):
            # Store in memory systems
            episodic_success = self.episodic_memory.store(
                feature_embedding, 
                emotional_embedding,
                temporal_embedding,
                metadata
            )
            
            semantic_success = self.semantic_memory.update(
                feature_embedding,
                emotional_embedding
            )
            
            # Update metrics
            self._update_metrics(
                feature_embedding,
                emotional_embedding, 
                consciousness_level
            )
            
            return episodic_success and semantic_success
            
        return False

    def retrieve_memory(
        self,
        query: Dict[str, torch.Tensor],
        emotional_context: Optional[Dict[str, float]] = None,
        consciousness_level: float = 0.0,
        k: int = 5
    ) -> List[Dict]:
        """
        Retrieve relevant memories based on query and context
        """
        # Generate query embeddings
        query_embedding = self.feature_encoder(query['state'])
        emotional_query = self.emotional_encoder(emotional_context) if emotional_context else None
        
        # Get relevant memories
        episodic_memories = self.episodic_memory.search(
            query_embedding,
            emotional_query,
            k=k
        )
        
        semantic_concepts = self.semantic_memory.search(
            query_embedding,
            k=k
        )
        
        return {
            'episodic': episodic_memories,
            'semantic': semantic_concepts,
            'metrics': self.get_metrics()
        }

    def _update_metrics(
        self,
        feature_embedding: torch.Tensor,
        emotional_embedding: torch.Tensor,
        consciousness_level: float
    ):
        """Update memory system metrics"""
        self.metrics.temporal_coherence = self._calculate_temporal_coherence()
        self.metrics.emotional_stability = self._calculate_emotional_stability()
        self.metrics.retrieval_quality = self._evaluate_retrieval_quality()
        self.metrics.consciousness_relevance = consciousness_level