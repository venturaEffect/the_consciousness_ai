# models/memory/emotional_memory_core.py  
"""
Core memory system for the Artificial Consciousness Module (ACM)

This module handles emotional memory formation and retrieval through:
1. Vector storage using Pinecone v2 for high-dimensional memory indexing
2. Emotional context integration with memories
3. Temporal sequence tracking 
4. Memory consolidation and optimization

Dependencies:
- pinecone-client==2.2.1 for vector storage
- models/emotion/emotional_processing.py for affect analysis
- models/core/consciousness_core.py for attention gating
"""

import torch
import torch.nn as nn
from typing import Dict, List
import time

@dataclass
class EmotionalMemory:
    """Represents an emotional memory with associated metadata"""
    embedding: torch.Tensor
    emotion_values: Dict[str, float]
    narrative: str
    attention_level: float
    timestamp: float
    importance: float
    context: Dict[str, any]
    temporal_context: Dict
    stress_level: float

class EmotionalMemoryCore:
    def __init__(self, config: Dict):
        """Initialize memory systems"""
        self.config = config
        self.vector_size = config.memory.vector_dimension
        self.pinecone = initialize_pinecone(config.memory.pinecone_key)
        
        # Initialize memory indices
        self.episodic_index = self.pinecone.Index("episodic-memories")
        self.semantic_index = self.pinecone.Index("semantic-memories")
        
    def store_experience(
        self,
        experience_data: Dict[str, torch.Tensor],
        emotional_context: Dict[str, float],
        attention_level: float
    ) -> bool:
        """Store new experience with emotional context"""
        # Generate memory embedding
        memory_vector = self._generate_memory_embedding(
            experience_data,
            emotional_context
        )
        
        # Store in episodic memory if attention is high
        if attention_level > self.config.memory.attention_threshold:
            self.episodic_index.upsert(
                vectors=[(str(uuid4()), memory_vector)],
                namespace="experiences"
            )