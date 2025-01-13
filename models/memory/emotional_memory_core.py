# models/memory/emotional_memory_core.py  
"""
Emotional Memory Core for the ACM's memory management system.

This module handles:
1. Storage and retrieval of emotional memories
2. Integration with Pinecone for vector storage
3. Memory consolidation and optimization
4. Emotional context indexing

Dependencies:
- models/evaluation/memory_evaluation.py for memory metrics
- models/memory/memory_store.py for base memory functionality
- models/memory/emotional_indexing.py for emotion-based indexing
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
        """Initialize emotional memory system"""
        self.config = config
        
        # Initialize memory components
        self.memory_store = MemoryStore(config)
        self.emotional_indexer = EmotionalIndexing(config)
        self.evaluator = MemoryEvaluator(config)
        
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