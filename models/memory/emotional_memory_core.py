"""
Emotional Memory Core Module

Implements memory storage and retrieval with emotional context indexing.
Based on the holonic architecture described in the research paper for 
creating dynamic self-representation through experience.
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

class EmotionalMemoryCore(nn.Module):
    """
    Core class for emotion-aware memory storage and retrieval.
    
    Features:
    - Emotional context indexing
    - Attention-based memory gating
    - Temporal coherence tracking
    """

    def __init__(self, config: Dict):
        """
        Initialize memory systems.

        Args:
            config: Configuration containing:
                - memory_size: Maximum memory capacity
                - embedding_dim: Dimension of memory vectors
                - emotion_dim: Dimension of emotional context
        """
        super().__init__()
        self.memory_index = PineconeIndex(config)  # Vector store for experiences
        self.emotional_indexer = EmotionalContextIndexer(config)  # Emotion embedding
        self.attention_gate = AttentionGatingMechanism(config)  # Memory gating

    def store(
        self,
        state: torch.Tensor,
        emotion: Dict[str, float],
        attention: float
    ):
        """
        Store experience with emotional context.
        
        Implements the memory formation process described in the paper,
        using attention levels to gate storage and emotional context for indexing.

        Args:
            state: Current experience state vector
            emotion: Emotional context values
            attention: Attention level for gating
        """
        # Generate emotional embedding for indexing
        emotional_context = self.emotional_indexer(emotion)
        
        # Gate storage based on attention level
        if self.attention_gate(attention):
            # Store in vector database with metadata
            self.memory_index.upsert(
                vectors=state,
                metadata={
                    'emotional_context': emotional_context,
                    'attention_level': attention,
                    'timestamp': time.time()
                }
            )

    def retrieve_relevant(
        self,
        current_emotion: Dict[str, float],
        k: int = 5
    ) -> List[Dict]:
        """
        Retrieve emotionally relevant memories.
        
        Uses emotional context to find similar past experiences,
        implementing the emotional memory retrieval described in the paper.

        Args:
            current_emotion: Current emotional state
            k: Number of memories to retrieve

        Returns:
            List of relevant memories with metadata
        """
        emotional_query = self.emotional_indexer(current_emotion)
        return self.memory_index.query(
            vector=emotional_query,
            top_k=k,
            include_metadata=True
        )