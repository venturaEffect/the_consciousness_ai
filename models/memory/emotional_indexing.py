# models/memory/emotional_indexing.py

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pinecone
from models.emotion.tgnn.emotional_graph import EmotionalGraphNetwork
from models.evaluation.consciousness_metrics import ConsciousnessMetrics

@dataclass
class MemoryIndexConfig:
    """Configuration for emotional memory indexing"""
    vector_dimension: int = 768
    index_name: str = "emotional-memories"
    metric: str = "cosine"
    pod_type: str = "p1.x1"
    embedding_batch_size: int = 32

class EmotionalMemoryIndex:
    """
    Indexes and retrieves emotional memories using vector similarity
    
    Key Features:
    1. Emotional context embedding
    2. Fast similarity search
    3. Temporal coherence tracking
    4. Consciousness-relevant retrieval
    """
    
    def __init__(self, config: MemoryIndexConfig):
        self.config = config
        
        # Initialize components
        self.emotion_network = EmotionalGraphNetwork()
        self.consciousness_metrics = ConsciousnessMetrics()
        
        # Initialize Pinecone index
        self._init_vector_store()
        
        # Memory statistics
        self.total_memories = 0
        self.memory_stats = {
            'emotional_coherence': 0.0,
            'temporal_consistency': 0.0,
            'consciousness_relevance': 0.0
        }
        
    def _init_vector_store(self):
        """Initialize Pinecone vector store"""
        if self.config.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=self.config.index_name,
                dimension=self.config.vector_dimension,
                metric=self.config.metric,
                pod_type=self.config.pod_type
            )
        self.index = pinecone.Index(self.config.index_name)
        
    def store_memory(
        self,
        state: torch.Tensor,
        emotion_values: Dict[str, float],
        attention_level: float,
        narrative: str,
        context: Optional[Dict] = None
    ) -> str:
        """Store emotional memory with indexed metadata"""
        
        # Generate emotional embedding
        emotional_embedding = self.emotion_network.get_embedding(emotion_values)
        
        # Calculate consciousness relevance
        consciousness_score = self.consciousness_metrics.evaluate_emotional_awareness(
            [{
                'state': state,
                'emotion': emotion_values,
                'attention': attention_level,
                'narrative': narrative
            }]
        )['mean_emotional_awareness']
        
        # Prepare memory vector and metadata
        memory_id = f"memory_{self.total_memories}"
        vector = emotional_embedding.cpu().numpy()
        metadata = {
            'emotion_values': emotion_values,
            'attention_level': float(attention_level),
            'narrative': narrative,
            'consciousness_score': float(consciousness_score),
            'timestamp': context.get('timestamp', 0.0) if context else 0.0
        }
        
        # Store in vector index
        self.index.upsert(
            vectors=[(memory_id, vector, metadata)],
            namespace="emotional_memories"
        )
        
        # Update statistics
        self.total_memories += 1
        self._update_memory_stats(consciousness_score)
        
        return memory_id
        
    def retrieve_similar_memories(
        self,
        emotion_query: Dict[str, float],
        k: int = 5,
        min_consciousness_score: float = 0.5
    ) -> List[Dict]:
        """Retrieve similar memories based on emotional context"""
        
        # Generate query embedding
        query_embedding = self.emotion_network.get_embedding(emotion_query)
        
        # Query vector store
        results = self.index.query(
            vector=query_embedding.cpu().numpy(),
            top_k=k * 2,  # Get extra results for filtering
            namespace="emotional_memories",
            include_metadata=True
        )
        
        # Filter and sort results
        memories = []
        for match in results.matches:
            if match.metadata['consciousness_score'] >= min_consciousness_score:
                memories.append({
                    'id': match.id,
                    'emotion_values': match.metadata['emotion_values'],
                    'attention_level': match.metadata['attention_level'],
                    'narrative': match.metadata['narrative'],
                    'consciousness_score': match.metadata['consciousness_score'],
                    'similarity': match.score
                })
                
        # Sort by similarity and consciousness score
        memories.sort(
            key=lambda x: (x['similarity'] + x['consciousness_score']) / 2,
            reverse=True
        )
        
        return memories[:k]
        
    def get_temporal_sequence(
        self,
        start_time: float,
        end_time: float,
        min_consciousness_score: float = 0.5
    ) -> List[Dict]:
        """Retrieve memories within a temporal window"""
        
        # Query vector store with time filter
        results = self.index.query(
            vector=[0] * self.config.vector_dimension,  # Dummy vector for metadata query
            namespace="emotional_memories",
            filter={
                'timestamp': {
                    '$gte': start_time,
                    '$lte': end_time
                },
                'consciousness_score': {
                    '$gte': min_consciousness_score
                }
            },
            include_metadata=True
        )
        
        # Sort by timestamp
        memories = [
            {
                'id': match.id,
                'emotion_values': match.metadata['emotion_values'],
                'attention_level': match.metadata['attention_level'],
                'narrative': match.metadata['narrative'],
                'consciousness_score': match.metadata['consciousness_score'],
                'timestamp': match.metadata['timestamp']
            }
            for match in results.matches
        ]
        memories.sort(key=lambda x: x['timestamp'])
        
        return memories
        
    def _update_memory_stats(self, consciousness_score: float):
        """Update memory statistics"""
        # Update running averages
        alpha = 0.01  # Smoothing factor
        self.memory_stats['consciousness_relevance'] = (
            (1 - alpha) * self.memory_stats['consciousness_relevance'] +
            alpha * consciousness_score
        )
        
        # Calculate temporal consistency if multiple memories exist
        if self.total_memories > 1:
            recent_memories = self.get_temporal_sequence(
                start_time=0.0,
                end_time=float('inf'),
                min_consciousness_score=0.0
            )
            
            if len(recent_memories) >= 2:
                consistency = np.mean([
                    self._calculate_temporal_consistency(m1, m2)
                    for m1, m2 in zip(recent_memories[:-1], recent_memories[1:])
                ])
                
                self.memory_stats['temporal_consistency'] = (
                    (1 - alpha) * self.memory_stats['temporal_consistency'] +
                    alpha * consistency
                )
                
    def _calculate_temporal_consistency(
        self,
        memory1: Dict,
        memory2: Dict
    ) -> float:
        """Calculate temporal consistency between consecutive memories"""
        # Compare emotional trajectories
        emotion_consistency = 1.0 - np.mean([
            abs(memory1['emotion_values'][k] - memory2['emotion_values'][k])
            for k in memory1['emotion_values']
        ])
        
        # Compare consciousness scores
        consciousness_consistency = 1.0 - abs(
            memory1['consciousness_score'] - memory2['consciousness_score']
        )
        
        return (emotion_consistency + consciousness_consistency) / 2.0