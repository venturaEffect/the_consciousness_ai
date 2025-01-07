import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pinecone import Pinecone, Index
from models.emotion.tgnn.emotional_graph import EmotionalGraphNetwork
from models.evaluation.consciousness_metrics import ConsciousnessMetrics

@dataclass
class MemoryMetrics:
    """Tracks memory system performance metrics"""
    coherence_score: float = 0.0
    retrieval_accuracy: float = 0.0
    emotional_context_strength: float = 0.0
    temporal_consistency: float = 0.0
    narrative_alignment: float = 0.0

class MemoryCore:
    """
    Advanced memory system for ACM that integrates:
    1. Emotional context embedding
    2. Temporal coherence tracking
    3. Consciousness-relevant memory formation
    4. Meta-learning capabilities
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.emotion_network = EmotionalGraphNetwork()
        self.consciousness_metrics = ConsciousnessMetrics(config)
        
        # Initialize Pinecone vector store
        self.pinecone = Pinecone(
            api_key=config['pinecone_api_key'],
            environment=config['pinecone_environment']
        )
        self.index = self.pinecone.Index(config['index_name'])
        
        # Memory tracking
        self.metrics = MemoryMetrics()
        self.recent_experiences = []
        self.attention_threshold = config.get('attention_threshold', 0.7)
        
    def store_experience(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        emotion_values: Dict[str, float],
        attention_level: float,
        narrative: Optional[str] = None
    ) -> str:
        """Store experience with emotional context"""
        
        # Get emotional embedding
        emotional_embedding = self.emotion_network.get_embedding(emotion_values)
        
        # Create memory vector
        memory_vector = self._create_memory_vector(
            state=state,
            action=action,
            emotional_embedding=emotional_embedding
        )
        
        # Store in Pinecone if attention level is high enough
        if attention_level >= self.attention_threshold:
            memory_id = self._generate_memory_id()
            self.index.upsert(
                vectors=[(
                    memory_id,
                    memory_vector.tolist(),
                    {
                        'emotion': emotion_values,
                        'attention': attention_level,
                        'reward': reward,
                        'narrative': narrative
                    }
                )]
            )
            
        # Update recent experiences
        self.recent_experiences.append({
            'state': state,
            'action': action,
            'emotion': emotion_values,
            'attention': attention_level,
            'reward': reward,
            'narrative': narrative,
            'vector': memory_vector
        })
        
        # Update memory metrics
        self.update_metrics()
        
        return memory_id
        
    def get_similar_experiences(
        self,
        query_vector: torch.Tensor,
        emotion_context: Optional[Dict[str, float]] = None,
        k: int = 5
    ) -> List[Dict]:
        """Retrieve similar experiences with optional emotional context"""
        
        # Add emotional context if provided
        if emotion_context is not None:
            emotional_embedding = self.emotion_network.get_embedding(emotion_context)
            query_vector = torch.cat([query_vector, emotional_embedding])
            
        # Query Pinecone
        results = self.index.query(
            vector=query_vector.tolist(),
            top_k=k,
            include_metadata=True
        )
        
        return [
            {
                'id': match.id,
                'score': match.score,
                'metadata': match.metadata
            }
            for match in results.matches
        ]
        
    def update_metrics(self):
        """Update memory system metrics"""
        if len(self.recent_experiences) < 2:
            return
            
        # Calculate coherence
        self.metrics.coherence_score = self._calculate_coherence()
        
        # Calculate retrieval accuracy
        self.metrics.retrieval_accuracy = self._calculate_retrieval_accuracy()
        
        # Calculate emotional context strength
        self.metrics.emotional_context_strength = self._calculate_emotional_strength()
        
        # Calculate temporal consistency
        self.metrics.temporal_consistency = self._calculate_temporal_consistency()
        
        # Calculate narrative alignment
        self.metrics.narrative_alignment = self._calculate_narrative_alignment()
        
    def _create_memory_vector(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        emotional_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Create combined memory vector"""
        return torch.cat([
            state,
            action,
            emotional_embedding
        ])
        
    def _calculate_coherence(self) -> float:
        """Calculate memory coherence score"""
        recent = self.recent_experiences[-100:]
        coherence_scores = []
        
        for i in range(len(recent) - 1):
            curr = recent[i]
            next_exp = recent[i + 1]
            
            # Calculate vector similarity
            similarity = torch.cosine_similarity(
                curr['vector'].unsqueeze(0),
                next_exp['vector'].unsqueeze(0)
            )
            
            coherence_scores.append(similarity.item())
            
        return np.mean(coherence_scores)
        
    def _calculate_emotional_strength(self) -> float:
        """Calculate emotional context strength"""
        recent = self.recent_experiences[-100:]
        return np.mean([
            exp['attention'] * abs(exp['emotion']['valence'])
            for exp in recent
        ])
        
    def _generate_memory_id(self) -> str:
        """Generate unique memory ID"""
        return f"mem_{len(self.recent_experiences)}_{int(time.time())}"

    def get_metrics(self) -> Dict:
        """Get current memory metrics"""
        return {
            'coherence_score': self.metrics.coherence_score,
            'retrieval_accuracy': self.metrics.retrieval_accuracy,
            'emotional_context_strength': self.metrics.emotional_context_strength,
            'temporal_consistency': self.metrics.temporal_consistency,
            'narrative_alignment': self.metrics.narrative_alignment
        }