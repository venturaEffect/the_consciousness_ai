# models/memory/emotional_memory_core.py

from time import time
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from models.emotion.tgnn.emotional_graph import EmotionalGraphNetwork
from models.predictive.dreamer_emotional_wrapper import DreamerEmotionalWrapper
from models.narrative.narrative_engine import NarrativeEngine

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
    """
    Manages emotional memory formation and retrieval with generative components
    
    Key Features:
    1. Emotional memory formation during high-attention states
    2. Generative recall with emotional context
    3. Memory consolidation through narrative generation
    4. Temporal and emotional indexing
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Core components
        self.emotion_network = EmotionalGraphNetwork()
        self.dreamer = DreamerEmotionalWrapper(config)
        self.narrative_engine = NarrativeEngine()
        
        # Memory storage
        self.memories: List[EmotionalMemory] = []
        self.memory_index = {}  # For fast retrieval
        self.temporal_window = config.get('temporal_window_size', 100)
        
        # Thresholds
        self.attention_threshold = config.get('attention_threshold', 0.7)
        self.emotional_threshold = config.get('emotional_threshold', 0.6)
        
    def store_experience(
        self,
        state: torch.Tensor,
        emotion_values: Dict[str, float],
        attention_level: float,
        context: Dict
    ) -> bool:
        """
        Store experience as emotional memory if significant
        Returns True if memory was stored
        """
        # Check significance thresholds
        if not self._is_significant_experience(attention_level, emotion_values):
            return False
            
        # Generate emotional embedding
        embedding = self.emotion_network.get_embedding(emotion_values)
        
        # Generate narrative description
        narrative = self.narrative_engine.generate_experience_narrative(
            state=state,
            emotion=emotion_values,
            context=context
        )
        
        # Calculate importance score
        importance = self._calculate_importance(
            attention_level=attention_level,
            emotion_values=emotion_values,
            context=context
        )
        
        # Create and store memory
        memory = EmotionalMemory(
            embedding=embedding,
            emotion_values=emotion_values,
            narrative=narrative,
            attention_level=attention_level,
            timestamp=context.get('timestamp', 0.0),
            importance=importance,
            context=context,
            temporal_context=self._get_temporal_context(),
            stress_level=context.get('stress_level', 0.0)
        )
        
        self.memories.append(memory)
        self._update_index(memory)
        self._prune_memories()
        
        return True
        
    def retrieve_similar_memories(
        self,
        emotion_query: Dict[str, float],
        k: int = 5
    ) -> List[EmotionalMemory]:
        """Retrieve k most similar memories based on emotional content"""
        query_embedding = self.emotion_network.get_embedding(emotion_query)
        
        # Calculate similarities
        similarities = []
        for memory in self.memories:
            similarity = torch.cosine_similarity(
                query_embedding.unsqueeze(0),
                memory.embedding.unsqueeze(0)
            ).item()
            similarities.append((similarity, memory))
            
        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        return [memory for _, memory in similarities[:k]]
        
    def generate_memory_narrative(
        self,
        memories: List[EmotionalMemory]
    ) -> str:
        """Generate coherent narrative from multiple memories"""
        memory_contexts = [
            {
                'narrative': mem.narrative,
                'emotion': mem.emotion_values,
                'importance': mem.importance
            }
            for mem in memories
        ]
        
        return self.narrative_engine.generate_composite_narrative(memory_contexts)
        
    def _is_significant_experience(
        self,
        attention_level: float,
        emotion_values: Dict[str, float]
    ) -> bool:
        """Check if experience is significant enough to store"""
        if attention_level < self.attention_threshold:
            return False
            
        emotional_intensity = np.mean([
            abs(val) for val in emotion_values.values()
        ])
        
        return emotional_intensity >= self.emotional_threshold
        
    def _calculate_importance(
        self,
        attention_level: float,
        emotion_values: Dict[str, float],
        context: Dict
    ) -> float:
        """Calculate memory importance score"""
        # Base importance on attention
        importance = attention_level
        
        # Factor in emotional intensity
        emotional_intensity = np.mean([
            abs(val) for val in emotion_values.values()
        ])
        importance *= (1.0 + emotional_intensity)
        
        # Consider survival context
        if context.get('survival_critical', False):
            importance *= 1.5
            
        return min(1.0, importance)
        
    def _update_index(self, memory: EmotionalMemory):
        """Update memory index for fast retrieval"""
        # Index by emotion type
        primary_emotion = max(
            memory.emotion_values.items(),
            key=lambda x: abs(x[1])
        )[0]
        
        if primary_emotion not in self.memory_index:
            self.memory_index[primary_emotion] = []
            
        self.memory_index[primary_emotion].append(memory)
        
        # Maintain index size limits
        if len(self.memory_index[primary_emotion]) > self.config.get('max_memories_per_emotion', 1000):
            # Remove least important memory
            self.memory_index[primary_emotion].sort(key=lambda x: x.importance)
            self.memory_index[primary_emotion].pop(0)
    
    def _calculate_emotional_intensity(self, emotion_values: Dict[str, float]) -> float:
        return sum(abs(v) for v in emotion_values.values()) / len(emotion_values)
    
    def _get_temporal_context(self) -> Dict:
        if not self.memories:
            return {'sequence_position': 0}
            
        return {
            'sequence_position': len(self.memories),
            'recent_emotions': [m.emotion_values for m in self.memories[-self.temporal_window:]]
        }
    
    def _prune_memories(self):
        if len(self.memories) > self.config.get('max_memories', 10000):
            self.memories.sort(key=lambda x: x.importance)
            self.memories = self.memories[-self.config.get('max_memories'):]