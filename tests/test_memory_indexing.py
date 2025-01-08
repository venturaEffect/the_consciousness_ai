# tests/test_memory_indexing.py

import unittest
import torch
import numpy as np
from typing import Dict, List
from models.memory.emotional_indexing import EmotionalMemoryIndex
from models.memory.emotional_memory_core import EmotionalMemoryCore
from models.emotion.tgnn.emotional_graph import EmotionalGraphNetwork

class TestMemoryIndexing(unittest.TestCase):
    """Test suite for emotional memory indexing and retrieval system"""

    def setUp(self):
        self.config = {
            'vector_dimension': 768,
            'index_name': 'test_emotional_memories',
            'embedding_batch_size': 32,
            'consciousness_thresholds': {
                'emotional_awareness': 0.7,
                'memory_coherence': 0.6,
                'attention_level': 0.8
            }
        }
        
        # Initialize core components
        self.memory_index = EmotionalMemoryIndex(self.config)
        self.memory_core = EmotionalMemoryCore(self.config)
        self.emotion_network = EmotionalGraphNetwork()
        
    def test_memory_storage_and_retrieval(self):
        """Test basic memory storage and retrieval functionality"""
        # Create test memory
        test_memory = {
            'state': torch.randn(32),
            'emotion_values': {
                'valence': 0.8,  # Positive emotion
                'arousal': 0.7,
                'dominance': 0.6
            },
            'attention_level': 0.9,
            'narrative': "Successfully completed challenging task with positive outcome"
        }
        
        # Store memory
        memory_id = self.memory_index.store_memory(
            state=test_memory['state'],
            emotion_values=test_memory['emotion_values'],
            attention_level=test_memory['attention_level'],
            narrative=test_memory['narrative']
        )
        
        # Retrieve similar memories
        retrieved_memories = self.memory_index.retrieve_similar_memories(
            emotion_query=test_memory['emotion_values'],
            k=1
        )
        
        # Verify retrieval
        self.assertEqual(len(retrieved_memories), 1)
        self.assertGreater(retrieved_memories[0]['similarity'], 0.8)
        
    def test_emotional_coherence(self):
        """Test emotional coherence in memory sequences"""
        # Create sequence of related memories
        memories = []
        base_valence = 0.3  # Start with negative emotion
        
        for i in range(5):
            memory = {
                'state': torch.randn(32),
                'emotion_values': {
                    'valence': min(1.0, base_valence + 0.1 * i),  # Gradually improving
                    'arousal': 0.7,
                    'dominance': 0.5 + 0.05 * i
                },
                'attention_level': 0.8 + 0.02 * i,
                'narrative': f"Memory {i} in emotional sequence"
            }
            memories.append(memory)
            
        # Store memories
        for memory in memories:
            self.memory_index.store_memory(
                state=memory['state'],
                emotion_values=memory['emotion_values'],
                attention_level=memory['attention_level'],
                narrative=memory['narrative']
            )
            
        # Verify temporal coherence
        temporal_sequence = self.memory_index.get_temporal_sequence(
            start_time=0.0,
            end_time=float('inf')
        )
        
        self.assertEqual(len(temporal_sequence), 5)
        
        # Check emotional progression
        valences = [mem['emotion_values']['valence'] for mem in temporal_sequence]
        self.assertTrue(all(x <= y for x, y in zip(valences, valences[1:])))
        
    def test_consciousness_relevant_retrieval(self):
        """Test retrieval based on consciousness relevance"""
        # Create memories with varying consciousness scores
        high_consciousness_memory = {
            'state': torch.randn(32),
            'emotion_values': {
                'valence': 0.8,
                'arousal': 0.9,
                'dominance': 0.7
            },
            'attention_level': 0.95,
            'narrative': "Highly conscious experience with deep emotional impact"
        }
        
        low_consciousness_memory = {
            'state': torch.randn(32),
            'emotion_values': {
                'valence': 0.4,
                'arousal': 0.3,
                'dominance': 0.4
            },
            'attention_level': 0.5,
            'narrative': "Low consciousness routine experience"
        }
        
        # Store memories
        self.memory_index.store_memory(
            state=high_consciousness_memory['state'],
            emotion_values=high_consciousness_memory['emotion_values'],
            attention_level=high_consciousness_memory['attention_level'],
            narrative=high_consciousness_memory['narrative']
        )
        
        self.memory_index.store_memory(
            state=low_consciousness_memory['state'],
            emotion_values=low_consciousness_memory['emotion_values'],
            attention_level=low_consciousness_memory['attention_level'],
            narrative=low_consciousness_memory['narrative']
        )
        
        # Retrieve with consciousness threshold
        retrieved = self.memory_index.retrieve_similar_memories(
            emotion_query={'valence': 0.6, 'arousal': 0.6, 'dominance': 0.6},
            min_consciousness_score=0.8
        )
        
        # Verify only high consciousness memories are retrieved
        self.assertEqual(len(retrieved), 1)
        self.assertGreater(retrieved[0]['consciousness_score'], 0.8)
        
    def test_memory_statistics(self):
        """Test memory statistics and metrics tracking"""
        # Store series of memories
        for i in range(10):
            memory = {
                'state': torch.randn(32),
                'emotion_values': {
                    'valence': np.random.random(),
                    'arousal': np.random.random(),
                    'dominance': np.random.random()
                },
                'attention_level': 0.7 + 0.02 * i,
                'narrative': f"Test memory {i}"
            }
            self.memory_index.store_memory(
                state=memory['state'],
                emotion_values=memory['emotion_values'],
                attention_level=memory['attention_level'],
                narrative=memory['narrative']
            )
            
        # Get statistics
        stats = self.memory_index.memory_stats
        
        # Verify statistics
        self.assertIn('emotional_coherence', stats)
        self.assertIn('temporal_consistency', stats)
        self.assertIn('consciousness_relevance', stats)
        self.assertTrue(all(0 <= v <= 1 for v in stats.values()))
        
if __name__ == '__main__':
    unittest.main()