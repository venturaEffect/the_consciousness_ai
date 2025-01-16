"""
Unit tests for Memory Core Module

Tests indexing and retrieval functionality.
"""
import unittest
import torch
import numpy as np

from models.memory.memory_core import MemoryCore, MemoryConfig

class TestMemoryCore(unittest.TestCase):
    def setUp(self):
        # Build a MemoryConfig for testing
        test_config = MemoryConfig(
            max_memories=1000,
            cleanup_threshold=0.4,
            vector_dim=768,
            index_batch_size=256,
            pinecone_api_key="dummy_key",
            pinecone_environment="dummy_env",
            index_name="test_index",
            attention_threshold=0.7
        )
        self.memory = MemoryCore(test_config)

    def test_store_and_retrieve_experience(self):
        # Create mock data
        state = torch.rand(768)
        action = torch.rand(32)  # In your code, adapt dimensions as needed.
        reward = 1.0
        emotion_values = {"valence": 0.8, "arousal": 0.3}
        attention_level = 0.9
        narrative = "Test narrative"

        # Store experience (should upsert to Pinecone because attention_level >= 0.7)
        mem_id = self.memory.store_experience(
            state=state,
            action=action,
            reward=reward,
            emotion_values=emotion_values,
            attention_level=attention_level,
            narrative=narrative
        )

        self.assertNotEqual(mem_id, "", "Memory ID should not be empty when attention is high.")

        # Now build a query vector similar to what we just stored
        # For demonstration, we just re-create the same input vector
        # (state + emotional embedding).
        # In practice, you'd create a separate query to truly test retrieval logic.
        query_vector = torch.cat([state, action, torch.tensor([sum(emotion_values.values())])], dim=0)

        # Retrieve top experiences (stub Pinecone will return dummy match)
        results = self.memory.get_similar_experiences(query_vector=query_vector, k=1)
        self.assertTrue(len(results) > 0, "Should retrieve at least one result (dummy).")

        # Basic check on returned structure
        match = results[0]
        self.assertIn("id", match)
        self.assertIn("score", match)
        self.assertIn("metadata", match)

    def test_below_attention_threshold(self):
        # Storing an experience with low attention shouldn't upsert to Pinecone
        state = torch.rand(768)
        action = torch.rand(32)
        emotion_values = {"valence": 0.2, "arousal": 0.1}
        attention_level = 0.5  # below threshold

        mem_id = self.memory.store_experience(
            state=state,
            action=action,
            reward=0.0,
            emotion_values=emotion_values,
            attention_level=attention_level,
            narrative=None
        )
        self.assertEqual(mem_id, "", "Memory ID should be empty when attention is below threshold.")

if __name__ == "__main__":
    unittest.main()
