# File: /tests/test_memory_core.py
"""
Unit tests for Memory Core Module

Tests indexing and retrieval functionality.
"""
import unittest
import numpy as np
from models.memory.memory_core import MemoryCore

class TestMemoryCore(unittest.TestCase):
    def setUp(self):
        self.memory = MemoryCore(api_key="dummy_key", index_name="test_index")

    def test_store_memory(self):
        embedding = np.random.rand(768)
        metadata = {"description": "Test memory"}
        self.memory.store_memory(embedding, metadata)
        # Validate storage by querying
        results = self.memory.retrieve_memory(embedding)
        self.assertGreater(len(results), 0)

    def test_retrieve_memory(self):
        embedding = np.random.rand(768)
        self.memory.store_memory(embedding)
        results = self.memory.retrieve_memory(embedding)
        self.assertGreater(len(results), 0)

if __name__ == "__main__":
    unittest.main()