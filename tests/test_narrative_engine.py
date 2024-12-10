# Setting up Tests for ACM Project

# File: /tests/test_narrative_engine.py
"""
Unit tests for Narrative Engine Module

Tests the generation of coherent narratives and integration with memory core.
"""
import unittest
from models.narrative.narrative_engine import NarrativeEngine

class TestNarrativeEngine(unittest.TestCase):
    def setUp(self):
        self.engine = NarrativeEngine()

    def test_generate_narrative(self):
        input_text = "Explain the implications of saving a human in danger."
        response = self.engine.generate_narrative(input_text)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_memory_integration(self):
        input_text = "What happened in the last task?"
        self.engine.memory_context = ["The agent successfully completed the task."]
        response = self.engine.generate_narrative(input_text)
        self.assertIn("completed the task", response)

if __name__ == "__main__":
    unittest.main()