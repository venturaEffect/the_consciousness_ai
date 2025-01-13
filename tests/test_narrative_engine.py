# Setting up Tests for ACM Project

# File: /tests/test_narrative_engine.py
"""
Test suite for the Narrative Engine component of ACM.

This module validates:
1. Narrative generation and coherence
2. Integration with emotional memory
3. Context maintenance across narrative chains
4. LLaMA 3.3 integration for story construction

Dependencies:
- models/narrative/narrative_engine.py for core functionality
- models/memory/emotional_memory_core.py for context retrieval
- configs/consciousness_metrics.yaml for evaluation parameters
"""
import unittest
from models.narrative.narrative_engine import NarrativeEngine

class TestNarrativeEngine(unittest.TestCase):
    def setUp(self):
        """Initialize narrative engine test components"""
        self.config = load_config('configs/consciousness_metrics.yaml')
        self.narrative_engine = NarrativeEngine(self.config)
        self.test_cases = self._load_test_narratives()

    def test_narrative_generation(self):
        """Test narrative generation with emotional context"""
        input_text = "The agent encountered a stressful situation"
        emotional_context = {
            'valence': -0.3,
            'arousal': 0.8,
            'dominance': 0.4
        }
        
        narrative = self.narrative_engine.generate_narrative(
            input_text, 
            emotional_context
        )
        
        self.assertIsNotNone(narrative)
        self.assertTrue(len(narrative) > 0)
        self.assertIn('stress', narrative.lower())

if __name__ == "__main__":
    unittest.main()