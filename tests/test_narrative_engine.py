import unittest
from models.narrative.narrative_engine import NarrativeEngine

class TestNarrativeEngine(unittest.TestCase):
    def setUp(self):
        self.engine = NarrativeEngine()
        
    def test_narrative_generation(self):
        current_state = {"location": "virtual_room", "action": "observe"}
        emotional_context = {"valence": 0.7, "arousal": 0.3}
        narrative = self.engine.generate_narrative(current_state, emotional_context)
        self.assertIsInstance(narrative, str)
        self.assertGreater(len(narrative), 0)