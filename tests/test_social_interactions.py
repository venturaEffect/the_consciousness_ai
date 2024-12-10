# File: /tests/test_social_interactions.py
"""
Unit tests for Social Interactions Module

Tests interaction success and evaluation criteria.
"""
import unittest
from simulations.scenarios.social_interactions import SocialInteraction, SocialInteractionManager, negotiation_success

class TestSocialInteractions(unittest.TestCase):
    def setUp(self):
        self.manager = SocialInteractionManager()
        self.interaction = SocialInteraction(
            interaction_id="interaction_1",
            participants=["agent_1", "human_1"],
            scenario="Negotiate resource allocation.",
            success_criteria=negotiation_success
        )
        self.manager.add_interaction(self.interaction)

    def test_interaction_success(self):
        interaction_state = {"agreement_reached": True}
        self.manager.evaluate_interactions(interaction_state)
        self.assertTrue(self.interaction.completed)

    def test_interaction_failure(self):
        interaction_state = {"agreement_reached": False}
        self.manager.evaluate_interactions(interaction_state)
        self.assertFalse(self.interaction.completed)

if __name__ == "__main__":
    unittest.main()
