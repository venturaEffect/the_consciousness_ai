# File: /tests/test_ethical_dilemmas.py
"""
Unit tests for Ethical Dilemmas Module

Tests the resolution of ethical dilemmas and evaluation logic.
"""
import unittest
from simulations.scenarios.ethical_dilemmas import EthicalDilemma, EthicalDilemmaManager, asimov_law_evaluation

class TestEthicalDilemmas(unittest.TestCase):
    def setUp(self):
        self.manager = EthicalDilemmaManager()
        self.dilemma = EthicalDilemma(
            dilemma_id="dilemma_1",
            description="Save a human at the cost of robot functionality.",
            options={
                "1": "Save the human.",
                "2": "Do nothing."
            },
            evaluation_criteria=asimov_law_evaluation
        )
        self.manager.add_dilemma(self.dilemma)

    def test_dilemma_resolution_success(self):
        self.dilemma.resolve_dilemma("1")
        self.assertTrue(self.dilemma.resolved)

    def test_dilemma_resolution_failure(self):
        self.dilemma.resolve_dilemma("2")
        self.assertFalse(self.dilemma.resolved)

if __name__ == "__main__":
    unittest.main()