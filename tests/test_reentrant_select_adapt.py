"""The `select_adapt` feedback rule of ReentrantProcessor.

`docs/decisions/2026_10_14_reentrant_feedback_rule_review.md`, option A with C. In each
cycle the module that leads the competition gains (1 + g) and the other senses lose
(1 - g). After each step the module that won adapts, so its next bids start lower. The
constants g = 0.1, rate = 0.1 and depth = 0.5 are assumed values, not fitted. Every
expected number is computed by hand.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.core.reentrant_processor import ReentrantProcessor


class FakeWorkspace:
    """Returns a fixed broadcast and winner list. The broadcast changes once, so the loop runs 2 cycles."""

    def __init__(self, winners):
        self.state = SimpleNamespace(winners=list(winners), phi_value=0.0, is_conscious=bool(winners))
        self.calls = []

    def run_competition(self, inputs, goal_vector, bids, payloads, pad_state, interoceptive_state):
        self.calls.append(dict(bids))
        return torch.ones(4) * len(self.calls), dict(bids)


class Sense:
    """A sense module. The legacy rule calls receive_broadcast. The new rule must not."""

    def __init__(self):
        self.called = 0

    def receive_broadcast(self, broadcast, bid):
        self.called += 1
        return bid


def settle(processor, winners, bids):
    senses = {"vision": Sense(), "audio": Sense(), "memory": object()}
    workspace = FakeWorkspace(winners)
    result = processor.settle(workspace, senses, dict(bids), {}, torch.zeros(4))
    return result, workspace, senses


def processor(rule="select_adapt"):
    return ReentrantProcessor({"max_cycles": 2, "min_cycles": 2, "convergence_threshold": 0.0, "feedback_rule": rule})


class TestSelect:
    def test_the_leader_gains_and_the_other_sense_loses_in_each_cycle(self):
        result, _, senses = settle(processor(), ["audio", "vision"], {"vision": 0.5, "audio": 0.4, "memory": 0.1})
        # 2 cycles: audio 0.4 * 1.1 * 1.1, vision 0.5 * 0.9 * 0.9, memory has no feedback
        assert result.final_bids["audio"] == pytest.approx(0.4 * 1.1 * 1.1)
        assert result.final_bids["vision"] == pytest.approx(0.5 * 0.9 * 0.9)
        assert result.final_bids["memory"] == pytest.approx(0.1)
        assert senses["vision"].called == 0 and senses["audio"].called == 0

    def test_a_bid_stops_at_one(self):
        result, _, _ = settle(processor(), ["vision"], {"vision": 0.95, "audio": 0.4})
        assert result.final_bids["vision"] == 1.0

    def test_without_a_winner_the_bids_do_not_change(self):
        result, _, _ = settle(processor(), [], {"vision": 0.5, "audio": 0.4})
        assert result.final_bids == {"vision": 0.5, "audio": 0.4}


class TestAdapt:
    def test_the_winner_starts_the_next_step_lower(self):
        rule = processor()
        settle(rule, ["audio"], {"vision": 0.5, "audio": 0.4})
        # audio won: its adaptation is 0.1, vision stays 0
        assert rule.adaptation == {"audio": pytest.approx(0.1), "vision": pytest.approx(0.0)}
        _, workspace, _ = settle(rule, ["audio"], {"vision": 0.5, "audio": 0.4})
        assert workspace.calls[0]["audio"] == pytest.approx(0.4 * (1 - 0.5 * 0.1))
        assert workspace.calls[0]["vision"] == pytest.approx(0.5)

    def test_adaptation_rises_toward_one_and_recovers_when_another_sense_wins(self):
        rule = processor()
        settle(rule, ["audio"], {"vision": 0.5, "audio": 0.4})
        settle(rule, ["audio"], {"vision": 0.5, "audio": 0.4})
        assert rule.adaptation["audio"] == pytest.approx(0.1 + 0.1 * 0.9)
        settle(rule, ["vision"], {"vision": 0.5, "audio": 0.4})
        assert rule.adaptation["audio"] == pytest.approx((0.1 + 0.1 * 0.9) * 0.9)
        assert rule.adaptation["vision"] == pytest.approx(0.1)

    def test_a_step_without_a_winner_lets_every_sense_recover(self):
        rule = processor()
        settle(rule, ["audio"], {"vision": 0.5, "audio": 0.4})
        settle(rule, [], {"vision": 0.5, "audio": 0.4})
        assert rule.adaptation["audio"] == pytest.approx(0.1 * 0.9)


class TestLegacy:
    def test_the_default_rule_is_legacy_and_calls_the_modules(self):
        rule = ReentrantProcessor({"max_cycles": 2, "min_cycles": 2, "convergence_threshold": 0.0})
        assert rule.feedback_rule == "legacy"
        _, _, senses = settle(rule, ["audio"], {"vision": 0.5, "audio": 0.4})
        assert senses["vision"].called == 2 and senses["audio"].called == 2

    def test_an_unknown_rule_is_refused(self):
        with pytest.raises(ValueError):
            ReentrantProcessor({"feedback_rule": "other"})
