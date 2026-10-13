"""Does the binding layer change the winner? (scripts/analysis/probe_binding_decisive.py)

Every expected number is computed by hand.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.analysis import probe_binding_decisive as probe


def record(modulated, bound, sync_r=0.2):
    names = ("vision", "audio", "memory", "body", "semantic")
    return {"sync_r": sync_r, "bound_bids": dict(zip(names, bound)),
            "internals": {"modulated_bids": dict(zip(names, modulated))}}


class TestLeader:
    def test_the_highest_bid_leads(self):
        assert probe.leader({"vision": 0.4, "audio": 0.7, "memory": 0.1}) == "audio"

    def test_a_tie_at_the_top_has_no_leader(self):
        assert probe.leader({"vision": 0.7, "audio": 0.7, "memory": 0.1}) is None

    def test_all_zero_has_no_leader(self):
        assert probe.leader({"vision": 0.0, "audio": 0.0}) is None


class TestBindingRow:
    def test_a_uniform_factor_keeps_the_leader(self):
        row = probe.binding_row(record([0.6, 0.5, 0.1, 0.2, 0.0], [0.9, 0.75, 0.15, 0.3, 0.0]))
        assert row["before"] == "vision" and row["after"] == "vision" and row["changed"] is False
        assert row["factors"]["vision"] == pytest.approx(1.5)
        assert row["factors"]["audio"] == pytest.approx(1.5)

    def test_a_larger_factor_for_the_second_module_changes_the_leader(self):
        row = probe.binding_row(record([0.6, 0.5, 0.1, 0.2, 0.0], [0.6, 0.75, 0.1, 0.2, 0.0]))
        assert row["before"] == "vision" and row["after"] == "audio" and row["changed"] is True

    def test_a_module_with_a_bid_of_zero_has_no_factor(self):
        row = probe.binding_row(record([0.6, 0.5, 0.1, 0.2, 0.0], [0.9, 0.75, 0.15, 0.3, 0.0]))
        assert "semantic" not in row["factors"]

    def test_a_step_with_no_leader_is_not_a_row(self):
        assert probe.binding_row(record([0.5, 0.5, 0.1, 0.2, 0.0], [0.7, 0.8, 0.1, 0.2, 0.0])) is None


class TestSummary:
    RECORDS = [
        record([0.6, 0.5, 0.1, 0.2, 0.0], [0.9, 0.75, 0.15, 0.3, 0.0]),      # kept
        record([0.6, 0.5, 0.1, 0.2, 0.0], [0.6, 0.75, 0.1, 0.2, 0.0]),       # changed
        record([0.2, 0.5, 0.1, 0.2, 0.0], [0.3, 0.75, 0.15, 0.3, 0.0]),      # kept
        record([0.5, 0.5, 0.1, 0.2, 0.0], [0.7, 0.8, 0.1, 0.2, 0.0]),        # tie, left out
    ]

    def test_share_of_steps_where_the_leader_changed(self):
        summary = probe.summarise(self.RECORDS)
        assert summary["steps"] == 3 and summary["left_out"] == 1
        assert summary["changed_share"] == pytest.approx(1 / 3)

    def test_the_gap_of_the_two_sense_factors_is_the_mean_absolute_difference(self):
        summary = probe.summarise(self.RECORDS)
        # vision and audio factors: (1.5, 1.5), (1.0, 1.5), (1.5, 1.5)
        assert summary["sense_factor_gap"] == pytest.approx(0.5 / 3)

    def test_no_step_gives_none_and_not_zero(self):
        summary = probe.summarise([])
        assert summary["changed_share"] is None and summary["sense_factor_gap"] is None


class TestRSquared:
    def test_an_exact_linear_relation_gives_one(self):
        bids = np.array([[0.1, 0.2], [0.4, 0.1], [0.3, 0.9], [0.8, 0.5]])
        assert probe.r_squared(bids, 0.5 * bids[:, 0] + 0.25 * bids[:, 1] + 0.1) == pytest.approx(1.0)

    def test_a_target_with_no_relation_gives_zero(self):
        bids = np.array([[0.0], [1.0], [0.0], [1.0]])
        assert probe.r_squared(bids, np.array([1.0, 1.0, 3.0, 3.0])) == pytest.approx(0.0)

    def test_a_constant_target_cannot_be_scored(self):
        assert probe.r_squared(np.array([[0.0], [1.0], [2.0]]), np.array([0.45, 0.45, 0.45])) is None


class TestVerdict:
    @staticmethod
    def seed(changed, r2):
        return {"changed_share": changed, "r_squared": r2}

    def test_supported_when_both_hold_at_eight_seeds(self):
        seeds = [self.seed(0.01, 0.95)] * 8 + [self.seed(0.20, 0.50)] * 2
        assert probe.verdict(seeds) == "SUPPORTED"

    def test_not_supported_when_one_prediction_fails_at_three_seeds(self):
        seeds = [self.seed(0.01, 0.95)] * 7 + [self.seed(0.20, 0.95)] * 3
        assert probe.verdict(seeds) == "NOT SUPPORTED"

    def test_an_unmeasured_seed_gives_undecided(self):
        seeds = [self.seed(0.01, 0.95)] * 7 + [self.seed(None, None)] * 3
        assert probe.verdict(seeds) == "UNDECIDED"
