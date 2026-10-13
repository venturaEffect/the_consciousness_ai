"""Probe of the bid change in the reentrant cycles (scripts/analysis/probe_reentrant_boost.py).

Every expected number is counted by hand on constructed step records.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.analysis.probe_reentrant_boost import bid_change, pair_directions, summarise, verdict


def record(raw_vision, raw_audio, final_vision, final_audio):
    return {"raw_bids": {"vision": raw_vision, "audio": raw_audio, "memory": 0.1},
            "internals": {"settle": {"final_bids": {"vision": final_vision, "audio": final_audio, "memory": 0.1}}}}


class TestBidChange:
    def test_audio_leads_and_vision_gets_the_change(self):
        change = bid_change(record(0.50, 0.60, 0.65, 0.60))
        assert change["leader"] == "audio"
        assert change["leader_change"] == pytest.approx(0.0)
        assert change["other_change"] == pytest.approx(0.15)
        assert change["flipped"] is True

    def test_vision_leads_and_keeps_the_lead(self):
        change = bid_change(record(0.80, 0.30, 0.90, 0.30))
        assert change["leader"] == "vision"
        assert change["leader_change"] == pytest.approx(0.10)
        assert change["other_change"] == pytest.approx(0.0)
        assert change["flipped"] is False

    def test_a_tie_has_no_leader(self):
        assert bid_change(record(0.5, 0.5, 0.6, 0.5)) is None

    def test_a_step_where_one_sense_bids_zero_is_left_out(self):
        assert bid_change(record(0.7, 0.0, 0.7, 0.0)) is None

    def test_a_final_tie_is_not_a_flip(self):
        assert bid_change(record(0.50, 0.60, 0.60, 0.60))["flipped"] is False


class TestSummarise:
    ROWS = [
        (0, record(0.50, 0.60, 0.65, 0.60), False),   # audio leads, flips
        (0, record(0.50, 0.70, 0.55, 0.70), False),   # audio leads, stays
        (0, record(0.80, 0.30, 0.90, 0.30), True),    # vision leads, stays
        (0, record(0.50, 0.50, 0.50, 0.50), True),    # tie, left out
    ]

    def test_counts_and_means_by_leader(self):
        summary = summarise(self.ROWS)
        assert summary["audio"]["steps"] == 2
        assert summary["audio"]["leader_change"] == pytest.approx(0.0)
        assert summary["audio"]["other_change"] == pytest.approx((0.15 + 0.05) / 2)
        assert summary["audio"]["flip_share"] == pytest.approx(0.5)
        assert summary["vision"]["steps"] == 1
        assert summary["vision"]["leader_change"] == pytest.approx(0.10)
        assert summary["vision"]["flip_share"] == pytest.approx(0.0)
        assert summary["left_out"] == 1

    def test_a_leader_with_no_step_reports_none_and_not_zero(self):
        summary = summarise([(0, record(0.80, 0.30, 0.90, 0.30), True)])
        assert summary["audio"] == {"steps": 0, "leader_change": None, "other_change": None, "flip_share": None}


class TestPairDirections:
    def test_reads_the_three_predictions(self):
        legacy = {"audio": {"steps": 5, "leader_change": 0.10, "other_change": 0.01, "flip_share": 0.05},
                  "vision": {"steps": 5, "leader_change": 0.08, "other_change": 0.00, "flip_share": 0.0}}
        top = {"audio": {"steps": 5, "leader_change": 0.00, "other_change": 0.12, "flip_share": 0.60},
               "vision": {"steps": 5, "leader_change": 0.01, "other_change": 0.09, "flip_share": 0.0}}
        assert pair_directions(legacy, top) == {"D1": True, "D2": True, "D3": True}

    def test_a_missing_value_gives_none_for_that_prediction(self):
        empty = {"steps": 0, "leader_change": None, "other_change": None, "flip_share": None}
        full = {"steps": 5, "leader_change": 0.10, "other_change": 0.01, "flip_share": 0.05}
        directions = pair_directions({"audio": empty, "vision": full}, {"audio": full, "vision": full})
        assert directions["D1"] is None and directions["D3"] is None


class TestVerdict:
    TRUE = {"D1": True, "D2": True, "D3": True}

    def test_supported_needs_every_prediction_true_at_every_pair(self):
        assert verdict([self.TRUE] * 6) == "SUPPORTED"

    def test_one_false_pair_is_undecided(self):
        assert verdict([self.TRUE] * 5 + [{"D1": True, "D2": False, "D3": True}]) == "UNDECIDED"

    def test_two_false_pairs_of_one_prediction_are_not_supported(self):
        assert verdict([self.TRUE] * 4 + [{"D1": True, "D2": False, "D3": True}] * 2) == "NOT SUPPORTED"

    def test_a_pair_that_is_not_measurable_is_undecided(self):
        assert verdict([self.TRUE] * 5 + [{"D1": None, "D2": True, "D3": None}]) == "UNDECIDED"
