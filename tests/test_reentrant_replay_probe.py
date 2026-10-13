"""Replay of the reentrant bid rule (scripts/analysis/probe_reentrant_replay.py).

Every expected number is computed by hand. The vision vector has size 1 and the audio
vector has size 16, at a right angle, so |audio - vision| is sqrt(257).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.analysis.probe_reentrant_replay import (
    feedback_bid,
    prediction_error,
    replay_settle,
    run_table,
    verdict,
)

VISION = np.array([1.0, 0.0, 0.0])
AUDIO = np.array([0.0, 16.0, 0.0])
VECTORS = {"vision": VISION, "audio": AUDIO}


class TestRule:
    def test_error_above_the_limit_adds_a_tenth_of_it(self):
        assert feedback_bid(0.40, 1.0) == pytest.approx(0.50)

    def test_the_bid_stops_at_one(self):
        assert feedback_bid(0.40, 12.0) == 1.0

    def test_error_at_or_below_the_limit_shrinks_the_bid(self):
        assert feedback_bid(0.50, 0.2) == pytest.approx(0.50 * (1 - 0.02))
        assert feedback_bid(0.50, 0.0) == pytest.approx(0.50)

    def test_error_is_the_distance_over_the_size_of_the_own_vector(self):
        assert prediction_error(AUDIO, VISION, unit=False) == pytest.approx(math.sqrt(257.0))
        assert prediction_error(VISION, AUDIO, unit=False) == pytest.approx(math.sqrt(257.0) / 16.0)

    def test_unit_size_gives_both_modules_the_same_error(self):
        assert prediction_error(AUDIO, VISION, unit=True) == pytest.approx(math.sqrt(2.0))
        assert prediction_error(VISION, AUDIO, unit=True) == pytest.approx(math.sqrt(2.0))


class TestReplay:
    def test_recorded_sizes_flip_an_audio_lead(self):
        final, cycles = replay_settle({"vision": 0.40, "audio": 0.62}, VECTORS, unit=False)
        assert final["vision"] == 1.0
        assert final["audio"] == pytest.approx(0.62 + 0.1 * math.sqrt(257.0) / 16.0)
        assert cycles == 3

    def test_unit_size_keeps_this_audio_lead(self):
        final, cycles = replay_settle({"vision": 0.40, "audio": 0.62}, VECTORS, unit=True)
        assert final["vision"] == pytest.approx(0.40 + 0.1 * math.sqrt(2.0))
        assert final["audio"] == pytest.approx(0.62)
        assert cycles == 2

    def test_unit_size_flips_a_lead_smaller_than_the_gain(self):
        final, _ = replay_settle({"vision": 0.55, "audio": 0.62}, VECTORS, unit=True)
        assert final["vision"] > final["audio"]

    def test_the_replay_stops_at_the_cycle_limit(self):
        _, cycles = replay_settle({"vision": 0.40, "audio": 0.62}, VECTORS, unit=False, max_cycles=2)
        assert cycles == 2


def step(raw_vision, raw_audio, final_vision, final_audio):
    record = {"raw_bids": {"vision": raw_vision, "audio": raw_audio},
              "internals": {"settle": {"final_bids": {"vision": final_vision, "audio": final_audio}}}}
    return record, VISION, AUDIO


class TestRunTable:
    STEPS = [
        step(0.40, 0.62, 1.0, 0.72),     # audio leads, recorded flip, replay flips, unit keeps
        step(0.55, 0.62, 1.0, 0.72),     # audio leads, recorded flip, replay flips, unit flips
        step(0.40, 0.62, 0.40, 0.62),    # audio leads, recorded no flip, replay flips: a disagreement
        step(0.90, 0.30, 0.90, 0.40),    # vision leads, nobody flips
        step(0.50, 0.50, 0.50, 0.50),    # tie, left out
    ]

    def test_counts_by_leader(self):
        table = run_table(self.STEPS)
        audio = table["audio"]
        assert audio["steps"] == 3
        assert audio["recorded_flip_share"] == pytest.approx(2 / 3)
        assert audio["replay_flip_share"] == pytest.approx(1.0)
        assert audio["agreement"] == pytest.approx(2 / 3)
        assert audio["unit_flip_share"] == pytest.approx(1 / 3)
        vision = table["vision"]
        assert vision["steps"] == 1
        assert vision["recorded_flip_share"] == 0.0
        assert vision["agreement"] == 1.0

    def test_a_leader_with_no_step_reports_none(self):
        table = run_table([step(0.90, 0.30, 0.90, 0.40)])
        assert table["audio"] == {"steps": 0, "recorded_flip_share": None, "replay_flip_share": None,
                                  "agreement": None, "unit_flip_share": None}


class TestVerdict:
    @staticmethod
    def table(agreement, replay, unit):
        row = {"steps": 10, "recorded_flip_share": replay, "replay_flip_share": replay, "agreement": agreement, "unit_flip_share": unit}
        return {"audio": row, "vision": dict(row, replay_flip_share=0.0, unit_flip_share=0.0)}

    def test_supported_when_unit_size_halves_the_flips_at_every_run(self):
        assert verdict([self.table(0.95, 0.60, 0.20)] * 6) == "SUPPORTED"

    def test_not_supported_when_two_runs_keep_half_or_more(self):
        assert verdict([self.table(0.95, 0.60, 0.20)] * 4 + [self.table(0.95, 0.60, 0.35)] * 2) == "NOT SUPPORTED"

    def test_undecided_when_one_run_keeps_half_or_more(self):
        assert verdict([self.table(0.95, 0.60, 0.20)] * 5 + [self.table(0.95, 0.60, 0.30)]) == "UNDECIDED"

    def test_void_when_the_replay_does_not_match_the_record(self):
        assert verdict([self.table(0.95, 0.60, 0.20)] * 5 + [self.table(0.80, 0.60, 0.20)]) == "VOID"
