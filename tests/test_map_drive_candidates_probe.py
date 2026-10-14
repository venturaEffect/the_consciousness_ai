"""The choice rule of `docs/decisions/2026_10_14_map_binding_drive.md`, as coded in
scripts/analysis/probe_map_drive_candidates.py."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.analysis import probe_map_drive_candidates as probe


def summary(multi, on_light, related=100):
    return {"steps": 200, "with_group": 200, "related": related, "multi": multi, "on_light": on_light}


class TestByStep:
    def test_a_vector_is_found_by_its_recorded_step_not_its_position(self):
        """Regression: episode 0 has no sound entry at step 0 (first comparison run, 2026-10-14)."""
        import numpy as np
        record = {"audio_spatial": np.array([[[0.1, 0.2]], [[0.3, 0.4]]]), "audio_spatial__steps": np.array([1, 2])}
        found = probe.by_step(record, "audio_spatial")
        assert sorted(found) == [1, 2]
        assert found[2].tolist() == [[0.3, 0.4]]


class TestSummarise:
    def test_shares_over_grouped_and_related_steps(self):
        records = [("related", 2, True), ("related", 1, False), ("unrelated", 0, False), (None, 3, True)]
        s = probe.summarise(records)
        assert s["with_group"] == 3 and s["related"] == 2
        assert s["multi"] == pytest.approx(2 / 3)
        assert s["on_light"] == pytest.approx(0.5)


class TestQualifies:
    def test_all_three_conditions_hold(self):
        assert probe.qualifies(summary(0.6, 0.7), {"a": summary(0.6, 0.7), "b": summary(0.4, 0.2, related=10)})

    def test_one_grouped_area_does_not_qualify(self):
        assert not probe.qualifies(summary(0.4, 0.9), {"a": summary(0.4, 0.9)})

    def test_exactly_half_on_the_light_does_not_qualify(self):
        assert not probe.qualifies(summary(0.9, 0.5), {"a": summary(0.9, 0.5)})

    def test_a_run_with_enough_related_steps_must_hold_too(self):
        assert not probe.qualifies(summary(0.9, 0.7), {"a": summary(0.9, 0.8), "b": summary(0.9, 0.4, related=50)})


class TestChoose:
    def test_controls_are_never_chosen(self):
        assert probe.choose({"K0": 0.9, "K1": 0.8}) is None

    def test_the_simpler_candidate_wins_within_the_margin(self):
        assert probe.choose({"K2": 0.62, "K3": 0.66}) == "K2"

    def test_the_other_candidate_wins_beyond_the_margin(self):
        assert probe.choose({"K2": 0.60, "K3": 0.66}) == "K3"

    def test_no_qualifying_candidate_gives_none(self):
        assert probe.choose({}) is None
