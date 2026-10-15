"""The orientation test of the visible decoy (scripts/analysis/probe_orientation.py)."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.analysis import probe_orientation as probe


def step(light, decoy, level):
    return {"internals": {"info_after_step": {"_truth_light_offset": list(light),
                                              "_truth_decoy_offset": list(decoy),
                                              "_truth_decoy_level": level}}}


class TestLouderAtStart:
    def test_the_level_decides_because_the_agent_starts_midway(self):
        assert probe.louder_at_start(0.5) == "light"
        assert probe.louder_at_start(3.0) == "decoy"
        assert probe.louder_at_start(1.0) is None


class TestOriented:
    def test_closer_to_the_louder_object_after_k_steps(self):
        steps = [step((60, 0), (-60, 0), 0.5)] * 9 + [step((20, 0), (-100, 0), 0.5)]
        assert probe.oriented(steps, k=10) is True

    def test_closer_to_the_quieter_object(self):
        steps = [step((60, 0), (-60, 0), 3.0)] * 9 + [step((20, 0), (-100, 0), 3.0)]
        assert probe.oriented(steps, k=10) is False

    def test_equal_distances_give_none(self):
        steps = [step((60, 0), (-60, 0), 0.5)] * 10
        assert probe.oriented(steps, k=10) is None

    def test_an_episode_shorter_than_k_gives_none(self):
        assert probe.oriented([step((60, 0), (-60, 0), 0.5)] * 5, k=10) is None

    def test_the_level_is_read_at_the_first_step(self):
        steps = [step((60, 0), (-60, 0), 3.0)] + [step((60, 0), (-60, 0), 0.5)] * 8 + [step((100, 0), (-20, 0), 0.5)]
        assert probe.oriented(steps, k=10) is True


class TestCompare:
    def test_pooled_difference_and_one_sided_fisher_p(self):
        on = {"a": [True] * 30 + [False] * 10, "b": [True] * 25 + [False] * 15}
        off = {"a": [True] * 20 + [False] * 20, "b": [True] * 20 + [False] * 20}
        result = probe.compare(on, off)
        assert result["per_seed"]["a"] == pytest.approx(30 / 40 - 20 / 40)
        assert result["per_seed"]["b"] == pytest.approx(25 / 40 - 20 / 40)
        assert result["pooled_on"] == pytest.approx(55 / 80)
        assert result["pooled_off"] == pytest.approx(40 / 80)
        from scipy.stats import fisher_exact
        expected = fisher_exact([[55, 25], [40, 40]], alternative="greater")[1]
        assert result["p_one_sided"] == pytest.approx(expected)

    def test_none_outcomes_are_left_out(self):
        result = probe.compare({"a": [True, None]}, {"a": [False, None]})
        assert result["per_seed"]["a"] == pytest.approx(1.0)
        assert result["episodes_on"] == 1 and result["episodes_off"] == 1
