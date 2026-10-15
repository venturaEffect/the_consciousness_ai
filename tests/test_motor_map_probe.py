"""The path gate of the tectal motor map (scripts/analysis/probe_motor_map.py)."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.analysis import probe_motor_map as probe


def step(light, decoy, collision=False):
    return {"internals": {"info_after_step": {"_truth_light_offset": list(light),
                                              "_truth_decoy_offset": list(decoy),
                                              "collision": collision}}}


class TestToward:
    def test_moving_toward_the_target_counts(self):
        # The agent moves +5 in x: the light offset falls from 60 to 55.
        steps = [step((60, 0), (-60, 0)), step((55, 0), (-65, 0))]
        targets = [{"target": [40.0, 0.0]}, {"target": [40.0, 0.0]}]
        assert probe.toward_shares(steps, targets) == [True]

    def test_moving_away_counts_against(self):
        steps = [step((60, 0), (-60, 0)), step((65, 0), (-55, 0))]
        assert probe.toward_shares(steps, [{"target": [40.0, 0.0]}] * 2) == [False]

    def test_steps_without_a_target_are_left_out(self):
        steps = [step((60, 0), (-60, 0)), step((55, 0), (-65, 0))]
        assert probe.toward_shares(steps, [{"target": None}, {"target": None}]) == []


class TestWallAndContact:
    def test_wall_share(self):
        assert probe.wall_share([step((0, 0), (0, 0), True), step((0, 0), (0, 0))]) == pytest.approx(0.5)

    def test_contact_within_the_limit(self):
        far, near = step((100, 0), (-100, 0)), step((59, 0), (-141, 0))
        assert probe.contact([far, far, near], limit=40) is True
        assert probe.contact([far, far, near], limit=2) is False

    def test_contact_with_the_decoy_counts(self):
        assert probe.contact([step((100, 0), (-50, 0))], limit=40) is True


class TestSplit:
    def test_history_is_split_by_episode_length(self):
        history = [{"target": None}] * 5
        assert [len(part) for part in probe.split(history, [2, 3])] == [2, 3]

    def test_a_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            probe.split([{"target": None}] * 4, [2, 3])


def leveled(light, decoy, level):
    record = step(light, decoy)
    record["internals"]["info_after_step"]["_truth_decoy_level"] = level
    return record


class TestFirstContact:
    def test_the_light_first_when_the_light_is_louder(self):
        steps = [leveled((100, 0), (-100, 0), 0.5), leveled((50, 0), (-150, 0), 0.5)]
        assert probe.first_contact_louder(steps) is True

    def test_the_decoy_first_when_the_light_is_louder(self):
        steps = [leveled((100, 0), (-100, 0), 0.5), leveled((150, 0), (-50, 0), 0.5)]
        assert probe.first_contact_louder(steps) is False

    def test_the_decoy_is_louder_above_level_one(self):
        steps = [leveled((100, 0), (-100, 0), 3.0), leveled((150, 0), (-50, 0), 3.0)]
        assert probe.first_contact_louder(steps) is True

    def test_no_contact_within_the_limit_counts_as_false(self):
        steps = [leveled((100, 0), (-100, 0), 0.5)] * 3 + [leveled((50, 0), (-150, 0), 0.5)]
        assert probe.first_contact_louder(steps, limit=3) is False

    def test_level_one_gives_none(self):
        assert probe.first_contact_louder([leveled((50, 0), (-150, 0), 1.0)]) is None


class TestCompareArms:
    def test_pooled_shares_and_one_sided_fisher_p(self):
        from scipy.stats import fisher_exact
        test = {"a": [True] * 30 + [False] * 10, "b": [True] * 25 + [False] * 15}
        base = {"a": [True] * 20 + [False] * 20, "b": [True] * 22 + [False] * 18}
        result = probe.compare_arms(test, base)
        assert result["per_seed"] == pytest.approx({"a": 0.25, "b": 0.075})
        assert result["pooled_test"] == pytest.approx(55 / 80) and result["pooled_base"] == pytest.approx(42 / 80)
        assert result["p_one_sided"] == pytest.approx(fisher_exact([[55, 25], [42, 38]], alternative="greater")[1])
