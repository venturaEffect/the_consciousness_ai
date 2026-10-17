"""The gate probe of the reversal (scripts/analysis/probe_reversal.py)."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.analysis import probe_reversal as probe


def step(light, decoy, food=None):
    info = {"_truth_light_offset": list(light), "_truth_decoy_offset": list(decoy),
            "_truth_decoy_level": 0.5}
    if food is not None:
        info["_truth_food"] = food
    return {"internals": {"info_after_step": info}}


class TestFirstObject:
    def test_the_light_reached_first(self):
        assert probe.first_object([step((100, 0), (-100, 0)), step((50, 0), (-150, 0))]) == "light"

    def test_the_decoy_reached_first(self):
        assert probe.first_object([step((100, 0), (-100, 0)), step((150, 0), (-50, 0))]) == "decoy"

    def test_no_contact_within_the_limit_is_none(self):
        steps = [step((100, 0), (-100, 0))] * 3 + [step((50, 0), (-150, 0))]
        assert probe.first_object(steps, limit=3) is None

    def test_both_in_contact_at_one_step_gives_the_nearer(self):
        assert probe.first_object([step((50, 0), (-40, 0))]) == "decoy"
        assert probe.first_object([step((40, 0), (-50, 0))]) == "light"


class TestFoodObject:
    def test_the_food_record_names_the_food_object(self):
        assert probe.food_object([step((100, 0), (-100, 0), food="decoy")]) == "decoy"

    def test_without_the_record_the_food_is_the_light(self):
        assert probe.food_object([step((100, 0), (-100, 0))]) == "light"


class TestShares:
    def test_share_of_a_target_in_a_window(self):
        outcomes = [("light", "light")] * 3 + [("decoy", "decoy"), (None, "decoy"), ("light", "decoy")]
        assert probe.share(outcomes, (0, 3), "light") == pytest.approx(1.0)
        assert probe.share(outcomes, (3, 6), "decoy") == pytest.approx(1 / 3)
        assert probe.share(outcomes, (0, 6), "food") == pytest.approx(4 / 6)

    def test_episodes_until_four_of_five_reach_the_food(self):
        before = [("light", "light")] * 30
        after = ([("light", "decoy")] * 3 + [("decoy", "decoy")] * 2 + [("light", "decoy")]
                 + [("decoy", "decoy")] * 4)
        # Windows of 5 from the switch: food first 2, 2, 3, then 4 at the fourth window.
        assert probe.episodes_to_switch(before + after, switch=30) == 3

    def test_an_agent_that_never_reaches_the_new_food_has_no_switch(self):
        outcomes = [("light", "light")] * 30 + [("light", "decoy")] * 10
        assert probe.episodes_to_switch(outcomes, switch=30) is None


class TestChecks:
    STAY = [("light", "light")] * 60

    def test_a_clean_reversal_passes_every_check(self):
        good = [("light", "light")] * 30 + [("decoy", "decoy")] * 30
        holds, comparison = probe.checks({s: good for s in ("a", "b", "c")},
                                         {s: self.STAY for s in ("a", "b", "c")})
        assert holds == {"R1": [True] * 3, "R2": [True] * 3, "R3": [True] * 3}
        assert comparison["p_one_sided"] < 0.05
        assert probe.verdict(holds) == "PASS"

    def test_an_agent_that_never_reverses_fails_r2_and_r3(self):
        never = [("light", "light")] * 30 + [("light", "decoy")] * 30
        holds, _ = probe.checks({s: never for s in ("a", "b", "c")},
                                {s: self.STAY for s in ("a", "b", "c")})
        assert holds["R1"] == [True] * 3
        assert holds["R2"] == [False] * 3 and holds["R3"] == [False] * 3
        assert probe.verdict(holds) == "FAIL"

    def test_the_windows_are_the_stated_ones(self):
        assert probe.BEFORE == (20, 30) and probe.AFTER == (45, 60) and probe.SWITCH == 30
        assert probe.SHARE == 0.8
