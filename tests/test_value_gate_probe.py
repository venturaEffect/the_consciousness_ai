"""The gate probe of the value gate (scripts/analysis/probe_value_gate.py)."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.analysis import probe_value_gate as probe


def step(light, decoy, level=0.5):
    return {"internals": {"info_after_step": {"_truth_light_offset": list(light),
                                              "_truth_decoy_offset": list(decoy),
                                              "_truth_decoy_level": level}}}


class TestLightFirst:
    def test_the_light_reached_first(self):
        assert probe.light_first([step((100, 0), (-100, 0)), step((50, 0), (-150, 0))]) is True

    def test_the_decoy_reached_first(self):
        assert probe.light_first([step((100, 0), (-100, 0)), step((150, 0), (-50, 0))]) is False

    def test_no_contact_within_the_limit_is_not_the_light(self):
        steps = [step((100, 0), (-100, 0))] * 3 + [step((50, 0), (-150, 0))]
        assert probe.light_first(steps, limit=3) is False


class TestSplit:
    def test_late_episodes_and_the_louder_decoy(self):
        outcomes = [(True, False)] * 20 + [(True, True), (False, True), (True, False), (False, False)]
        late, late_loud = probe.late_shares(outcomes)
        assert late == pytest.approx(2 / 4)
        assert late_loud == pytest.approx(1 / 2)


class TestLabels:
    def test_the_seed_label_names_the_seed_not_the_arm(self):
        assert probe.seed_label("C:/runs/vg_learn_s601") == "s601"
        assert probe.seed_label("runs/vg_frozen_s602") == "s602"


class TestVerdict:
    def test_pass_fail_undecided(self):
        assert probe.verdict({"L1": [True] * 3, "L2": [True] * 3, "L3": [True] * 3}) == "PASS"
        assert probe.verdict({"L1": [True] * 3, "L2": [False, False, True], "L3": [True] * 3}) == "FAIL"
        assert probe.verdict({"L1": [True] * 3, "L2": [False, True, True], "L3": [True] * 3}) == "UNDECIDED"
