"""Gate G1 probe (scripts/analysis/probe_binding_selectivity.py). Values computed by hand."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.analysis import probe_binding_selectivity as probe


def info(light, decoy, in_view=True):
    return {"_truth_light_offset": list(light), "_truth_decoy_offset": list(decoy), "_truth_light_in_view": in_view}


class TestCondition:
    def test_the_nearer_source_is_the_louder_one(self):
        assert probe.condition(info((3, 4), (30, 40)), True) == "related"
        assert probe.condition(info((30, 40), (3, 4)), True) == "unrelated"

    def test_no_condition_without_the_light_in_view_or_without_a_decoy(self):
        assert probe.condition(info((3, 4), (30, 40)), False) is None
        assert probe.condition({"_truth_light_offset": [3, 4]}, True) is None

    def test_the_level_of_the_decoy_is_part_of_its_loudness(self):
        # distances 5 and 10: amplitudes 0.5/(1+5/56) and 0.5/(1+10/56); a level of 3 makes the decoy louder
        assert probe.condition(info((3, 4), (6, 8)), True) == "related"
        assert probe.condition(info((3, 4), (6, 8)), True, level=3.0) == "unrelated"

    def test_a_recorded_level_comes_before_the_given_one(self):
        recorded = dict(info((3, 4), (6, 8)), _truth_decoy_level=3.0)
        assert probe.condition(recorded, True, level=1.0) == "unrelated"

    def test_episode_levels_follow_the_extra_reset_of_the_training_script(self):
        assert [probe.episode_level(e, (0.5, 3.0)) for e in range(4)] == [3.0, 0.5, 3.0, 0.5]
        assert probe.episode_level(7, (1.5,)) == 1.5

    def test_equal_loudness_has_no_condition(self):
        assert probe.condition(info((3, 4), (4, 3)), True) is None


class TestJointAlignment:
    def test_the_alignment_is_read_back_from_the_factor(self):
        record = {"internals": {"modulated_bids": {"vision": 0.4, "audio": 0.2}},
                  "bound_bids": {"vision": 0.4 * 1.5, "audio": 0.2 * 1.25}}
        # alignments 1.0 and 0.5
        assert probe.joint_alignment(record) == pytest.approx(0.75)

    def test_a_sense_with_no_bid_gives_none(self):
        record = {"internals": {"modulated_bids": {"vision": 0.4, "audio": 0.0}},
                  "bound_bids": {"vision": 0.6, "audio": 0.0}}
        assert probe.joint_alignment(record) is None


class TestResiduals:
    def test_a_target_made_of_the_bids_leaves_nothing(self):
        bids = np.array([[0.1, 0.2], [0.4, 0.1], [0.3, 0.9], [0.8, 0.5]])
        assert np.allclose(probe.residuals(bids, 2.0 * bids[:, 0] - bids[:, 1] + 0.3), 0.0, atol=1e-12)


class TestShuffle:
    def test_a_clear_difference_has_a_small_p(self):
        values = np.array([1.0] * 20 + [0.0] * 20)
        labels = np.array([True] * 20 + [False] * 20)
        result = probe.shuffle_test(values, labels, shuffles=200)
        assert result["difference"] == pytest.approx(1.0)
        assert result["p"] == pytest.approx(1 / 201)

    def test_equal_groups_have_a_p_of_one(self):
        result = probe.shuffle_test(np.ones(10), np.array([True, False] * 5), shuffles=50)
        assert result["difference"] == pytest.approx(0.0)
        assert result["p"] == pytest.approx(1.0)


class TestVerdict:
    @staticmethod
    def seed(p, valid=True):
        return {"valid": valid, "p": p}

    def test_two_seeds_with_no_effect_confirm(self):
        assert probe.verdict([self.seed(0.4), self.seed(0.2), self.seed(0.01)]) == "CONFIRMED"

    def test_two_seeds_with_an_effect_reject(self):
        assert probe.verdict([self.seed(0.4), self.seed(0.01), self.seed(0.03)]) == "REJECTED"

    def test_too_few_steps_give_void(self):
        assert probe.verdict([self.seed(0.4), self.seed(0.4), self.seed(None, valid=False)]) == "VOID"

    def test_too_few_steps_of_one_condition_are_not_valid(self):
        rows = [{"condition": "related", "joint": 0.5, "bids": [0.1] * 5}] * 150 + \
               [{"condition": "unrelated", "joint": 0.5, "bids": [0.1] * 5}] * 99
        assert probe.evaluate(rows)["valid"] is False
