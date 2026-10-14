"""Stage 2b of the binding design: the agreement is the visual activity at the place of the sound.

Values computed by hand. The grid has a feature norm of 1 on most cells (the background),
4 on a block (the object) and 7 on one cell (the peak). The median is 1 and the maximum 7,
so the agreement at a cell is (norm - 1) / 6, cut to [0, 1].
"""
from __future__ import annotations

import argparse
import os
import sys
from unittest import mock

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.core.global_workspace import GlobalWorkspace
from models.core.sensory_tectum import SensoryTectum, overlap_agreement


def grid():
    g = torch.zeros(1, 1, 16, 16)
    g[0, 0] = 1.0
    g[0, 0, 2:6, 10:14] = 4.0      # the object, rows 2 to 5, columns 10 to 13
    g[0, 0, 3, 12] = 7.0           # its peak
    return g


def sound_at(column, row):
    spatial = torch.zeros(1, 4, 2)
    spatial[0, 0, 0] = (column + 0.5) / 15 * 2 - 1
    spatial[0, 0, 1] = (row + 0.5) / 15 * 2 - 1
    return spatial


class TestOverlapAgreement:
    def test_a_sound_on_the_object_but_not_on_its_peak_agrees_in_part(self):
        assert overlap_agreement(grid(), sound_at(10, 5)) == pytest.approx(3 / 6)

    def test_a_sound_on_the_peak_agrees_fully(self):
        assert overlap_agreement(grid(), sound_at(12, 3)) == pytest.approx(1.0)

    def test_a_sound_on_the_background_does_not_agree(self):
        assert overlap_agreement(grid(), sound_at(1, 14)) == pytest.approx(0.0)

    def test_a_flat_map_gives_zero_and_no_sound_gives_none(self):
        assert overlap_agreement(torch.ones(1, 1, 16, 16), sound_at(3, 3)) == 0.0
        assert overlap_agreement(grid(), torch.zeros(1, 4, 2)) is None


class TestSwitches:
    def test_the_workspace_accepts_overlap(self):
        assert GlobalWorkspace({"binding_relation": "overlap"}).binding_relation == "overlap"
        with pytest.raises(ValueError):
            GlobalWorkspace({"binding_relation": "overlap", "binding_mechanism": "komplex"})

    def test_the_tectum_uses_the_mode_it_is_given(self):
        tectum = SensoryTectum({})
        assert tectum.place_agreement_mode == "place"

    def test_the_training_flag_offers_overlap(self):
        from scripts.training import train_rlhf
        captured = {}

        def capture(self, *args, **kwargs):
            captured["parser"] = self
            raise RuntimeError("captured")

        with mock.patch.object(argparse.ArgumentParser, "parse_args", capture):
            with pytest.raises(RuntimeError):
                train_rlhf.main()
        assert captured["parser"].parse_args(["--binding-relation", "overlap"]).binding_relation == "overlap"
