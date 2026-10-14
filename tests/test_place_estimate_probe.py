"""The place estimate diagnostic (scripts/analysis/probe_place_estimate.py). Values computed by hand."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.analysis import probe_place_estimate as probe


class TestLightCell:
    def test_the_centre_of_the_view_is_cell_8(self):
        assert probe.light_cell((0.0, 0.0)) == (8, 8)

    def test_the_edges_of_the_view(self):
        # (-48 + 48) / 96 * 16 = 0 ; (47 + 48) / 96 * 16 = 15.83 -> 15
        assert probe.light_cell((-48.0, 47.0)) == (0, 15)

    def test_a_light_outside_the_window_is_cut_to_the_edge(self):
        assert probe.light_cell((-80.0, 100.0)) == (0, 15)


class TestShares:
    @staticmethod
    def info(offset, in_view=True):
        return {"_truth_light_offset": list(offset), "_truth_light_in_view": in_view}

    def test_counts_the_peak_and_the_sound_near_the_light(self):
        history = [[0.5, [[8, 8], [8, 8]]],     # step 0, left out
                   [0.5, [[8, 8], [8, 8]]],     # light at cell (8, 8): peak near, sound near
                   [0.1, [[0, 0], [12, 8]]],    # light at (8, 8): peak 4 cells away, sound far
                   [0.1, [[0, 0], [8, 8]]]]     # light out of view, left out
        infos = [None, self.info((0, 0)), self.info((0, 0)), self.info((0, 0), in_view=False)]
        result = probe.shares(history, infos)
        assert result["in_view_steps"] == 2
        assert result["D1_peak_near_light"] == pytest.approx(0.5)
        assert result["D2_sound_near_light_when_louder"] == pytest.approx(0.5)
        assert result["peak_in_centre"] == pytest.approx(0.5)

    def test_a_record_of_another_length_is_refused(self):
        with pytest.raises(ValueError):
            probe.shares([[0.5, None]], [None, None])
