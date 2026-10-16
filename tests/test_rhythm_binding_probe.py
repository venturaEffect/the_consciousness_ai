"""The gate probe of binding by timing (scripts/analysis/probe_rhythm_binding.py)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.analysis import probe_rhythm_binding as probe
from simulations.environments.audio_mixin import binaural_amplitude


def cell(row, col):
    return row * 16 + col


class TestGeometry:
    def test_object_places_are_the_cell_centres_within_40_room_pixels(self):
        mask = probe.object_cells([45.0, 3.0])          # the centre of cell (8, 15)
        assert mask[cell(8, 15)] and mask[cell(8, 9)] and not mask[cell(8, 7)]
        # Cell (8, 9) is 36 room pixels left of it; cell (8, 8) is 42.
        assert not mask[cell(8, 8)]

    def test_the_louder_object_uses_level_and_distance(self):
        near_decoy = {"_truth_light_offset": [100.0, 0.0], "_truth_decoy_offset": [-10.0, 0.0],
                      "_truth_decoy_level": 0.5}
        light = binaural_amplitude(100.0)
        decoy = 0.5 * binaural_amplitude(10.0)
        assert probe.louder_now(near_decoy) == ("decoy" if decoy > light else "light")
        far = dict(near_decoy, _truth_decoy_offset=[-100.0, 0.0])
        assert probe.louder_now(far) == "light"


def row(coherence_on, coherence_off, louder="light", group=None, peak_on=0.9, peak_off=0.9, level=0.5):
    """One counted step: the light covers cells (2, 2) and (2, 3), the decoy (12, 12) and (12, 13)."""
    coherence = np.zeros(256)
    peak = np.zeros(256)
    light = np.zeros(256, dtype=bool)
    decoy = np.zeros(256, dtype=bool)
    light[[cell(2, 2), cell(2, 3)]] = True
    decoy[[cell(12, 12), cell(12, 13)]] = True
    on, off = (light, decoy) if louder == "light" else (decoy, light)
    coherence[on], coherence[off] = coherence_on, coherence_off
    peak[on], peak[off] = peak_on, peak_off
    return {"coherence": coherence, "peak": peak, "light": light, "decoy": decoy, "louder": louder,
            "group": list(np.flatnonzero(on)) if group is None else group, "level": level}


class TestEpisodeMeasures:
    def test_the_coherence_difference_is_louder_minus_other(self):
        rows = [row(0.9, 0.2), row(0.7, 0.4, louder="decoy")]
        assert probe.coherence_difference(rows) == pytest.approx(((0.9 - 0.2) + (0.7 - 0.4)) / 2)

    def test_the_luminance_difference_is_louder_minus_other(self):
        rows = [row(0.9, 0.2, peak_on=0.95, peak_off=0.75)]
        assert probe.luminance_difference(rows) == pytest.approx(0.2)

    def test_selection_counts_a_group_on_the_louder_object(self):
        rows = [row(0.9, 0.2), row(0.9, 0.2, group=[cell(12, 12), cell(12, 13)]), row(0.9, 0.2, group=[])]
        assert probe.selection_share(rows) == pytest.approx(1 / 3)

    def test_g4_reads_the_light_at_level_half_and_the_decoy_at_level_three(self):
        light_rows = [row(0.9, 0.2, louder="light", level=0.5)] * 3
        decoy_rows = [row(0.9, 0.2, louder="decoy", level=3.0)] * 2 + [row(0.9, 0.2, louder="decoy", level=3.0, group=[])]
        assert probe.g4_share(light_rows + decoy_rows, 0.5, "light") == pytest.approx(1.0)
        assert probe.g4_share(light_rows + decoy_rows, 3.0, "decoy") == pytest.approx(2 / 3)


class TestInterceptTest:
    def test_the_intercept_and_its_one_sided_p_by_hand(self):
        from scipy import stats
        x = np.array([-0.2, -0.1, 0.0, 0.1, 0.2, 0.05])
        y = np.array([0.31, 0.38, 0.42, 0.47, 0.55, 0.44])
        intercept, p = probe.intercept_test(x, y)
        slope_hand, intercept_hand = np.polyfit(x, y, 1)
        residual = y - (intercept_hand + slope_hand * x)
        n = len(x)
        s2 = (residual ** 2).sum() / (n - 2)
        se = np.sqrt(s2 * (1 / n + x.mean() ** 2 / ((x - x.mean()) ** 2).sum()))
        assert intercept == pytest.approx(intercept_hand)
        assert p == pytest.approx(stats.t.sf(intercept_hand / se, n - 2))


class TestVerdict:
    def test_pass_fail_and_undecided(self):
        assert probe.verdict({"G1": [True] * 3, "G2": [True] * 3, "G3": [True] * 3, "G4": [True] * 3}) == "PASS"
        assert probe.verdict({"G1": [True, False, False], "G2": [True] * 3, "G3": [True] * 3, "G4": [True] * 3}) == "FAIL"
        assert probe.verdict({"G1": [True, True, False], "G2": [True] * 3, "G3": [True] * 3, "G4": [True] * 3}) == "UNDECIDED"
