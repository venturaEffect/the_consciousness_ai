"""The check that the selected group of the oscillator map is on the light (scripts/analysis/probe_map_selection.py)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.analysis import probe_map_selection as probe
from simulations.environments.simple_visual_env import LIGHT_COLOUR


def frame_with_light(rows, cols):
    """A 224 by 224 frame, black, with the light colour on whole map cells (14 px each)."""
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    for r in rows:
        for c in cols:
            frame[r * 14:(r + 1) * 14, c * 14:(c + 1) * 14] = LIGHT_COLOUR
    return frame


class TestLightCells:
    def test_cells_covered_by_the_light_colour(self):
        cells = probe.light_cells(frame_with_light(range(2, 4), range(5, 8)))
        assert cells.shape == (16, 16)
        assert int(cells.sum()) == 6 and bool(cells[2, 5]) and not bool(cells[0, 0])


class TestOnLight:
    def test_half_or_more_of_the_group_on_the_light_counts(self):
        light = np.zeros((16, 16), dtype=bool)
        light[0, 0:2] = True
        assert probe.on_light([0, 1, 2, 3], light) is True      # 2 of 4
        assert probe.on_light([0, 2, 3], light) is False        # 1 of 3

    def test_an_empty_group_is_not_on_the_light(self):
        assert probe.on_light([], np.ones((16, 16), dtype=bool)) is False


class TestShare:
    def test_share_over_related_steps_only(self):
        light = np.zeros((16, 16), dtype=bool)
        light[0, 0] = True
        rows = [("related", [0], light), ("related", [5], light), ("unrelated", [5], light), (None, [0], light)]
        assert probe.share_on_light(rows) == pytest.approx(0.5)
