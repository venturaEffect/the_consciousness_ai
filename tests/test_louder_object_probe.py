"""The check that the map selects the object with the louder tone (scripts/analysis/probe_louder_object.py)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.analysis import probe_louder_object as probe
from scripts.analysis.probe_map_selection import light_cells
from simulations.environments.simple_visual_env import SimpleVisualEnv


def both_in_view(seed=5):
    """A real frame of the room with the agent between the light and the visible decoy."""
    np.random.seed(seed)
    env = SimpleVisualEnv(width=224, height=224, audio="binaural", audio_channels=4,
                          view="agent_centered", decoy_sound=True, decoy_visible=True)
    env.seed_audio(seed)
    env.seed_decoy(seed)
    for _ in range(100):
        env.reset()
        middle = (env.light_pos + env.decoy_pos) / 2.0
        if np.all(np.abs(env.light_pos - middle) <= env.view_radius):
            break
    env.agent_pos = middle.astype(np.float32)
    frame, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
    return env, frame, info


class TestFrameCentre:
    def test_the_agent_place_is_the_centre_of_the_frame(self):
        assert probe.frame_centre([0.0, 0.0]) == pytest.approx((112.0, 112.0))

    def test_the_window_edge_is_the_frame_edge(self):
        # Offset -48 room px is the left and top edge; +48 the right and bottom edge.
        assert probe.frame_centre([-48.0, 48.0]) == pytest.approx((0.0, 224.0))


class TestObjectCells:
    def test_the_two_masks_split_the_light_coloured_cells_of_a_real_frame(self):
        env, frame, info = both_in_view()
        assert info["_truth_light_in_view"] and info["_truth_decoy_in_view"]
        masks = probe.object_cells(frame, info["_truth_light_offset"], info["_truth_decoy_offset"])
        assert not np.any(masks["light"] & masks["decoy"])
        assert np.array_equal(masks["light"] | masks["decoy"], light_cells(frame))
        assert masks["light"].sum() > 0 and masks["decoy"].sum() > 0

    def test_each_mask_lies_on_the_side_of_its_object(self):
        env, frame, info = both_in_view()
        masks = probe.object_cells(frame, info["_truth_light_offset"], info["_truth_decoy_offset"])
        rows, cols = np.mgrid[0:16, 0:16]
        light_x, decoy_x = info["_truth_light_offset"][0], info["_truth_decoy_offset"][0]
        if light_x != decoy_x:
            sign = 1.0 if light_x > decoy_x else -1.0
            assert sign * (cols[masks["light"]].mean() - cols[masks["decoy"]].mean()) > 0


class TestLouderObject:
    def test_related_means_the_light_and_unrelated_the_decoy(self):
        assert probe.louder_object("related") == "light"
        assert probe.louder_object("unrelated") == "decoy"
        assert probe.louder_object(None) is None


class TestShare:
    def test_share_over_steps_with_both_objects_in_view(self):
        light = np.zeros((16, 16), dtype=bool)
        decoy = np.zeros((16, 16), dtype=bool)
        light[0, 0], decoy[0, 5] = True, True
        masks = {"light": light, "decoy": decoy}
        rows = [("related", [0], masks, True),      # on the louder object
                ("unrelated", [0], masks, True),    # on the quiet object
                ("unrelated", [5], masks, True),    # on the louder object
                ("related", [5], masks, False)]     # not both in view: not counted
        assert probe.share_on_louder(rows) == pytest.approx(2.0 / 3.0)

    def test_no_counted_step_gives_none(self):
        assert probe.share_on_louder([]) is None
