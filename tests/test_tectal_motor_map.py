"""The motor map of the tectum (models/core/tectal_motor_map.py).

`docs/decisions/2026_10_15_tectal_motor_map.md`. Expected values are computed by hand from the
geometry of the agent-centred view: 16 by 16 cells of 14 frame pixels, a window of 96 room pixels.
"""
from __future__ import annotations

import argparse
import os
import sys
from unittest import mock

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.core import tectal_motor_map as motor


def cell(row, col):
    return row * 16 + col


class TestGeometry:
    def test_cell_centres_by_hand(self):
        dx, dy = motor.cell_offsets()
        # Cell (0, 0): frame pixel 7, room offset 7 * 96 / 224 - 48 = -45.
        assert dx[0, 0] == pytest.approx(-45.0)
        assert dy[0, 0] == pytest.approx(-45.0)
        # Cell (8, 15): x = 217 * 96 / 224 - 48 = 45, y = 119 * 96 / 224 - 48 = 3.
        assert dx[8, 15] == pytest.approx(45.0)
        assert dy[8, 15] == pytest.approx(3.0)

    def test_cell_centres_agree_with_the_probe_geometry(self):
        from scripts.analysis.probe_louder_object import frame_centre
        dx, dy = motor.cell_offsets()
        for row, col in ((0, 0), (3, 11), (15, 7)):
            x, y = frame_centre([dx[row, col], dy[row, col]])
            assert x == pytest.approx((col + 0.5) * 14)
            assert y == pytest.approx((row + 0.5) * 14)

    def test_the_body_mask_has_32_cells_around_the_centre(self):
        mask = motor.body_mask()
        assert int(mask.sum()) == 32
        assert mask[7, 7] and mask[8, 8] and not mask[0, 0] and not mask[8, 15]


class TestOrientVector:
    def test_centre_of_gravity_weighted_by_drive(self):
        dx, dy = motor.cell_offsets()
        weights = np.zeros((16, 16))
        weights[8, 15], weights[0, 0] = 3.0, 1.0
        vector = motor.orient_vector([cell(8, 15), cell(0, 0)], weights)
        expected = (3.0 * np.array([dx[8, 15], dy[8, 15]]) + np.array([dx[0, 0], dy[0, 0]])) / 4.0
        assert vector == pytest.approx(expected)

    def test_body_cells_are_left_out(self):
        weights = np.ones((16, 16))
        dx, dy = motor.cell_offsets()
        vector = motor.orient_vector([cell(8, 8), cell(8, 15)], weights)
        assert vector == pytest.approx([dx[8, 15], dy[8, 15]])

    def test_a_target_only_on_the_body_gives_none(self):
        assert motor.orient_vector([cell(7, 7), cell(8, 8)], np.ones((16, 16))) is None

    def test_no_cells_or_no_weight_gives_none(self):
        assert motor.orient_vector([], np.ones((16, 16))) is None
        assert motor.orient_vector([cell(0, 0)], np.zeros((16, 16))) is None


class TestCommand:
    def test_speed_times_the_unit_vector_plus_noise(self):
        noise = np.array([0.03, -0.02])
        action = motor.command(np.array([30.0, 40.0]), noise)
        assert action == pytest.approx([0.5 * 0.6 + 0.03, 0.5 * 0.8 - 0.02])

    def test_the_command_is_clipped_to_the_action_range(self):
        assert motor.command(np.array([1.0, 0.0]), np.array([0.9, 0.0])) == pytest.approx([1.0, 0.0])


class FakeBinding:
    """The record and the drives that a map binding leaves after one step."""

    def __init__(self, group, vision, audio):
        self.last_vision, self.last_audio = torch.as_tensor(vision), torch.as_tensor(audio)
        self.record = {"group": group}


class TestTargets:
    def test_group_takes_the_selected_cells(self):
        vision = torch.zeros(16, 16)
        vision[8, 15] = 1.0
        mm = motor.TectalMotorMap("group", seed=0)
        cells, _ = mm.target(FakeBinding([cell(8, 15)], vision, torch.zeros(16, 16)), [cell(8, 15)])
        assert cells == [cell(8, 15)]

    def test_active_takes_every_cell_at_half_maximum_or_more(self):
        vision = torch.zeros(16, 16)
        vision[0, 0], vision[8, 15], vision[3, 3] = 1.0, 0.6, 0.4
        mm = motor.TectalMotorMap("active", seed=0)
        cells, _ = mm.target(FakeBinding([], vision, torch.zeros(16, 16)), [])
        assert sorted(cells) == sorted([cell(0, 0), cell(8, 15)])

    def test_audio_takes_the_sound_place(self):
        audio = torch.zeros(16, 16)
        audio[2, 13], audio[2, 12] = 1.0, 0.4
        mm = motor.TectalMotorMap("audio", seed=0)
        cells, weights = mm.target(FakeBinding([], torch.zeros(16, 16), audio), [])
        assert cells == [cell(2, 13)]
        assert weights[2, 13] == pytest.approx(1.0)

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ValueError):
            motor.TectalMotorMap("vision", seed=0)


class TestStep:
    def test_a_target_gives_a_command_toward_it(self):
        vision = torch.zeros(16, 16)
        vision[8, 15] = 1.0
        mm = motor.TectalMotorMap("group", seed=0, noise_std=0.0)
        action = mm.step(FakeBinding([cell(8, 15)], vision, torch.zeros(16, 16)), [cell(8, 15)])
        dx, dy = motor.cell_offsets()
        unit = np.array([dx[8, 15], dy[8, 15]]) / np.hypot(dx[8, 15], dy[8, 15])
        assert action == pytest.approx(0.5 * unit)
        assert mm.history[-1]["target"] == pytest.approx([dx[8, 15], dy[8, 15]])

    def test_no_target_gives_none_and_is_recorded(self):
        mm = motor.TectalMotorMap("group", seed=0)
        assert mm.step(FakeBinding([], torch.zeros(16, 16), torch.zeros(16, 16)), []) is None
        assert mm.history[-1]["target"] is None

    def test_the_noise_comes_from_its_own_seed(self):
        vision = torch.zeros(16, 16)
        vision[8, 15] = 1.0
        binding = FakeBinding([cell(8, 15)], vision, torch.zeros(16, 16))
        state = np.random.get_state()
        first = motor.TectalMotorMap("group", seed=3).step(binding, [cell(8, 15)])
        second = motor.TectalMotorMap("group", seed=3).step(binding, [cell(8, 15)])
        assert first == pytest.approx(second)
        assert np.array_equal(np.random.get_state()[1], state[1])


class TestWiring:
    @staticmethod
    def parser():
        from scripts.training import train_rlhf
        captured = {}

        def capture(self, *args, **kwargs):
            captured["parser"] = self
            raise RuntimeError("captured")

        with mock.patch.object(argparse.ArgumentParser, "parse_args", capture):
            with pytest.raises(RuntimeError):
                train_rlhf.main()
        return captured["parser"]

    def test_the_flag_is_off_by_default_and_its_help_names_the_rules(self):
        parser = self.parser()
        assert parser.parse_args([]).tectal_orienting == "off"
        for mode in ("group", "active", "audio"):
            assert parser.parse_args(["--tectal-orienting", mode]).tectal_orienting == mode
        text = next(a.help for a in parser._actions if "--tectal-orienting" in a.option_strings)
        assert "--map-binding" in text and "body" in text

    def test_the_wrapper_calls_the_policy_first_and_replaces_its_action_with_a_command(self):
        from models.core.sensory_tectum import SensoryTectum
        from models.core.global_workspace import GlobalWorkspace
        from scripts.training import train_rlhf
        tectum, workspace = SensoryTectum({}), GlobalWorkspace({"map_binding": True})
        train_rlhf._wire_map_binding(tectum, workspace, None, "trimmed")
        calls = []

        class Policy:
            def select_action(self, state, emotion_arousal=0.5, rpe_cache=0.0, self_vector=None):
                calls.append(state)
                return np.array([0.9, 0.9], dtype=np.float32), 0.25

        policy = Policy()
        train_rlhf._wire_tectal_orienting(policy, tectum, "group", seed=0, log_dir=None)
        frame = torch.zeros(1, 3, 224, 224)
        frame[0, :, 112 - 7:112 + 7, 200:224] = 1.0          # a bright patch right of the agent
        tectum(frame, torch.zeros(1, tectum.feature_dim, 2))
        action, value = policy.select_action("broadcast", emotion_arousal=0.3)
        assert calls == ["broadcast"] and value == 0.25
        record = tectum.tectal_motor_map.history[-1]
        if record["target"] is None:
            assert action == pytest.approx([0.9, 0.9])
        else:
            assert np.hypot(*action) < 1.0 and action.dtype == np.float32

    def test_orienting_needs_map_binding(self):
        from models.core.sensory_tectum import SensoryTectum
        from scripts.training import train_rlhf
        with pytest.raises(ValueError):
            train_rlhf._wire_tectal_orienting(object(), SensoryTectum({}), "group", seed=0, log_dir=None)
