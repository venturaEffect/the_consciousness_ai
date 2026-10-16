"""The selection switch that learns the value of each heard tone (models/core/value_gate.py).

`docs/decisions/2026_10_16_value_gate.md`. Expected values are closed forms computed in each test.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.core import value_gate as vg


class TestChoice:
    def test_the_probabilities_are_the_softmax_of_salience_plus_twice_the_value(self):
        gate = vg.ValueGate(seed=0)
        gate.values = {18: 0.5, 23: 0.0}
        p = gate.probabilities({18: 0.4, 23: 1.0})
        a, b = (0.4 + 2 * 0.5) / 0.25, (1.0 + 2 * 0.0) / 0.25
        assert p[18] == pytest.approx(np.exp(a) / (np.exp(a) + np.exp(b)))
        assert p[23] == pytest.approx(np.exp(b) / (np.exp(a) + np.exp(b)))

    def test_one_candidate_is_chosen_with_certainty(self):
        gate = vg.ValueGate(seed=0)
        assert gate.probabilities({23: 0.7}) == {23: 1.0}

    def test_the_choice_repeats_with_the_seed_and_leaves_the_global_stream_alone(self):
        np.random.seed(4)
        state = np.random.get_state()[1].copy()
        first = [vg.ValueGate(seed=7).choose({18: 1.0, 23: 0.9}) for _ in range(1)]
        second = [vg.ValueGate(seed=7).choose({18: 1.0, 23: 0.9}) for _ in range(1)]
        assert first == second
        assert np.array_equal(np.random.get_state()[1], state)

    def test_the_choice_is_kept_for_the_episode(self):
        gate = vg.ValueGate(seed=0)
        band = gate.choose({18: 1.0, 23: 0.4})
        assert gate.choice == band
        assert gate.choose({18: 0.1, 23: 1.0}) == band     # a second call does not choose again


class TestLearning:
    def test_the_value_after_n_fed_episodes_is_one_minus_seven_tenths_to_the_n(self):
        gate = vg.ValueGate(seed=0)
        for _ in range(5):
            gate.choose({18: 1.0})
            gate.observe_reward(1.0)
            gate.new_episode()
        assert gate.values[18] == pytest.approx(1 - 0.7 ** 5)

    def test_food_updates_at_once_and_only_once(self):
        gate = vg.ValueGate(seed=0)
        gate.choose({18: 1.0})
        gate.observe_reward(1.0)
        gate.observe_reward(1.0)
        assert gate.values[18] == pytest.approx(0.3)
        assert len(gate.episodes) == 1 and gate.episodes[0]["outcome"] == 1.0
        assert gate.episodes[0]["error"] == pytest.approx(1.0)

    def test_no_food_within_sixty_steps_gives_zero(self):
        gate = vg.ValueGate(seed=0)
        gate.values = {23: 0.5}
        gate.choose({23: 1.0})
        for _ in range(59):
            gate.observe_reward(-0.01)
        assert len(gate.episodes) == 0
        gate.observe_reward(-0.01)
        assert gate.values[23] == pytest.approx(0.5 + 0.3 * (0.0 - 0.5))
        gate.observe_reward(1.0)                               # food after the window does not count
        assert gate.values[23] == pytest.approx(0.35)

    def test_a_new_episode_closes_an_open_choice_with_zero(self):
        gate = vg.ValueGate(seed=0)
        gate.values = {18: 0.4}
        gate.choose({18: 1.0})
        gate.observe_reward(-0.01)
        gate.new_episode()
        assert gate.values[18] == pytest.approx(0.4 * 0.7)
        assert gate.choice is None

    def test_reward_before_any_choice_is_ignored(self):
        gate = vg.ValueGate(seed=0)
        gate.observe_reward(1.0)
        gate.new_episode()
        assert gate.values == {} and gate.episodes == []

    def test_a_learning_rate_of_zero_keeps_the_values(self):
        gate = vg.ValueGate(seed=0, learning_rate=0.0)
        gate.choose({18: 1.0})
        gate.observe_reward(1.0)
        assert gate.values.get(18, 0.0) == 0.0
        assert gate.episodes[0]["error"] == pytest.approx(1.0)

    def test_the_record_holds_the_choice_and_the_values(self):
        gate = vg.ValueGate(seed=0)
        gate.choose({18: 1.0, 23: 0.4})
        gate.observe_reward(1.0)
        entry = gate.episodes[0]
        assert set(entry) >= {"choice", "priorities", "value_before", "value_after", "outcome", "error"}


class TestWiring:
    @staticmethod
    def parser():
        import argparse
        from unittest import mock
        from scripts.training import train_rlhf
        captured = {}

        def capture(self, *args, **kwargs):
            captured["parser"] = self
            raise RuntimeError("captured")

        with mock.patch.object(argparse.ArgumentParser, "parse_args", capture):
            with pytest.raises(RuntimeError):
                train_rlhf.main()
        return captured["parser"]

    def test_the_flags(self):
        parser = self.parser()
        assert parser.parse_args(["--tectal-orienting", "value"]).tectal_orienting == "value"
        assert parser.parse_args([]).value_learning_rate == pytest.approx(0.3)
        text = next(a.help for a in parser._actions if "--value-learning-rate" in a.option_strings)
        assert "frozen" in text

    @staticmethod
    def wired(learning_rate=0.3):
        import torch
        from models.core.global_workspace import GlobalWorkspace
        from models.core.sensory_tectum import SensoryTectum
        from scripts.training import train_rlhf
        from tests.test_rhythm_binding import FakeSpecialist
        tectum, workspace = SensoryTectum({}), GlobalWorkspace({"map_binding": True})
        spec = FakeSpecialist()
        train_rlhf._wire_map_binding(tectum, workspace, None, "rhythm", auditory_specialist=spec, seed=0,
                                     bands_to_bind=2)
        calls = []

        class Policy:
            def select_action(self, state, emotion_arousal=0.5, rpe_cache=0.0, self_vector=None):
                calls.append("select")
                return np.array([0.6, -0.4], dtype=np.float32), 0.1

            def step(self, **kwargs):
                calls.append(("step", kwargs["raw_reward"]))
                return {}

        policy = Policy()
        train_rlhf._wire_value_gate(policy, tectum, seed=0, log_dir=None, idle="still",
                                    learning_rate=learning_rate)
        return tectum, spec, policy, calls, torch

    def test_the_gate_waits_then_chooses_one_band_and_follows_its_object(self):
        from tests.test_rhythm_binding import BLOCK_A, BLOCK_B, pulse_bands, pulse_frames
        tectum, spec, policy, calls, torch = self.wired()
        actions = []
        for t in range(20):
            if t > 0:
                spec.hear(pulse_bands(t))
            tectum(pulse_frames(t), torch.zeros(1, tectum.feature_dim, 2))
            action, _ = policy.select_action("broadcast")
            policy.step(raw_reward=-0.01)
            actions.append(action)
        gate = tectum.value_gate
        assert all(np.allclose(a, 0.0) for a in actions[:12])        # no choice before the wait ends
        assert gate.choice in (18, 23)
        target = tectum.tectal_motor_map.history[-1]["target"]
        assert target is not None
        block = BLOCK_A if gate.choice == 18 else BLOCK_B
        from models.core.tectal_motor_map import cell_offsets
        dx, dy = cell_offsets()
        centre = np.mean([[dx.flat[i], dy.flat[i]] for i in block], axis=0)
        assert np.dot(target, centre) > 0                              # it goes to the chosen object
        assert calls.count("select") == 20 and len([c for c in calls if c != "select"]) == 20

    def test_a_step_without_sound_closes_the_episode(self):
        from tests.test_rhythm_binding import pulse_bands, pulse_frames
        tectum, spec, policy, calls, torch = self.wired()
        for t in range(16):
            if t > 0:
                spec.hear(pulse_bands(t))
            tectum(pulse_frames(t), torch.zeros(1, tectum.feature_dim, 2))
            policy.select_action("broadcast")
            policy.step(raw_reward=-0.01)
        assert tectum.value_gate.choice is not None
        tectum(pulse_frames(0), torch.zeros(1, tectum.feature_dim, 2))   # no new sound: next episode
        policy.select_action("broadcast")
        assert tectum.value_gate.choice is None
        assert len(tectum.value_gate.episodes) == 1 and tectum.value_gate.episodes[0]["outcome"] == 0.0

    def test_food_reaches_the_gate(self):
        from tests.test_rhythm_binding import pulse_bands, pulse_frames
        tectum, spec, policy, calls, torch = self.wired()
        for t in range(14):
            if t > 0:
                spec.hear(pulse_bands(t))
            tectum(pulse_frames(t), torch.zeros(1, tectum.feature_dim, 2))
            policy.select_action("broadcast")
            policy.step(raw_reward=1.0 if t == 13 else -0.01)
        gate = tectum.value_gate
        assert gate.episodes[0]["outcome"] == 1.0 and gate.values[gate.choice] == pytest.approx(0.3)

    def test_the_value_gate_needs_two_bound_bands(self):
        from models.core.global_workspace import GlobalWorkspace
        from models.core.sensory_tectum import SensoryTectum
        from scripts.training import train_rlhf
        from tests.test_rhythm_binding import FakeSpecialist
        tectum, workspace = SensoryTectum({}), GlobalWorkspace({"map_binding": True})
        train_rlhf._wire_map_binding(tectum, workspace, None, "rhythm", auditory_specialist=FakeSpecialist(), seed=0)
        with pytest.raises(ValueError):
            train_rlhf._wire_value_gate(object(), tectum, seed=0, log_dir=None)
