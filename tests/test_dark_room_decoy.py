"""The second sound source of the dark room (`decoy_sound`), stage 1 of the binding design.

The decoy is a tone with no visual object. It gives steps where the two senses report
different places, which the binding gates need. The two digests below were recorded on
the code before this change (revision d8d059c), so they show that the default output is
unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulations.environments import audio_mixin
from simulations.environments.simple_visual_env import DECOY_MIN_GAP, LIGHT_COLOUR, SimpleVisualEnv

ACTION = np.array([0.3, -0.7], dtype=np.float32)
DIGEST_BEFORE = {(4, "agent_centered"): "7023c4a2a2bccfd98d42b60ecdaaa7ab",
                 (2, "full"): "7c398585df7b0e4d15c6aac564edb47e"}


def make_env(channels=4, view="agent_centered", decoy=False, seed=5):
    np.random.seed(seed)
    env = SimpleVisualEnv(width=224, height=224, audio="binaural", audio_channels=channels,
                          report_collision=True, view=view, decoy_sound=decoy)
    env.seed_audio(seed)
    if decoy:
        env.seed_decoy(seed)
    return env


def run(env, steps=6):
    digest = hashlib.md5()
    obs, info = env.reset()
    digest.update(obs.tobytes())
    infos = []
    for _ in range(steps):
        obs, _, _, _, info = env.step(ACTION)
        digest.update(obs.tobytes())
        digest.update(info["audio_waveform"].tobytes())
        infos.append(info)
    return digest.hexdigest(), infos


def peak_level(waveform, frequency, sample_rate):
    """Spectral level of the first ear at one frequency."""
    spectrum = np.abs(np.fft.rfft(waveform[0]))
    bins = np.fft.rfftfreq(waveform.shape[1], 1.0 / sample_rate)
    return float(spectrum[np.argmin(np.abs(bins - frequency))])


class TestDefaultIsUnchanged:
    @pytest.mark.parametrize("channels, view", [(4, "agent_centered"), (2, "full")])
    def test_without_the_decoy_the_output_is_the_output_before_the_change(self, channels, view):
        digest, infos = run(make_env(channels, view))
        assert digest == DIGEST_BEFORE[(channels, view)]
        assert all("_truth_decoy_offset" not in info for info in infos)


class TestDecoy:
    @staticmethod
    def reset_from(env, seed=5):
        # Both environments draw the agent and the light from numpy's global stream,
        # so each reset starts from the same state of that stream.
        np.random.seed(seed)
        return env.reset()

    def test_the_agent_and_the_light_are_where_they_are_without_the_decoy(self):
        plain, decoy = make_env(), make_env(decoy=True)
        self.reset_from(plain)
        self.reset_from(decoy)
        assert np.array_equal(plain.agent_pos, decoy.agent_pos)
        assert np.array_equal(plain.light_pos, decoy.light_pos)

    def test_the_frames_are_the_same_because_the_decoy_cannot_be_seen(self):
        plain, decoy = make_env(), make_env(decoy=True)
        assert np.array_equal(self.reset_from(plain)[0], self.reset_from(decoy)[0])
        assert np.array_equal(plain.step(ACTION)[0], decoy.step(ACTION)[0])

    def test_the_decoy_adds_its_own_tone_to_the_sound(self):
        _, plain = run(make_env())
        _, decoy = run(make_env(decoy=True))
        sample_rate = make_env()._audio_sample_rate
        wave_plain, wave_decoy = plain[-1]["audio_waveform"], decoy[-1]["audio_waveform"]
        assert peak_level(wave_decoy, audio_mixin.DECOY_TONE_HZ, sample_rate) > \
            5.0 * peak_level(wave_plain, audio_mixin.DECOY_TONE_HZ, sample_rate)
        assert decoy[-1]["_truth_decoy_offset"] is not None

    def test_the_offset_is_the_place_of_the_decoy_seen_from_the_agent(self):
        env = make_env(decoy=True)
        env.reset()
        _, _, _, _, info = env.step(ACTION)
        assert info["_truth_decoy_offset"] == pytest.approx(list(env.decoy_pos - env.agent_pos))
        assert info["_truth_decoy_level"] == env.decoy_level

    def test_the_decoy_place_comes_from_its_own_seed(self):
        first, second, other = make_env(decoy=True), make_env(decoy=True), make_env(decoy=True)
        other.seed_decoy(6)
        for env in (first, second, other):
            env.reset()
        assert np.array_equal(first.decoy_pos, second.decoy_pos)
        assert not np.array_equal(first.decoy_pos, other.decoy_pos)

    def test_the_level_scales_the_tone_of_the_decoy(self):
        sample_rate = make_env()._audio_sample_rate
        # Each environment is built just before its run, because make_env seeds the
        # global stream that places the agent and the light.
        _, quiet_infos = run(make_env(decoy=True))
        loud = make_env(decoy=True)
        loud.decoy_levels = (1.5,)
        _, loud_infos = run(loud)
        ratio = peak_level(loud_infos[-1]["audio_waveform"], audio_mixin.DECOY_TONE_HZ, sample_rate) / \
            peak_level(quiet_infos[-1]["audio_waveform"], audio_mixin.DECOY_TONE_HZ, sample_rate)
        assert 1.35 < ratio < 1.6

    def test_a_list_of_levels_is_used_in_turn_one_for_each_reset(self):
        env = make_env(decoy=True)
        env.decoy_levels = (0.5, 3.0)
        seen = []
        for _ in range(3):
            env.reset()
            seen.append(env.decoy_level)
        assert seen == [0.5, 3.0, 0.5]

    def test_the_training_script_reads_one_level_or_a_list(self):
        from scripts.training.train_rlhf import parse_levels
        assert parse_levels("1.5") == 1.5
        assert parse_levels("0.5,3.0") == (0.5, 3.0)

    def test_a_level_of_zero_or_less_is_refused(self):
        with pytest.raises(ValueError):
            SimpleVisualEnv(width=224, height=224, audio="binaural", audio_channels=4,
                            decoy_sound=True, decoy_level=0.0)

    def test_the_decoy_needs_the_binaural_sound(self):
        with pytest.raises(ValueError):
            SimpleVisualEnv(width=224, height=224, audio="legacy", decoy_sound=True)


class TestTrainingFlag:
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

    def test_the_flag_is_off_by_default_and_its_help_names_the_second_source(self):
        parser = self.parser()
        assert parser.parse_args([]).dark_room_decoy_sound is False
        text = next(a.help for a in parser._actions if "--dark-room-decoy-sound" in a.option_strings)
        assert "second" in text and "binaural" in text


def make_visible(seed=5, view="agent_centered"):
    np.random.seed(seed)
    env = SimpleVisualEnv(width=224, height=224, audio="binaural", audio_channels=4,
                          report_collision=True, view=view, decoy_sound=True, decoy_visible=True)
    env.seed_audio(seed)
    env.seed_decoy(seed)
    return env


def light_pixels(frame):
    return int(np.all(frame == np.array(LIGHT_COLOUR, dtype=frame.dtype), axis=-1).sum())


class TestVisibleDecoy:
    """The decoy as an object that looks like the light, gives no reward and has the second tone."""

    def test_a_visible_decoy_needs_the_decoy_sound(self):
        with pytest.raises(ValueError):
            SimpleVisualEnv(width=224, height=224, audio="binaural", audio_channels=4,
                            decoy_visible=True)

    def test_the_agent_the_light_and_the_decoy_tone_place_follow_their_own_streams(self):
        hidden, visible = make_env(decoy=True), make_visible()
        np.random.seed(5)
        hidden.reset()
        np.random.seed(5)
        visible.reset()
        assert np.array_equal(hidden.agent_pos, visible.agent_pos)
        assert np.array_equal(hidden.light_pos, visible.light_pos)

    def test_the_decoy_is_drawn_with_the_light_colour_and_size_at_its_place(self):
        env = make_visible(view="full")
        frame, _ = env.reset()
        x, y = env.decoy_pos.astype(int)
        assert tuple(frame[y, x]) == LIGHT_COLOUR
        # Two discs of the light radius, apart by more than two radii: twice the light alone.
        plain = make_env(view="full", decoy=True)
        np.random.seed(5)
        plain_frame, _ = plain.reset()
        np.random.seed(5)
        frame, _ = make_visible(view="full").reset()
        assert light_pixels(frame) > 1.8 * light_pixels(plain_frame)

    def test_the_decoy_keeps_its_gap_from_the_light_at_every_reset(self):
        env = make_visible()
        for _ in range(200):
            env.reset()
            gap = np.linalg.norm(env.decoy_pos - env.light_pos)
            assert DECOY_MIN_GAP <= gap <= env.decoy_max_gap

    def test_the_largest_gap_lets_the_agent_see_both_from_the_middle(self):
        env = make_visible()
        assert env.decoy_max_gap == 2 * (env.view_radius + env.light_radius)

    def test_the_decoy_stays_inside_the_room(self):
        env = make_visible()
        for _ in range(200):
            env.reset()
            assert np.all(env.decoy_pos >= 0) and np.all(env.decoy_pos <= [env.width, env.height])

    def test_the_decoy_takes_many_directions_from_the_light(self):
        env = make_visible()
        angles = []
        for _ in range(200):
            env.reset()
            dx, dy = env.decoy_pos - env.light_pos
            angles.append(np.arctan2(dy, dx))
        assert np.histogram(angles, bins=4, range=(-np.pi, np.pi))[0].min() > 0

    def test_the_gap_is_wide_enough_that_the_agent_cannot_touch_both(self):
        env = make_visible()
        assert DECOY_MIN_GAP == 2 * (env.light_radius + env.agent_radius)

    def test_touching_the_decoy_gives_no_reward(self):
        env = make_visible()
        env.reset()
        env.agent_pos = env.decoy_pos.astype(np.float32).copy()
        _, reward, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        assert reward == -0.01
        assert info["in_light"] is False

    def test_the_agent_centred_frame_shows_the_decoy_when_it_is_in_view(self):
        env = make_visible()
        env.reset()
        env.agent_pos = env.decoy_pos.astype(np.float32).copy()
        frame, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        assert info["_truth_decoy_in_view"] is True
        assert light_pixels(frame) > 0

    def test_the_in_view_record_uses_the_rule_of_the_light(self):
        env = make_visible()
        env.reset()
        _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        reach = env.view_radius + env.light_radius
        gap = np.abs(env.decoy_pos - env.agent_pos)
        assert info["_truth_decoy_in_view"] == bool(gap[0] <= reach and gap[1] <= reach)

    def test_the_hidden_decoy_has_no_in_view_record(self):
        _, infos = run(make_env(decoy=True))
        assert all("_truth_decoy_in_view" not in info for info in infos)

    def test_the_training_flag_is_off_by_default_and_its_help_names_the_rules(self):
        parser = TestTrainingFlag.parser()
        assert parser.parse_args([]).dark_room_decoy_visible is False
        text = next(a.help for a in parser._actions if "--dark-room-decoy-visible" in a.option_strings)
        assert "light colour" in text and "no reward" in text and "--dark-room-decoy-sound" in text


def make_between(seed=5):
    np.random.seed(seed)
    env = SimpleVisualEnv(width=224, height=224, audio="binaural", audio_channels=4,
                          report_collision=True, view="agent_centered", decoy_sound=True,
                          decoy_visible=True, start_between=True)
    env.seed_audio(seed)
    env.seed_decoy(seed)
    return env


class TestStartBetween:
    """The agent starts each episode midway between the light and the visible decoy."""

    def test_it_needs_the_visible_decoy(self):
        with pytest.raises(ValueError):
            SimpleVisualEnv(width=224, height=224, audio="binaural", audio_channels=4,
                            decoy_sound=True, start_between=True)

    def test_the_agent_starts_at_the_middle_of_the_two_objects(self):
        env = make_between()
        for _ in range(20):
            env.reset()
            assert env.agent_pos == pytest.approx((env.light_pos + env.decoy_pos) / 2.0)

    def test_both_objects_are_in_view_at_the_start(self):
        env = make_between()
        for _ in range(50):
            env.reset()
            _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
            assert info["_truth_light_in_view"] and info["_truth_decoy_in_view"]

    def test_the_light_and_the_decoy_are_where_they_are_without_it(self):
        # Both rooms draw the agent and the light from numpy's global stream, so each pair of
        # resets starts from the same state of that stream.
        plain, between = make_visible(), make_between()
        for k in range(5):
            np.random.seed(k)
            plain.reset()
            np.random.seed(k)
            between.reset()
            assert np.array_equal(plain.light_pos, between.light_pos)
            assert np.array_equal(plain.decoy_pos, between.decoy_pos)

    def test_the_training_flag_is_off_by_default_and_its_help_names_the_rule(self):
        parser = TestTrainingFlag.parser()
        assert parser.parse_args([]).dark_room_start_between is False
        text = next(a.help for a in parser._actions if "--dark-room-start-between" in a.option_strings)
        assert "midway" in text and "--dark-room-decoy-visible" in text


def make_pulse(seed=5, view="agent_centered", direction=True):
    np.random.seed(seed)
    env = SimpleVisualEnv(width=224, height=224, audio="binaural", audio_channels=4,
                          report_collision=True, view=view, decoy_sound=True,
                          decoy_visible=True, start_between=True, pulse=True,
                          sound_direction=direction)
    env.seed_audio(seed)
    env.seed_decoy(seed)
    return env


def gain(pair, t, mean, depth):
    frequency, phase = pair
    return mean + depth * np.sin(2 * np.pi * frequency * t + phase)


class TestPulse:
    """Each object pulses in time with its own tone (docs/decisions/2026_10_15_rhythm_binding.md)."""

    def test_the_pulse_needs_the_visible_decoy(self):
        with pytest.raises(ValueError):
            SimpleVisualEnv(width=224, height=224, audio="binaural", audio_channels=4,
                            decoy_sound=True, pulse=True)

    def test_the_two_rhythms_are_one_tenth_and_one_sixth_in_some_order(self):
        env = make_pulse()
        env.reset()
        assert sorted([env.pulse["light"][0], env.pulse["decoy"][0]]) == pytest.approx([1 / 10, 1 / 6])
        assert all(0.0 <= env.pulse[name][1] < 2 * np.pi for name in ("light", "decoy"))

    def test_the_rhythms_repeat_with_the_seed_and_leave_the_global_stream_alone(self):
        first, second = make_pulse(), make_pulse()
        np.random.seed(9)
        first.reset()
        state = np.random.get_state()[1].copy()
        np.random.seed(9)
        second.reset()
        assert first.pulse == second.pulse
        assert np.array_equal(np.random.get_state()[1], state)

    def test_the_light_and_the_decoy_keep_their_places(self):
        plain, pulsed = make_between(), make_pulse()
        np.random.seed(3)
        plain.reset()
        np.random.seed(3)
        pulsed.reset()
        assert np.array_equal(plain.light_pos, pulsed.light_pos)
        assert np.array_equal(plain.decoy_pos, pulsed.decoy_pos)

    def test_the_colour_follows_the_rhythm_at_reset_and_after_a_step(self):
        env = make_pulse(view="full")
        frame, _ = env.reset()
        for t in (0, 1):
            x, y = env.light_pos.astype(int)
            g = gain(env.pulse["light"], t, 0.85, 0.15)
            assert tuple(frame[y, x]) == tuple(int(round(c * g)) for c in LIGHT_COLOUR)
            x, y = env.decoy_pos.astype(int)
            g = gain(env.pulse["decoy"], t, 0.85, 0.15)
            assert tuple(frame[y, x]) == tuple(int(round(c * g)) for c in LIGHT_COLOUR)
            frame, _, _, _, _ = env.step(np.zeros(2, dtype=np.float32))

    def test_the_brightest_dim_object_stays_above_the_agent_ring(self):
        dim = 0.85 - 0.15
        luminance = (0.299 * LIGHT_COLOUR[0] + 0.587 * LIGHT_COLOUR[1] + 0.114 * LIGHT_COLOUR[2]) / 255
        assert luminance * dim > (0.299 * 217 + 0.587 * 119 + 0.114 * 87) / 255

    def test_the_tone_follows_the_rhythm(self):
        env = make_pulse()
        env.reset()
        sample_rate = env._audio_sample_rate
        levels = []
        for t in (1, 2, 3, 4, 5, 6):
            _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
            levels.append((peak_level(info["audio_waveform"], audio_mixin.BINAURAL_TONE_HZ, sample_rate),
                           gain(env.pulse["light"], t, 0.7, 0.3)))
        measured, expected = np.array(levels).T
        assert np.corrcoef(measured, expected)[0, 1] > 0.99
        assert measured / measured[0] == pytest.approx(expected / expected[0], rel=0.05)

    def test_the_truth_record_names_the_rhythms(self):
        env = make_pulse()
        env.reset()
        _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        assert info["_truth_pulse_light"] == pytest.approx(list(env.pulse["light"]))
        assert info["_truth_pulse_decoy"] == pytest.approx(list(env.pulse["decoy"]))
        assert info["_truth_pulse_step"] == 1

    def test_without_the_pulse_there_is_no_record(self):
        env = make_between()
        env.reset()
        _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        assert "_truth_pulse_light" not in info


class TestSoundDirection:
    def test_without_direction_every_ear_hears_the_same_tone(self):
        env = make_pulse(direction=False)
        env.reset()
        _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        wave = info["audio_waveform"]
        sample_rate = env._audio_sample_rate
        tone = peak_level(wave, audio_mixin.BINAURAL_TONE_HZ, sample_rate)
        for ear in range(1, 4):
            difference = peak_level(wave[[0]] - wave[[ear]], audio_mixin.BINAURAL_TONE_HZ, sample_rate)
            assert difference < 0.05 * tone

    def test_with_direction_the_ears_differ(self):
        env = make_pulse(direction=True)
        env.reset()
        env.agent_pos = (env.light_pos + np.array([-30.0, 0.0])).astype(np.float32)
        _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        wave = info["audio_waveform"]
        sample_rate = env._audio_sample_rate
        assert peak_level(wave[[0]] - wave[[1]], audio_mixin.BINAURAL_TONE_HZ, sample_rate) > \
            0.1 * peak_level(wave, audio_mixin.BINAURAL_TONE_HZ, sample_rate)

    def test_the_training_flags_are_off_by_default(self):
        parser = TestTrainingFlag.parser()
        args = parser.parse_args([])
        assert args.dark_room_pulse is False and args.dark_room_sound_direction == "on"
        text = next(a.help for a in parser._actions if "--dark-room-pulse" in a.option_strings)
        assert "--dark-room-decoy-visible" in text and "rhythm" in text
