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
from simulations.environments.simple_visual_env import SimpleVisualEnv

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
