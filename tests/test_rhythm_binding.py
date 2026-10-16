"""Binding by timing with oscillators (models/core/rhythm_binding.py).

`docs/decisions/2026_10_15_rhythm_binding.md`. Expected values are closed forms of the leaky
recursions, computed independently in each test.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.core import rhythm_binding as rb


class TestPeakLuminance:
    def test_each_cell_reads_its_brightest_pixel(self):
        frames = torch.zeros(1, 3, 224, 224)
        frames[0, :, 3, 5] = 1.0                    # one white pixel in cell (0, 0)
        frames[0, 0, 20:28, 30:40] = 1.0            # a red patch in cell (1, 2)
        peak = rb.cell_peak_luminance(frames)
        assert peak.shape == (16, 16)
        assert peak[0, 0] == pytest.approx(1.0)
        assert peak[1, 2] == pytest.approx(0.299)
        assert peak[5, 5] == 0.0


class TestOscillatorBank:
    def test_the_recursion_equals_its_closed_form(self):
        a, f = 1 / 6, np.array([0.1, 0.25])
        bank = rb.OscillatorBank(1, frequencies=f, rate=a, mean_rate=None)
        x = np.random.default_rng(0).standard_normal(25)
        for value in x:
            z = bank.step(np.array([value]))
        t = np.arange(1, 26)
        expected = [a * np.sum((1 - a) ** (25 - t) * x * np.exp(-2j * np.pi * fk * t)) for fk in f]
        assert z[0] == pytest.approx(np.array(expected))

    def test_a_cosine_at_the_oscillator_frequency_gives_half_its_amplitude(self):
        a, fk, amplitude, phase = 1 / 6, 0.1, 0.8, 0.7
        bank = rb.OscillatorBank(1, frequencies=np.array([fk]), rate=a, mean_rate=None)
        for t in range(1, 201):
            z = bank.step(np.array([amplitude * np.cos(2 * np.pi * fk * t + phase)]))
        # z = (A/2) e^(i phase) (1 - (1-a)^t) plus a term at 2f of size at most
        # (A/2) a / |1 - (1-a) e^(-i 4 pi f)|.
        bound = amplitude / 2 * a / abs(1 - (1 - a) * np.exp(-4j * np.pi * fk)) + amplitude / 2 * (1 - a) ** 200
        assert abs(z[0, 0] - amplitude / 2 * np.exp(1j * phase)) <= bound + 1e-12

    def test_the_slow_mean_starts_at_the_first_input(self):
        bank = rb.OscillatorBank(1, frequencies=np.array([0.1]), rate=1 / 6, mean_rate=0.1)
        z = bank.step(np.array([5.0]))
        assert z[0, 0] == pytest.approx(0.0)

    def test_reset_clears_the_state(self):
        bank = rb.OscillatorBank(1, frequencies=np.array([0.1]), rate=1 / 6, mean_rate=None)
        bank.step(np.array([1.0]))
        bank.reset()
        assert bank.t == 0 and np.all(bank.z == 0)


class TestCoherence:
    """Amplitude-weighted locking: C = |<z_c conj(z_r)>| / sqrt(<|z_c|^2> <|z_r|^2>), leaky means at rate b.
    A cell without input has no phase, so its C is 0, not a constant phase that looks locked."""

    def test_identical_oscillators_give_one(self):
        lock = rb.Coherence(2, rate=1 / 8)
        for t in range(1, 11):
            z = np.array([0.5 * np.exp(0.3j * t), 0.2 * np.exp(-1j * t)])
            c = lock.step(z, z)
        assert c == pytest.approx(np.ones(2))

    def test_a_constant_phase_step_gives_the_closed_form(self):
        b, d, t_end = 1 / 8, 0.9, 20
        lock = rb.Coherence(1, rate=b)
        for t in range(1, t_end + 1):
            c = lock.step(np.array([np.exp(1j * d * t)]), 1.0 + 0j)
        q = (1 - b) * np.exp(-1j * d)
        mean = b * np.exp(1j * d * t_end) * (1 - q ** t_end) / (1 - q)
        assert c[0] == pytest.approx(abs(mean) / (1 - (1 - b) ** t_end))

    def test_a_cell_without_input_gives_zero(self):
        lock = rb.Coherence(1, rate=1 / 8)
        for t in range(1, 11):
            c = lock.step(np.array([0j]), np.exp(0.1j * t))
        assert c[0] == 0.0


class TestLoudestBand:
    def test_the_slow_mean_decides_not_the_moment(self):
        binder = rb.RhythmBinder(grid=2, bands=2)
        for value in ([1.0, 0.2],) * 30:
            binder.track_bands(np.array(value))
        binder.track_bands(np.array([0.1, 0.9]))      # one moment with the order swapped
        assert binder.loudest_band() == 0


def blobs_frame(t, f_a=0.1, f_b=1 / 6):
    """A 2 x 2 map: cells 0 and 1 pulse at f_a, cell 2 at f_b, cell 3 is dark."""
    a = 0.85 + 0.15 * np.sin(2 * np.pi * f_a * t)
    b = 0.85 + 0.15 * np.sin(2 * np.pi * f_b * t + 1.0)
    return np.array([[a, a], [b, 0.0]])


def bands(t, f_a=0.1, f_b=1 / 6):
    """Band 0 is the louder tone and follows f_a; band 1 is quieter and follows f_b."""
    return np.array([1.0 * (0.7 + 0.3 * np.sin(2 * np.pi * f_a * t)),
                     0.5 * (0.7 + 0.3 * np.sin(2 * np.pi * f_b * t + 1.0))])


class TestRhythmBinder:
    def test_the_cells_with_the_louder_tones_rhythm_lock(self):
        binder = rb.RhythmBinder(grid=2, bands=2)
        for t in range(1, 41):
            plv = binder.step(blobs_frame(t), bands(t))
        assert plv[0, 0] > 0.9 and plv[0, 1] > 0.9
        assert plv[1, 0] < 0.5
        assert plv[1, 1] == 0.0                     # the dark cell has no rhythm
        assert binder.loudest_band() == 0

    def test_a_step_without_sound_restarts_the_oscillators(self):
        binder = rb.RhythmBinder(grid=2, bands=2)
        for t in range(1, 21):
            binder.step(blobs_frame(t), bands(t))
        plv = binder.step(blobs_frame(21), None)
        assert np.all(plv == 0) and binder.bank_cells.t == 0

    def test_scrambling_the_sound_breaks_the_lock(self):
        plain, scrambled = rb.RhythmBinder(grid=2, bands=2), rb.RhythmBinder(grid=2, bands=2, scramble_seed=4)
        for t in range(1, 41):
            locked = plain.step(blobs_frame(t), bands(t))
            broken = scrambled.step(blobs_frame(t), bands(t))
        assert broken[0, 0] < locked[0, 0] - 0.3

    def test_the_group_is_the_cells_at_or_above_the_threshold(self):
        plv = np.array([[0.9, 0.6], [0.59, 0.2]])
        assert rb.bound_cells(plv, threshold=0.6) == [0, 1]


def pulse_frames(t):
    """[1, 3, 224, 224] frames: block A (rows 2-4, cols 10-13) and the centre block (rows 7-8,
    cols 7-8, where the agent's body is) follow rhythm A; block B (rows 11-13, cols 2-5) follows rhythm B."""
    a = 0.85 + 0.15 * np.sin(2 * np.pi * 0.1 * t)
    b = 0.85 + 0.15 * np.sin(2 * np.pi * (1 / 6) * t + 1.0)
    frames = torch.zeros(1, 3, 224, 224)
    frames[0, :, 2 * 14:5 * 14, 10 * 14:14 * 14] = a
    frames[0, :, 7 * 14:9 * 14, 7 * 14:9 * 14] = a
    frames[0, :, 11 * 14:14 * 14, 2 * 14:6 * 14] = b
    return frames


def pulse_bands(t):
    """64 bands: band 18 follows rhythm A at full level, band 23 follows rhythm B at half level."""
    envelope = np.full(64, 0.01)
    envelope[18] = 1.0 * (0.7 + 0.3 * np.sin(2 * np.pi * 0.1 * t))
    envelope[23] = 0.5 * (0.7 + 0.3 * np.sin(2 * np.pi * (1 / 6) * t + 1.0))
    return envelope


BLOCK_A = {r * 16 + c for r in range(2, 5) for c in range(10, 14)}
BLOCK_B = {r * 16 + c for r in range(11, 14) for c in range(2, 6)}
CENTRE = {r * 16 + c for r in range(7, 9) for c in range(7, 9)}


class FakeSpecialist:
    def __init__(self):
        self.last_band_envelope, self.band_envelope_count = None, 0

    def hear(self, envelope):
        self.last_band_envelope = envelope
        self.band_envelope_count += 1


class TestFreshEnvelope:
    def test_a_new_count_gives_the_envelope_and_the_same_count_gives_none(self):
        spec = FakeSpecialist()
        source = rb.fresh_band_envelope(spec)
        assert source() is None
        spec.hear(np.ones(64))
        assert source() == pytest.approx(np.ones(64))
        assert source() is None                    # no sound came in since


class TestRhythmMapBinding:
    @staticmethod
    def run(binding, spec, steps=40):
        factors, record = None, None
        for t in range(steps + 1):
            if t > 0:
                spec.hear(pulse_bands(t))
            factors, record = binding.observe(torch.zeros(1, 64, 16, 16), torch.zeros(1, 64, 2), pulse_frames(t))
        return factors, record

    def test_the_group_is_the_place_that_pulses_with_the_louder_tone_without_the_body(self):
        spec = FakeSpecialist()
        binding = rb.RhythmMapBinding(rb.fresh_band_envelope(spec))
        factors, record = self.run(binding, spec)
        group = set(record["group"])
        assert group and group <= BLOCK_A and len(group) >= 0.75 * len(BLOCK_A)
        assert not group & CENTRE and not group & BLOCK_B
        assert record["band"] == 18 and len(record["coherence"]) == 256
        top = max(record["coherence"][i] for i in group)
        assert factors["vision"] == factors["audio"] == pytest.approx(1.0 + 0.5 * top, abs=1e-3)

    def test_the_drives_for_the_motor_map_are_the_peak_luminance_and_no_place_for_sound(self):
        spec = FakeSpecialist()
        binding = rb.RhythmMapBinding(rb.fresh_band_envelope(spec))
        self.run(binding, spec, steps=3)
        expected = rb.cell_peak_luminance(pulse_frames(3))
        assert binding.last_vision.numpy() == pytest.approx(expected)
        assert float(binding.last_audio.abs().sum()) == 0.0

    def test_no_group_before_twelve_steps_of_sound(self):
        # With one or two samples the coherence of any two signals is 1, so the binding waits.
        spec = FakeSpecialist()
        binding = rb.RhythmMapBinding(rb.fresh_band_envelope(spec))
        groups = []
        for t in range(14):
            if t > 0:
                spec.hear(pulse_bands(t))
            _, record = binding.observe(torch.zeros(1, 64, 16, 16), torch.zeros(1, 64, 2), pulse_frames(t))
            groups.append(record["group"])
        assert all(g == [] for g in groups[:12])        # steps 0 to 11: at most 11 steps of sound
        assert groups[12] and groups[13]                # step 12: 12 steps of sound

    def test_a_step_without_sound_gives_no_group_and_neutral_factors(self):
        spec = FakeSpecialist()
        binding = rb.RhythmMapBinding(rb.fresh_band_envelope(spec))
        factors, record = binding.observe(torch.zeros(1, 64, 16, 16), torch.zeros(1, 64, 2), pulse_frames(0))
        assert record["group"] == [] and factors == {"vision": 1.0, "audio": 1.0}

    def test_observe_needs_the_image(self):
        binding = rb.RhythmMapBinding(rb.fresh_band_envelope(FakeSpecialist()))
        with pytest.raises(ValueError):
            binding.observe(torch.zeros(1, 64, 16, 16), torch.zeros(1, 64, 2))

    def test_the_scrambled_form_loses_the_place(self):
        spec_a, spec_b = FakeSpecialist(), FakeSpecialist()
        plain = rb.RhythmMapBinding(rb.fresh_band_envelope(spec_a))
        scrambled = rb.RhythmMapBinding(rb.fresh_band_envelope(spec_b), scramble_seed=2)
        _, kept = self.run(plain, spec_a)
        _, lost = self.run(scrambled, spec_b)

        def on_a(record):
            return np.mean([record["coherence"][i] for i in BLOCK_A])

        assert on_a(lost) < on_a(kept) - 0.3


class TestTrainingWiring:
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
        assert parser.parse_args(["--map-binding", "rhythm"]).map_binding == "rhythm"
        assert parser.parse_args(["--map-binding", "rhythm-scrambled"]).map_binding == "rhythm-scrambled"
        assert parser.parse_args([]).tectal_orienting_idle == "policy"
        assert parser.parse_args(["--tectal-orienting-idle", "still"]).tectal_orienting_idle == "still"

    def test_the_wiring_builds_both_forms_and_needs_the_cochlea(self):
        from models.core.global_workspace import GlobalWorkspace
        from models.core.sensory_tectum import SensoryTectum
        from scripts.training import train_rlhf
        for mode, scrambled in (("rhythm", False), ("rhythm-scrambled", True)):
            tectum, workspace = SensoryTectum({}), GlobalWorkspace({"map_binding": True})
            train_rlhf._wire_map_binding(tectum, workspace, None, mode, auditory_specialist=FakeSpecialist(), seed=3)
            assert isinstance(tectum.map_binding, rb.RhythmMapBinding) and tectum.map_binding.needs_frames
            assert (tectum.map_binding.binder.scramble is not None) == scrambled
            assert workspace.map_factor_source == tectum.map_factors
        with pytest.raises(ValueError):
            train_rlhf._wire_map_binding(SensoryTectum({}), GlobalWorkspace({"map_binding": True}), None, "rhythm")

    def test_still_gives_no_movement_without_a_target(self):
        from models.core.global_workspace import GlobalWorkspace
        from models.core.sensory_tectum import SensoryTectum
        from scripts.training import train_rlhf
        tectum, workspace = SensoryTectum({}), GlobalWorkspace({"map_binding": True})
        train_rlhf._wire_map_binding(tectum, workspace, None, "rhythm", auditory_specialist=FakeSpecialist(), seed=0)

        class Policy:
            def select_action(self, state, emotion_arousal=0.5, rpe_cache=0.0, self_vector=None):
                return np.array([0.6, -0.4], dtype=np.float32), 0.1

        policy = Policy()
        train_rlhf._wire_tectal_orienting(policy, tectum, "group", seed=0, log_dir=None, idle="still")
        tectum(pulse_frames(0), torch.zeros(1, tectum.feature_dim, 2))
        action, value = policy.select_action("broadcast")
        assert tectum.tectal_motor_map.history[-1]["target"] is None
        assert action == pytest.approx([0.0, 0.0]) and action.dtype == np.float32 and value == 0.1

    def test_the_tectum_forward_records_the_coherence(self):
        from models.core.global_workspace import GlobalWorkspace
        from models.core.sensory_tectum import SensoryTectum
        from scripts.training import train_rlhf
        tectum, workspace = SensoryTectum({}), GlobalWorkspace({"map_binding": True})
        spec = FakeSpecialist()
        train_rlhf._wire_map_binding(tectum, workspace, None, "rhythm", auditory_specialist=spec, seed=0)
        for t in range(3):
            if t:
                spec.hear(pulse_bands(t))
            tectum(pulse_frames(t), torch.zeros(1, tectum.feature_dim, 2))
        entry = tectum.map_history[-1]
        assert len(entry["coherence"]) == 256 and entry["band"] == 18


class TestTwoBands:
    """Binding of the 2 loudest bands, each to its own object (docs/decisions/2026_10_16_value_gate.md)."""

    @staticmethod
    def run(binding, spec, steps=40):
        record = None
        for t in range(steps + 1):
            if t > 0:
                spec.hear(pulse_bands(t))
            _, record = binding.observe(torch.zeros(1, 64, 16, 16), torch.zeros(1, 64, 2), pulse_frames(t))
        return record

    def test_each_of_the_two_loudest_bands_gets_its_own_object(self):
        spec = FakeSpecialist()
        binding = rb.RhythmMapBinding(rb.fresh_band_envelope(spec), bands_to_bind=2)
        record = self.run(binding, spec)
        groups = {int(band): set(cells) for band, cells in record["groups"].items()}
        assert set(groups) == {18, 23}
        assert groups[18] and groups[18] <= BLOCK_A and not groups[18] & CENTRE
        assert groups[23] and groups[23] <= BLOCK_B
        assert set(record["group"]) == groups[18]           # "group" stays the loudest band's group

    def test_the_salience_of_each_band_is_its_slow_mean_over_the_largest(self):
        spec = FakeSpecialist()
        binding = rb.RhythmMapBinding(rb.fresh_band_envelope(spec), bands_to_bind=2)
        record = self.run(binding, spec)
        mean = binding.binder.band_mean
        assert record["salience"]["18"] == pytest.approx(1.0)
        assert record["salience"]["23"] == pytest.approx(mean[23] / mean[18])

    def test_one_band_per_tone_even_when_a_neighbour_band_is_louder_than_the_other_tone(self):
        # The cochlea spreads a loud tone into its neighbour bands (measured: band 22 at 0.70 of
        # band 23 when the decoy is 3 times louder). The two candidates are the two largest local
        # peaks of the band profile, one for each tone, not the two loudest bands.
        profile = np.full(64, 0.01)
        profile[22], profile[23], profile[24] = 0.48, 0.68, 0.41
        profile[17], profile[18], profile[19] = 0.20, 0.33, 0.15
        assert rb.tone_bands(profile, 2) == [23, 18]

    def test_a_profile_with_one_peak_gives_one_band(self):
        profile = np.full(64, 0.01)
        profile[18] = 0.5
        assert rb.tone_bands(profile, 2) == [18]

    def test_one_band_keeps_the_record_of_the_gate(self):
        spec = FakeSpecialist()
        binding = rb.RhythmMapBinding(rb.fresh_band_envelope(spec))
        record = self.run(binding, spec)
        assert "groups" not in record and "salience" not in record

    def test_one_band_gives_the_same_coherence_as_before(self):
        spec_a, spec_b = FakeSpecialist(), FakeSpecialist()
        one = rb.RhythmMapBinding(rb.fresh_band_envelope(spec_a))
        two = rb.RhythmMapBinding(rb.fresh_band_envelope(spec_b), bands_to_bind=2)
        a, b = self.run(one, spec_a), self.run(two, spec_b)
        # The loudest band never changes here, so its own accumulator equals the single one.
        assert a["coherence"] == b["coherence"]
