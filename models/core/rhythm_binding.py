"""Binding by timing: oscillators driven by the rhythm of their input lock when the rhythms match.

`docs/decisions/2026_10_15_rhythm_binding.md`. Each place on the tectum map and each band of the
cochlea drives its own bank of damped oscillators, one for each rhythm frequency. When a seen
object changes in time with a heard tone, the oscillators of its places and the oscillator of the
tone's band keep a fixed phase relation: they are synchronised, and that synchrony is the binding.
After the temporal coherence account of audiovisual binding (Atilgan and colleagues, Neuron 2018)
and the entrainment of neural oscillations to stimulus rhythms (Lakatos and colleagues, Science
2008). Frequencies are in cycles per step, with no Hz reading.

All values were set on 2026-10-15 before any run, from an offline check on synthetic rhythms.
"""
from __future__ import annotations

from collections import deque
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

RHYTHM_FREQUENCIES = np.geomspace(1.0 / 16.0, 1.0 / 4.0, 8)
DRIVE_RATE = 1.0 / 6.0        # a: each oscillator's response to its input, per step
MEAN_RATE = 0.1               # the slow mean removed from each input
LOCK_RATE = 1.0 / 8.0         # b: the leaky means of the coherence
BAND_MEAN_RATE = 0.1          # the slow mean of each band that picks the loudest band
SCRAMBLE_HISTORY = 30         # steps the disruption draws from
_LUMINANCE = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)


def cell_peak_luminance(frames: torch.Tensor, grid: int = 16) -> np.ndarray:
    """[grid, grid]: the largest luminance in each map cell. `frames` [1, 3, h, w], values 0 to 1.

    An object reads the same wherever it covers part of a cell, so motion changes which cells
    see the object, not the value they read."""
    luminance = (frames.detach().float().cpu() * _LUMINANCE).sum(dim=1, keepdim=True)
    return F.adaptive_max_pool2d(luminance, (grid, grid))[0, 0].numpy().astype(np.float64)


class OscillatorBank:
    """Damped driven oscillators: z_k(t) = (1 - a) z_k(t-1) + a (x(t) - m(t)) e^(-i 2 pi f_k t).

    One row for each channel, one column for each frequency. m is a slow mean of the input that
    starts at the first input (`mean_rate` None removes nothing)."""

    def __init__(self, channels: int, frequencies=RHYTHM_FREQUENCIES, rate: float = DRIVE_RATE,
                 mean_rate: Optional[float] = MEAN_RATE):
        self.frequencies = np.asarray(frequencies, dtype=np.float64)
        self.rate, self.mean_rate, self.channels = rate, mean_rate, channels
        self.reset()

    def reset(self) -> None:
        self.z = np.zeros((self.channels, len(self.frequencies)), dtype=np.complex128)
        self.mean = None
        self.t = 0

    def step(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(self.channels)
        self.t += 1
        if self.mean_rate is None:
            centred = x
        else:
            self.mean = x.copy() if self.mean is None else self.mean + self.mean_rate * (x - self.mean)
            centred = x - self.mean
        rotation = np.exp(-2j * np.pi * self.frequencies * self.t)
        self.z = (1.0 - self.rate) * self.z + self.rate * centred[:, None] * rotation[None, :]
        return self.z


class Coherence:
    """C = |<z_c conj(z_r)>| / sqrt(<|z_c|^2> <|z_r|^2>), with leaky means at `rate`.

    Phase locking weighted by amplitude: a channel without input has no phase and gives 0."""

    def __init__(self, channels: int, rate: float = LOCK_RATE):
        self.channels, self.rate = channels, rate
        self.reset()

    def reset(self) -> None:
        self.cross = np.zeros(self.channels, dtype=np.complex128)
        self.power = np.zeros(self.channels)
        self.reference_power = 0.0

    def step(self, z_channels: np.ndarray, z_reference: complex) -> np.ndarray:
        b = self.rate
        self.cross = (1 - b) * self.cross + b * z_channels * np.conj(z_reference)
        self.power = (1 - b) * self.power + b * np.abs(z_channels) ** 2
        self.reference_power = (1 - b) * self.reference_power + b * abs(z_reference) ** 2
        denominator = np.sqrt(self.power * self.reference_power)
        return np.where(denominator > 0.0, np.abs(self.cross) / np.where(denominator > 0.0, denominator, 1.0), 0.0)


class RhythmBinder:
    """The map places that are synchronised with the loudest cochlear band.

    Each step: the places' peak luminance drives one oscillator bank, the cochlear band envelopes
    drive another. The loudest band is the one with the largest slow mean. Its strongest frequency
    k* is the frequency of the comparison. The coherence of each place with that band at k* is the
    binding. A step without sound (the first step of an episode) restarts everything.

    `scramble_seed`: the disruption. The band envelope of each step is replaced by the envelope
    of a step drawn at random from the last SCRAMBLE_HISTORY steps, which destroys the rhythm and
    keeps the values."""

    def __init__(self, grid: int = 16, bands: int = 64, scramble_seed: Optional[int] = None):
        self.grid, self.bands = grid, bands
        self.bank_cells = OscillatorBank(grid * grid)
        self.bank_bands = OscillatorBank(bands)
        self.coherence = Coherence(grid * grid)
        self.scramble = None if scramble_seed is None else np.random.default_rng(scramble_seed)
        self.history = deque(maxlen=SCRAMBLE_HISTORY)
        self.band_mean = None
        self.last = {"band": None, "frequency_index": None}

    def reset(self) -> None:
        self.bank_cells.reset()
        self.bank_bands.reset()
        self.coherence.reset()
        self.history.clear()
        self.band_mean = None
        self.last = {"band": None, "frequency_index": None}

    def track_bands(self, envelope: np.ndarray) -> None:
        envelope = np.asarray(envelope, dtype=np.float64)
        self.band_mean = envelope.copy() if self.band_mean is None else \
            self.band_mean + BAND_MEAN_RATE * (envelope - self.band_mean)

    def loudest_band(self) -> Optional[int]:
        return None if self.band_mean is None else int(np.argmax(self.band_mean))

    def _heard(self, envelope: np.ndarray) -> np.ndarray:
        if self.scramble is None:
            return envelope
        self.history.append(envelope)
        return self.history[int(self.scramble.integers(len(self.history)))]

    def step(self, peak_luminance: np.ndarray, envelope: Optional[np.ndarray]) -> np.ndarray:
        """[grid, grid] coherence of each place with the loudest band; zeros without sound."""
        if envelope is None:
            self.reset()
            return np.zeros((self.grid, self.grid))
        heard = self._heard(np.asarray(envelope, dtype=np.float64))
        self.track_bands(heard)
        z_cells = self.bank_cells.step(np.asarray(peak_luminance).reshape(-1))
        z_bands = self.bank_bands.step(heard)
        band = self.loudest_band()
        k = int(np.argmax(np.abs(z_bands[band])))
        self.last = {"band": band, "frequency_index": k}
        return self.coherence.step(z_cells[:, k], z_bands[band, k]).reshape(self.grid, self.grid)


def bound_cells(coherence: np.ndarray, threshold: float) -> List[int]:
    """Flat indices of the places whose coherence is at least `threshold`."""
    return [int(i) for i in np.flatnonzero(np.asarray(coherence).reshape(-1) >= threshold)]


# Chosen on room layouts 301 to 315 of the offline diagnosis, before the held-out test
# (docs/decisions/2026_10_15_rhythm_binding.md).
RHYTHM_THRESHOLD = 0.55
# No group before this many steps of sound since the last restart. With one or two samples the
# coherence of any two signals is 1. The diagnosis scored from step 12, and the plan set that the
# oscillators need about 12 steps before the agent moves.
WARM_UP = 12


def fresh_band_envelope(specialist):
    """A source of the cochlear band envelope of this step, or None when no new sound came in.

    The training loop does not call the cochlea at the first step of an episode, so the last
    stored envelope there is from the step before. The specialist counts its steps with sound;
    the same count twice means no new sound."""
    state = {"count": getattr(specialist, "band_envelope_count", 0)}

    def source():
        count = specialist.band_envelope_count
        if count == state["count"]:
            return None
        state["count"] = count
        return specialist.last_band_envelope

    return source


class RhythmMapBinding:
    """Map binding by timing, with the `observe` contract of the other map bindings.

    The group is the places locked with the loudest band (coherence of at least `threshold`),
    without the places of the agent's own body. Both senses get the same factor,
    1 + 0.5 x the largest coherence in the group, because they are one object; 1 without a group.
    `last_vision` is the peak luminance and `last_audio` is zero (the sound has no place here), so
    the motor map reads the drives of the binding as it does for the other forms."""

    needs_frames = True

    def __init__(self, band_source, scramble_seed: Optional[int] = None,
                 threshold: float = RHYTHM_THRESHOLD, grid: int = 16, bands: int = 64):
        from models.core.tectal_motor_map import body_mask
        self.band_source, self.threshold, self.grid = band_source, threshold, grid
        self.binder = RhythmBinder(grid=grid, bands=bands, scramble_seed=scramble_seed)
        self.body = body_mask().reshape(-1)
        self.last_vision = torch.zeros(grid, grid)
        self.last_audio = torch.zeros(grid, grid)

    def observe(self, vision_grid: torch.Tensor, audio_spatial: torch.Tensor, frames: torch.Tensor = None) -> tuple:
        """(factors, record) of one step."""
        if frames is None:
            raise ValueError("RhythmMapBinding needs the image of the step (frames [1, 3, h, w])")
        peak = cell_peak_luminance(frames, self.grid)
        coherence = self.binder.step(peak, self.band_source()).reshape(-1)
        group = [i for i in bound_cells(coherence, self.threshold) if not self.body[i]]
        if self.binder.bank_cells.t < WARM_UP:
            group = []
        self.last_vision = torch.from_numpy(peak).float()
        self.last_audio = torch.zeros_like(self.last_vision)
        factor = 1.0 + 0.5 * float(max(coherence[i] for i in group)) if group else 1.0
        record = {"blocks": 1 if group else 0, "group": group, "band": self.binder.last["band"],
                  "frequency_index": self.binder.last["frequency_index"],
                  "coherence": [round(float(c), 3) for c in coherence]}
        return {"vision": factor, "audio": factor}, record
