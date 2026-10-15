"""The motor map of the tectum: the selected place on the map gives an orienting movement.

`docs/decisions/2026_10_15_tectal_motor_map.md`. In the vertebrates studied, from lamprey to
monkey, the deep tectal layers hold a motor map aligned with the sensory maps, and the selected
place drives orienting toward it (Isa, Marquez-Legorreta, Grillner and Scott, Curr. Biol. 2021).
The movement is the weighted average over the active population (Lee, Rohrer and Sparks, Nature
1988). The basal ganglia gate this output (Hikosaka, Takikawa and Kawagoe, Physiol. Rev. 2000);
here the gate stays open, an innate reflex.

All values were set on 2026-10-15 before any run.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from models.core.map_binding import HALF_MAXIMUM, combine

GRID, FRAME, VIEW_RADIUS = 16, 224, 48       # map cells, frame pixels, half-width of the view in room pixels
AGENT_RADIUS = 20.0                          # room pixels; cells closer to the centre show the agent itself
SPEED, NOISE_STD = 0.5, 0.1                  # action units (one unit moves 10 room pixels per step)
AUDIO_PLACE = 0.5                            # sound drive at or above this marks the sound place
MODES = ("group", "active", "audio")


def cell_offsets(grid: int = GRID, view_radius: int = VIEW_RADIUS, frame: int = FRAME) -> tuple:
    """(dx, dy), each [grid, grid]: room offset from the agent to the centre of each map cell."""
    centres = (np.arange(grid) + 0.5) * (frame / grid) * (2.0 * view_radius / frame) - view_radius
    dx, dy = np.meshgrid(centres, centres)
    return dx, dy


def body_mask(radius: float = AGENT_RADIUS) -> np.ndarray:
    """[16, 16] bool: the cells whose centre lies on the agent's own body."""
    dx, dy = cell_offsets()
    return np.hypot(dx, dy) < radius


_OFFSETS = np.stack(cell_offsets(), axis=-1).reshape(-1, 2)
_BODY = body_mask().reshape(-1)


def orient_vector(cells: List[int], weights) -> Optional[np.ndarray]:
    """Room offset of the weighted centre of the target cells without the body. None without a target."""
    flat = np.asarray(weights, dtype=np.float64).reshape(-1)
    kept = [i for i in cells if not _BODY[i] and flat[i] > 0.0]
    if not kept:
        return None
    w = flat[kept]
    vector = (w[:, None] * _OFFSETS[kept]).sum(axis=0) / w.sum()
    return vector if np.hypot(*vector) > 1e-9 else None


def command(vector: np.ndarray, noise: np.ndarray, speed: float = SPEED) -> np.ndarray:
    """Action: speed times the unit vector toward the target, plus noise, inside [-1, 1]."""
    unit = vector / np.hypot(*vector)
    return np.clip(speed * unit + noise, -1.0, 1.0)


class TectalMotorMap:
    """Turns the map binding's last step into an orienting command, or None without a target.

    `mode` chooses the target cells:
      "group":  the cells of the group the map binding selected;
      "active": every cell whose combined drive is at least half maximum, with no selection
                (the disruption arm: two targets give an averaged movement);
      "audio":  the sound place, cells whose sound drive is at least AUDIO_PLACE.
    The noise comes from its own generator, so no other random stream changes.
    """

    def __init__(self, mode: str, seed: int, speed: float = SPEED, noise_std: float = NOISE_STD):
        if mode not in MODES:
            raise ValueError("mode must be one of %s, got %r" % (MODES, mode))
        self.mode, self.speed, self.noise_std = mode, speed, noise_std
        self.rng = np.random.default_rng(seed)
        self.history = []

    def target(self, binding, group: List[int]) -> tuple:
        """(cells, weights [16, 16]) of this step's target."""
        vision, audio = binding.last_vision.detach().float(), binding.last_audio.detach().float()
        drive = combine(vision, audio).numpy()
        if self.mode == "group":
            return list(group), drive
        if self.mode == "active":
            return [int(i) for i in np.flatnonzero(drive.reshape(-1) >= HALF_MAXIMUM)], drive
        sound = audio.numpy()
        return [int(i) for i in np.flatnonzero(sound.reshape(-1) >= AUDIO_PLACE)], sound

    def step(self, binding, group: List[int]) -> Optional[np.ndarray]:
        """The command of this step, or None without a target. The target is recorded."""
        vector = orient_vector(*self.target(binding, group))
        self.history.append({"target": None if vector is None else [float(v) for v in vector]})
        if vector is None:
            return None
        return command(vector, self.rng.normal(0.0, self.noise_std, 2), self.speed)
