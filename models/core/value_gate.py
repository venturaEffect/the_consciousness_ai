"""The selection switch: a learned value for each heard tone decides which bound object the tectum follows.

`docs/decisions/2026_10_16_value_gate.md`. In monkeys, the caudate learns the value of objects from
reward, and through the substantia nigra it releases gaze toward high-value objects in the
superior colliculus (Kim and Hikosaka, Neuron 2013; Hikosaka, Takikawa and Kawagoe, Physiol. Rev.
2000). A learned value captures attention beyond salience (Anderson, Laurent and Yantis, PNAS
2011). The value is updated by the reward prediction error that dopamine neurons signal (Schultz,
Dayan and Montague, Science 1997).

Here a value V belongs to a cochlear band (a heard tone). Once in each episode, at the first step
with bound objects, the gate chooses a band by softmax over salience + VALUE_WEIGHT x V with
temperature TEMPERATURE, from a generator of its own, and keeps the choice to the episode end. The
outcome is 1 when food comes (reward above FOOD) within OUTCOME_STEPS steps after the choice, else 0.
The error is outcome - V, and V += learning rate x error, once per episode.

All values were set on 2026-10-16 before any run, and checked in an abstract model only.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

VALUE_WEIGHT = 2.0
TEMPERATURE = 0.25
LEARNING_RATE = 0.3
OUTCOME_STEPS = 60
FOOD = 0.5          # the reward of a step in the light is about 1.0; outside it is below 0


class ValueGate:
    """Values per band, one choice per episode, one update per episode."""

    def __init__(self, seed: int, learning_rate: float = LEARNING_RATE, value_weight: float = VALUE_WEIGHT,
                 temperature: float = TEMPERATURE, outcome_steps: int = OUTCOME_STEPS):
        self.learning_rate, self.value_weight = learning_rate, value_weight
        self.temperature, self.outcome_steps = temperature, outcome_steps
        self.rng = np.random.default_rng(seed)
        self.values: Dict[int, float] = {}
        self.episodes = []
        self.choice: Optional[int] = None
        self._open = None

    def probabilities(self, salience: Dict[int, float]) -> Dict[int, float]:
        """Softmax over salience + weight x value, at the temperature."""
        bands = list(salience)
        if len(bands) == 1:
            return {bands[0]: 1.0}
        score = np.array([salience[b] + self.value_weight * self.values.get(b, 0.0) for b in bands])
        score = (score - score.max()) / self.temperature
        p = np.exp(score) / np.exp(score).sum()
        return {b: float(q) for b, q in zip(bands, p)}

    def choose(self, salience: Dict[int, float]) -> int:
        """The band to follow in this episode. A choice already made is kept."""
        if self.choice is not None:
            return self.choice
        p = self.probabilities(salience)
        bands = list(p)
        self.choice = bands[int(self.rng.choice(len(bands), p=[p[b] for b in bands]))]
        self._open = {"choice": self.choice, "salience": {str(b): float(s) for b, s in salience.items()},
                      "priorities": {str(b): float(salience[b] + self.value_weight * self.values.get(b, 0.0))
                                     for b in salience},
                      "value_before": float(self.values.get(self.choice, 0.0)), "steps": 0}
        return self.choice

    def observe_reward(self, reward: float) -> None:
        """Food within the window closes the choice with 1; the end of the window closes it with 0."""
        if self._open is None:
            return
        self._open["steps"] += 1
        if reward > FOOD:
            self._close(1.0)
        elif self._open["steps"] >= self.outcome_steps:
            self._close(0.0)

    def new_episode(self) -> None:
        """A step without sound starts the next episode: an open choice closes with 0."""
        if self._open is not None:
            self._close(0.0)
        self.choice = None

    def _close(self, outcome: float) -> None:
        band = self._open["choice"]
        before = self.values.get(band, 0.0)
        error = outcome - before
        self.values[band] = before + self.learning_rate * error
        self.episodes.append(dict(self._open, outcome=outcome, error=float(error),
                                  value_after=float(self.values[band])))
        self._open = None
