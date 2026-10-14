"""Does the place estimate of stage 2 mark the light? A diagnostic of the stage 2 failure.

`docs/results/binding_place_relation_2026_10.md` FAILED, and it names a probable reason.
The mean place agreement r was 0.13 to 0.17, so the vision and audio oscillators
repelled each other at most steps. The reason may be that the visual peak (the cell
with the largest feature norm) does not mark the light, for example because the agent's
own mark is drawn at the centre of the view.

DATA. One run with the stage 2 command and `--episodes 20`, seed 128, made with the
per-step place record (`binding_relation.json`, key `history`: one entry for each tectum
forward pass, [r, [[sound column, sound row], [peak column, peak row]]]).

TRUE PLACE. The frame of step t was drawn after step t - 1, so the ground truth of step
t - 1 is used. The agent-centred view is a window of half-width R = 48 px around the
agent, scaled onto 16 by 16 cells. The light cell is floor((d + R) / (2 R) * 16) for the
offset d of each axis, cut to 0 to 15. Steps with the light out of view, and step 0 of
each episode, are left out.

PREDICTION, written 2026-10-14 before the run.
  D1. The visual peak is within 2 cells of the light cell at less than half of the
      in-view steps.
The reading "the visual peak is the weak part of the stage 2 estimate" is CONFIRMED when
D1 holds, and NOT CONFIRMED when it does not. Reported only: D2, the same share for the
sound cell at steps where the light tone is the louder one, and the share of steps with
the peak in the four centre cells.

FAILED, AN ERROR IN THIS MEASURE, found 2026-10-14 after the run. The light has a
radius of 40 room pixels, about 6.7 cells, so a peak on the light can be far more than 2
cells from its centre. D1 was too strict and its verdict is VOID. A re-analysis with the
light pixels of the recorded frames is in `docs/results/binding_place_relation_2026_10.md`.

This is a statement about an internal estimate of place. It moves no indicator.

Usage:
    python -m scripts.analysis.probe_place_estimate --run runs/place_diag_s128
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from simulations.environments.audio_mixin import binaural_amplitude  # noqa: E402

GRID, VIEW_RADIUS, NEAR = 16, 48.0, 2.0
CENTRE = {(7, 7), (7, 8), (8, 7), (8, 8)}


def light_cell(offset, radius: float = VIEW_RADIUS, grid: int = GRID) -> tuple:
    """(column, row) of the light in the agent-centred view, cut to the grid."""
    return tuple(int(min(grid - 1, max(0, np.floor((d + radius) / (2 * radius) * grid)))) for d in offset)


def near(first, second, limit: float = NEAR) -> bool:
    return float(np.hypot(first[0] - second[0], first[1] - second[1])) <= limit


def step_infos(run: str) -> List[Optional[dict]]:
    """For each recorded step in order: the ground truth of the step before, or None at step 0."""
    infos = []
    for folder in sorted(glob.glob(os.path.join(run, "episodes", "ep_*"))):
        steps = [json.loads(line) for line in open(os.path.join(folder, "steps.jsonl"), encoding="utf-8")]
        infos.append(None)
        infos.extend(previous["internals"]["info_after_step"] for previous in steps[:-1])
    return infos


def shares(history: list, infos: List[Optional[dict]]) -> Dict[str, object]:
    if len(history) != len(infos):
        raise ValueError("the place record has %d entries and the step record %d" % (len(history), len(infos)))
    peak_near, sound_near, centre = [], [], []
    for entry, info in zip(history, infos):
        if info is None or entry[1] is None or not info.get("_truth_light_in_view"):
            continue
        (sound, peak), light = entry[1], light_cell(info["_truth_light_offset"])
        peak_near.append(near(peak, light))
        centre.append(tuple(peak) in CENTRE)
        decoy = info.get("_truth_decoy_offset")
        level = float(info.get("_truth_decoy_level", 1.0))
        light_amp = binaural_amplitude(float(np.hypot(*info["_truth_light_offset"])))
        if decoy is None or light_amp > level * binaural_amplitude(float(np.hypot(*decoy))):
            sound_near.append(near(sound, light))
    mean = lambda flags: float(np.mean(flags)) if flags else None  # noqa: E731
    return {"in_view_steps": len(peak_near), "D1_peak_near_light": mean(peak_near),
            "D2_sound_near_light_when_louder": mean(sound_near), "related_steps": len(sound_near),
            "peak_in_centre": mean(centre)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    with open(os.path.join(args.run, "binding_relation.json"), encoding="utf-8") as handle:
        history = json.load(handle)["history"]
    result = shares(history, step_infos(args.run))
    result["reading"] = "CONFIRMED" if result["D1_peak_near_light"] is not None and result["D1_peak_near_light"] < 0.5 else "NOT CONFIRMED"
    print(json.dumps(result, indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
