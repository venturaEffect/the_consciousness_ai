"""Is the group that the oscillator map selects on the light? A check of stage 3.

Read-only probe on runs made with `--map-binding` and `--dark-room-decoy-sound`.

REFERENCE. The light pixels of the recorded frame: a map cell (14 by 14 px of the 224 px
frame) is a light cell when more than half of its pixels have the light colour (within
30 of LIGHT_COLOUR, summed over the three channels). The re-analysis of stage 2 showed
that this reference is reliable and that a computed light cell is not.

STEPS. A step counts when the light was in view and the light tone was the louder source
at the agent (the "related" condition of `probe_binding_selectivity.condition`, with the
decoy level). The selected group of the step is on the light when at least half of its
cells are light cells.

PREDICTION, written in `docs/results/binding_map_stage3_2026_10.md` before the runs: the
selected group is on the light at more than half of the related steps, in each run.

Usage:
    python -m scripts.analysis.probe_map_selection --runs runs/s3b_s132 --decoy-levels 0.5,3.0
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import List, Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.analysis.probe_binding_selectivity import condition, episode_level  # noqa: E402
from simulations.environments.simple_visual_env import LIGHT_COLOUR  # noqa: E402

GRID, CELL, TOLERANCE = 16, 14, 30


def light_cells(frame: np.ndarray) -> np.ndarray:
    """[16, 16] bool: the map cells that are more than half covered by the light colour."""
    match = np.abs(frame.astype(float) - np.array(LIGHT_COLOUR, dtype=float)).sum(-1) < TOLERANCE
    return match.reshape(GRID, CELL, GRID, CELL).mean(axis=(1, 3)) > 0.5


def on_light(group: List[int], light: np.ndarray) -> bool:
    """True when at least half of the cells of the group are light cells."""
    if not group:
        return False
    hits = sum(1 for index in group if light.flat[index])
    return hits * 2 >= len(group)


def share_on_light(rows) -> Optional[float]:
    """Share of the related steps with the selected group on the light. rows: (condition, group, light)."""
    flags = [on_light(group, light) for kind, group, light in rows if kind == "related"]
    return float(np.mean(flags)) if flags else None


def load(run: str, levels) -> list:
    """(condition, group, light cells) for every step except step 0 of an episode, in record order."""
    with open(os.path.join(run, "map_binding.json"), encoding="utf-8") as handle:
        history = json.load(handle)["history"]
    rows, index = [], 0
    for folder in sorted(glob.glob(os.path.join(run, "episodes", "ep_*"))):
        steps = [json.loads(line) for line in open(os.path.join(folder, "steps.jsonl"), encoding="utf-8")]
        frames = np.load(os.path.join(folder, "frames.npz"))["frames"]
        level = episode_level(int(folder.rsplit("_", 1)[1]), levels)
        for t in range(len(steps)):
            entry, index = history[index], index + 1
            if t == 0:
                continue
            info = steps[t - 1]["internals"]["info_after_step"]
            kind = condition(info, bool(info.get("_truth_light_in_view", False)), level)
            rows.append((kind, entry["group"], light_cells(frames[t])))
    if index != len(history):
        raise ValueError("the map record has %d entries and the step record %d" % (len(history), index))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--decoy-levels", default="1.0")
    parser.add_argument("--output-json")
    args = parser.parse_args()
    levels = tuple(float(part) for part in args.decoy_levels.split(","))
    result = {}
    for run in args.runs:
        rows = load(run, levels)
        result[run] = {"related_steps": sum(1 for row in rows if row[0] == "related"), "share_on_light": share_on_light(rows)}
        print(run, result[run])
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
