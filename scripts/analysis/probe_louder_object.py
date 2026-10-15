"""Does the oscillator map select the object with the louder tone? A check of the visible decoy.

Read-only probe on runs made with `--dark-room-decoy-visible` (the decoy is drawn like the
light, so the colour cannot tell the two objects apart).

REFERENCE. The light-coloured pixels of the recorded frame (within 30 of LIGHT_COLOUR, summed
over the three channels), split into 8-connected parts. Each part belongs to the object whose
recorded place, seen from the agent, is nearest to the centre of the part. A map cell (14 by 14
px) belongs to an object when more than half of its pixels belong to that object.

STEPS. A step counts when both objects were in view and one tone was the louder at the agent
(`probe_binding_selectivity.condition`). The louder object is the light at a "related" step and
the decoy at an "unrelated" step. The selected group is on the louder object when at least half
of its cells are cells of that object.

Without `map_binding.json` (map binding off) the probe counts the steps only.

Usage:
    python -m scripts.analysis.probe_louder_object --runs runs/vd_s200 --decoy-levels 0.5,3.0
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Optional

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.analysis.probe_binding_selectivity import condition, episode_level  # noqa: E402
from scripts.analysis.probe_map_selection import CELL, GRID, TOLERANCE, on_light  # noqa: E402
from simulations.environments.simple_visual_env import DEFAULT_VIEW_RADIUS, LIGHT_COLOUR  # noqa: E402

FRAME = GRID * CELL
EIGHT = np.ones((3, 3), dtype=int)


def frame_centre(offset, view_radius: int = DEFAULT_VIEW_RADIUS) -> tuple:
    """(x, y) frame pixels of a room offset from the agent, in the agent-centred view."""
    scale = FRAME / (2.0 * view_radius)
    return ((offset[0] + view_radius) * scale, (offset[1] + view_radius) * scale)


def object_cells(frame: np.ndarray, light_offset, decoy_offset, view_radius: int = DEFAULT_VIEW_RADIUS) -> dict:
    """{"light": [16, 16] bool, "decoy": [16, 16] bool}: the map cells of each object."""
    match = np.abs(frame.astype(float) - np.array(LIGHT_COLOUR, dtype=float)).sum(-1) < TOLERANCE
    labels, count = ndimage.label(match, structure=EIGHT)
    centres = {name: np.array(frame_centre(offset, view_radius))
               for name, offset in (("light", light_offset), ("decoy", decoy_offset))}
    pixels = {name: np.zeros(match.shape, dtype=bool) for name in centres}
    for part, (y, x) in enumerate(ndimage.center_of_mass(match, labels, range(1, count + 1)), start=1):
        nearest = min(centres, key=lambda name: np.hypot(*(centres[name] - (x, y))))
        pixels[nearest] |= labels == part
    return {name: mask.reshape(GRID, CELL, GRID, CELL).mean(axis=(1, 3)) > 0.5 for name, mask in pixels.items()}


def louder_object(kind: Optional[str]) -> Optional[str]:
    """The object with the louder tone: the light at a related step, the decoy at an unrelated step."""
    return {"related": "light", "unrelated": "decoy"}.get(kind)


def share_on_louder(rows) -> Optional[float]:
    """Share of the counted steps with the group on the louder object. rows: (kind, group, masks, both)."""
    flags = [on_light(group, masks[louder_object(kind)])
             for kind, group, masks, both in rows if both and louder_object(kind) is not None]
    return float(np.mean(flags)) if flags else None


def load(run: str, levels) -> list:
    """(condition, group or None, masks or None, both in view) for every step after step 0."""
    path = os.path.join(run, "map_binding.json")
    history = json.load(open(path, encoding="utf-8"))["history"] if os.path.exists(path) else None
    rows, index = [], 0
    for folder in sorted(glob.glob(os.path.join(run, "episodes", "ep_*"))):
        steps = [json.loads(line) for line in open(os.path.join(folder, "steps.jsonl"), encoding="utf-8")]
        frames = np.load(os.path.join(folder, "frames.npz"))["frames"] if history is not None else None
        level = episode_level(int(folder.rsplit("_", 1)[1]), levels)
        for t in range(len(steps)):
            index += 1
            if t == 0:
                continue
            info = steps[t - 1]["internals"]["info_after_step"]
            both = bool(info.get("_truth_light_in_view")) and bool(info.get("_truth_decoy_in_view"))
            kind = condition(info, bool(info.get("_truth_light_in_view", False)), level)
            if history is None or not both:
                rows.append((kind, None, None, both))
                continue
            masks = object_cells(frames[t], info["_truth_light_offset"], info["_truth_decoy_offset"])
            rows.append((kind, history[index - 1]["group"], masks, both))
    if history is not None and index != len(history):
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
        counted = [row for row in rows if row[3] and louder_object(row[0]) is not None]
        result[run] = {"steps": len(rows), "both_in_view": len(counted),
                       "both_in_view_related": sum(1 for row in counted if row[0] == "related"),
                       "both_in_view_unrelated": sum(1 for row in counted if row[0] == "unrelated"),
                       "share_on_louder": share_on_louder(rows) if counted and counted[0][2] is not None else None}
        print(run, result[run])
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
