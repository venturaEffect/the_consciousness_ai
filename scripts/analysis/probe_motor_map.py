"""Path gate of the tectal motor map: does the agent move toward what its tectum selects?

Read-only probe on runs made with `--tectal-orienting` (docs/decisions/2026_10_15_tectal_motor_map.md).
It reads `tectal_orienting.json` (one target per step, as a room offset from the agent, or None)
and the step records.

  toward:  at a step with a target, the agent's move of that step has a positive dot product with
           the target offset. The move is read from the light offset before and after the step.
  wall:    the share of steps with `collision` (a move stopped by the room's edge).
  contact: the episode reaches a distance below 60 room pixels (light radius 40 plus agent radius
           20) to the light or the decoy within the first `limit` steps.
  first contact with the louder object: the G3 measure of the motor map test. With `--base`, the
           runs are compared with the runs of a second arm (one-sided Fisher p, pooled).

Usage:
    python -m scripts.analysis.probe_motor_map --runs runs/tm_path_s197 ...
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import fisher_exact

CONTACT = 60.0
CONTACT_STEPS = 40


def _offsets(record: dict, name: str) -> np.ndarray:
    return np.asarray(record["internals"]["info_after_step"][name], dtype=float)


def toward_shares(steps: List[dict], targets: List[dict]) -> List[bool]:
    """One flag for each step after the first that has a target: did the agent move toward it?"""
    flags = []
    for t in range(1, len(steps)):
        target = targets[t]["target"]
        if target is None:
            continue
        move = _offsets(steps[t - 1], "_truth_light_offset") - _offsets(steps[t], "_truth_light_offset")
        flags.append(bool(float(np.dot(move, target)) > 0.0))
    return flags


def wall_share(steps: List[dict]) -> float:
    return float(np.mean([bool(s["internals"]["info_after_step"].get("collision")) for s in steps]))


def contact(steps: List[dict], limit: int = CONTACT_STEPS) -> bool:
    """True when the agent comes within CONTACT of the light or the decoy within `limit` steps."""
    for record in steps[:limit]:
        for name in ("_truth_light_offset", "_truth_decoy_offset"):
            if np.hypot(*_offsets(record, name)) < CONTACT:
                return True
    return False


def first_contact_louder(steps: List[dict], limit: int = CONTACT_STEPS) -> Optional[bool]:
    """True when the first object touched within `limit` steps is the one with the louder tone.

    The agent starts midway between the objects (--dark-room-start-between), so the decoy level
    decides which tone is louder: the light below 1, the decoy above 1. No contact counts as False.
    None at level 1."""
    level = float(steps[0]["internals"]["info_after_step"]["_truth_decoy_level"])
    if level == 1.0:
        return None
    louder = "_truth_decoy_offset" if level > 1.0 else "_truth_light_offset"
    for record in steps[:limit]:
        touched = [name for name in ("_truth_light_offset", "_truth_decoy_offset")
                   if np.hypot(*_offsets(record, name)) < CONTACT]
        if touched:
            return touched == [louder]
    return False


def compare_arms(test: Dict[str, list], base: Dict[str, list]) -> dict:
    """Per seed, share of True in the test arm minus the base arm; pooled shares; one-sided Fisher p."""
    per_seed, counts = {}, {"test": [0, 0], "base": [0, 0]}
    for seed in test:
        for arm, outcomes in (("test", test[seed]), ("base", base[seed])):
            kept = [o for o in outcomes if o is not None]
            counts[arm][0] += sum(kept)
            counts[arm][1] += len(kept)
        a, b = [o for o in test[seed] if o is not None], [o for o in base[seed] if o is not None]
        per_seed[seed] = float(np.mean(a) - np.mean(b)) if a and b else None
    (ht, nt), (hb, nb) = counts["test"], counts["base"]
    p = fisher_exact([[ht, nt - ht], [hb, nb - hb]], alternative="greater")[1]
    return {"per_seed": per_seed, "pooled_test": ht / nt if nt else None,
            "pooled_base": hb / nb if nb else None, "p_one_sided": float(p)}


def split(history: List[dict], lengths: List[int]) -> List[List[dict]]:
    """The motor map history cut into episodes. Raises when the lengths do not add up."""
    if sum(lengths) != len(history):
        raise ValueError("the motor map has %d entries and the step records %d" % (len(history), sum(lengths)))
    parts, start = [], 0
    for n in lengths:
        parts.append(history[start:start + n])
        start += n
    return parts


def summarise(run: str) -> dict:
    with open(os.path.join(run, "tectal_orienting.json"), encoding="utf-8") as handle:
        history = json.load(handle)["history"]
    episodes = [[json.loads(line) for line in open(os.path.join(folder, "steps.jsonl"), encoding="utf-8")]
                for folder in sorted(glob.glob(os.path.join(run, "episodes", "ep_*")))]
    targets = split(history, [len(steps) for steps in episodes])
    toward = [flag for steps, part in zip(episodes, targets) for flag in toward_shares(steps, part)]
    return {"episodes": len(episodes),
            "steps_with_target": sum(1 for h in history if h["target"] is not None),
            "toward_share": float(np.mean(toward)) if toward else None,
            "wall_share": float(np.mean([wall_share(steps) for steps in episodes])),
            "contact_share": float(np.mean([contact(steps) for steps in episodes])),
            "first_contact_louder": [first_contact_louder(steps) for steps in episodes]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--base", nargs="+", help="runs of the comparison arm, same seeds, same order")
    parser.add_argument("--output-json")
    args = parser.parse_args()
    result = {run: summarise(run) for run in args.runs + (args.base or [])}
    for run, values in result.items():
        louder = [o for o in values["first_contact_louder"] if o is not None]
        print(run, {k: v for k, v in values.items() if k != "first_contact_louder"},
              "first contact with the louder object %d of %d" % (sum(louder), len(louder)))
    if args.base:
        if len(args.base) != len(args.runs):
            parser.error("--base needs one run for each run of --runs")
        names = [os.path.basename(run) for run in args.runs]
        result["comparison"] = compare_arms(
            {n: result[r]["first_contact_louder"] for n, r in zip(names, args.runs)},
            {n: result[b]["first_contact_louder"] for n, b in zip(names, args.base)})
        print("comparison", result["comparison"])
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
