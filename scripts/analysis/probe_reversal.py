"""Gate of the reversal (docs/decisions/2026_10_16_reversal.md): R1 to R3.

Read-only probe on runs made with `--tectal-orienting value`, one arm with
`--dark-room-food-switch 30` (reversal) and one without (stay). First object: within the first 60
steps of an episode, the first object closer than 60 room pixels (light radius 40 plus agent radius
20); no contact gives None. The food object of an episode is info["_truth_food"], or the light when
the run has no switch.

  R1: reversal arm, episodes 21 to 30: light first >= 0.8.
  R2: reversal arm, episodes 46 to 60: decoy first >= 0.8.
  R3: decoy first in episodes 46 to 60, reversal arm above the stay arm (pooled one-sided Fisher
      p < 0.05, difference > 0 at each seed).

Usage:
    python -m scripts.analysis.probe_reversal --reversal runs/rv_switch_s701 ... \
        --stay runs/rv_stay_s701 ...
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from scripts.analysis.probe_motor_map import compare_arms
from scripts.analysis.probe_value_gate import seed_label, verdict

CONTACT, LIMIT = 60.0, 60
BEFORE, AFTER, EARLY = (20, 30), (45, 60), (30, 35)
SWITCH, SHARE = 30, 0.8

Outcome = Tuple[Optional[str], str]


def first_object(steps: List[dict], limit: int = LIMIT) -> Optional[str]:
    """The first object within CONTACT in the first `limit` steps; the nearer one on a tie; None."""
    for record in steps[:limit]:
        info = record["internals"]["info_after_step"]
        distance = {name: float(np.hypot(*info["_truth_%s_offset" % name])) for name in ("light", "decoy")}
        near = [name for name in ("light", "decoy") if distance[name] < CONTACT]
        if near:
            return min(near, key=lambda name: distance[name])
    return None


def food_object(steps: List[dict]) -> str:
    """The food object of the episode, from its first step; the light without a switch."""
    return steps[0]["internals"]["info_after_step"].get("_truth_food", "light")


def run_outcomes(run: str) -> List[Outcome]:
    """(first object, food object) for each episode, in order."""
    outcomes = []
    for folder in sorted(glob.glob(os.path.join(run, "episodes", "ep_*"))):
        with open(os.path.join(folder, "steps.jsonl"), encoding="utf-8") as handle:
            steps = [json.loads(line) for _, line in zip(range(LIMIT), handle)]
        outcomes.append((first_object(steps), food_object(steps)))
    return outcomes


def share(outcomes: List[Outcome], window: Tuple[int, int], target: str) -> float:
    """Share of the episodes in [start, stop) whose first object is `target` ("food": the food object)."""
    part = outcomes[window[0]:window[1]]
    if not part:
        return float("nan")
    return float(np.mean([first == (food if target == "food" else target) for first, food in part]))


def episodes_to_switch(outcomes: List[Outcome], switch: int = SWITCH, width: int = 5, need: int = 4) -> Optional[int]:
    """Episodes after the switch until `need` of `width` reach the food object first; None if never."""
    hits = [first == food for first, food in outcomes[switch:]]
    for k in range(len(hits) - width + 1):
        if sum(hits[k:k + width]) >= need:
            return k
    return None


def checks(reversal: Dict[str, List[Outcome]], stay: Dict[str, List[Outcome]]) -> tuple:
    """(holds of R1 to R3 per seed, the R3 comparison). Both arms are keyed by the same seeds."""
    late = {arm: {name: [first == "decoy" for first, _ in values[AFTER[0]:AFTER[1]]] for name, values in runs.items()}
            for arm, runs in (("reversal", reversal), ("stay", stay))}
    comparison = compare_arms(late["reversal"], late["stay"])
    holds = {"R1": [], "R2": [], "R3": []}
    for name, values in reversal.items():
        holds["R1"].append(share(values, BEFORE, "light") >= SHARE)
        holds["R2"].append(share(values, AFTER, "decoy") >= SHARE)
        holds["R3"].append(comparison["p_one_sided"] < 0.05 and (comparison["per_seed"][name] or 0.0) > 0.0)
    return holds, comparison


def report(name: str, values: List[Outcome]) -> dict:
    """The reported-only numbers of one run."""
    return {"light_first_21_30": share(values, BEFORE, "light"), "decoy_first_46_60": share(values, AFTER, "decoy"),
            "old_object_first_31_35": share(values, EARLY, "light"), "episodes": len(values),
            "episodes_to_switch": episodes_to_switch(values)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reversal", nargs="+", required=True)
    parser.add_argument("--stay", nargs="+", required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    if len(args.reversal) != len(args.stay):
        parser.error("each arm needs one run for each seed, in the same order")
    names = [seed_label(run) for run in args.reversal]
    reversal = {name: run_outcomes(run) for name, run in zip(names, args.reversal)}
    stay = {name: run_outcomes(run) for name, run in zip(names, args.stay)}
    result = {"reversal": {n: report(n, v) for n, v in reversal.items()},
              "stay": {n: report(n, v) for n, v in stay.items()}, "values": {}}
    for arm in ("reversal", "stay"):
        for name, numbers in result[arm].items():
            print(arm, name, numbers)
    for name, run in zip(names, args.reversal):
        path = os.path.join(run, "value_gate.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                result["values"][name] = json.load(handle)["values"]
            print("learned values", name, result["values"][name])
    holds, comparison = checks(reversal, stay)
    result["comparison"], result["holds"], result["verdict"] = comparison, holds, verdict(holds)
    print("decoy first, episodes 46 to 60, reversal against stay:", comparison)
    print("holds:", holds)
    print("VERDICT:", result["verdict"])
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
