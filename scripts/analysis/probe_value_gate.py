"""Gate of the value gate (docs/decisions/2026_10_16_value_gate.md): L1 to L3.

Read-only probe on runs made with `--tectal-orienting value`. Light first: within the first 60
steps of an episode, the first object closer than 60 room pixels (light radius 40 plus agent
radius 20) is the light; no contact counts as not. The decoy is louder in episodes with level 3.0.

  L1: learn arm, episodes 21 to 40: light first >= 0.8; and above the frozen arm (pooled one-sided
      Fisher p < 0.05, difference > 0 at each seed).
  L2: learn arm, level-3.0 episodes among 21 to 40: light first >= 0.7.
  L3: learn arm above the scrambled arm in episodes 21 to 40 (pooled one-sided Fisher p < 0.05,
      difference > 0 at each seed).

Usage:
    python -m scripts.analysis.probe_value_gate --learn runs/vg_learn_s601 ... \
        --frozen runs/vg_frozen_s601 ... --scrambled runs/vg_scrambled_s601 ...
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np

from scripts.analysis.probe_motor_map import compare_arms

CONTACT, LIMIT, LATE, SHARE_LATE, SHARE_LOUD = 60.0, 60, 20, 0.8, 0.7


def light_first(steps: List[dict], limit: int = LIMIT) -> bool:
    """True when the first object within CONTACT, in the first `limit` steps, is the light."""
    for record in steps[:limit]:
        info = record["internals"]["info_after_step"]
        near = [name for name in ("light", "decoy") if np.hypot(*info["_truth_%s_offset" % name]) < CONTACT]
        if near:
            return near == ["light"] or (len(near) == 2 and
                                          np.hypot(*info["_truth_light_offset"]) <= np.hypot(*info["_truth_decoy_offset"]))
    return False


def run_outcomes(run: str) -> List[tuple]:
    """(light first, decoy louder) for each episode, in order."""
    outcomes = []
    for folder in sorted(glob.glob(os.path.join(run, "episodes", "ep_*"))):
        with open(os.path.join(folder, "steps.jsonl"), encoding="utf-8") as handle:
            steps = [json.loads(line) for _, line in zip(range(LIMIT), handle)]
        level = float(steps[0]["internals"]["info_after_step"]["_truth_decoy_level"])
        outcomes.append((light_first(steps), level > 1.0))
    return outcomes


def late_shares(outcomes: List[tuple]) -> tuple:
    """(light first in episodes 21 and later, the same in those of them with the decoy louder)."""
    late = outcomes[LATE:]
    loud = [hit for hit, decoy_louder in late if decoy_louder]
    return (float(np.mean([hit for hit, _ in late])) if late else float("nan"),
            float(np.mean(loud)) if loud else float("nan"))


def seed_label(run: str) -> str:
    """The seed part of a run folder name ("vg_learn_s601" gives "s601"), shared by the three arms."""
    return os.path.basename(run.rstrip("/\\")).rsplit("_", 1)[-1]


def verdict(holds: Dict[str, List[bool]]) -> str:
    """PASS when every check holds at every seed; FAIL when one fails at 2 or more seeds."""
    if all(all(seeds) for seeds in holds.values()):
        return "PASS"
    if any(sum(1 for ok in seeds if not ok) >= 2 for seeds in holds.values()):
        return "FAIL"
    return "UNDECIDED"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for arm in ("learn", "frozen", "scrambled"):
        parser.add_argument("--" + arm, nargs="+", required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    if not len(args.learn) == len(args.frozen) == len(args.scrambled):
        parser.error("each arm needs one run for each seed, in the same order")
    names = [seed_label(run) for run in args.learn]
    outcomes = {arm: {n: run_outcomes(r) for n, r in zip(names, getattr(args, arm))}
                for arm in ("learn", "frozen", "scrambled")}
    result = {"shares": {}, "values": {}}
    for arm, per_seed in outcomes.items():
        for name, values in per_seed.items():
            early = float(np.mean([hit for hit, _ in values[:LATE]]))
            late, loud = late_shares(values)
            result["shares"]["%s %s" % (arm, name)] = {"episodes_1_20": early, "episodes_21_40": late,
                                                       "decoy_louder_21_40": loud, "episodes": len(values)}
            print(arm, name, result["shares"]["%s %s" % (arm, name)])
    for name, run in zip(names, args.learn):
        path = os.path.join(run, "value_gate.json")
        if os.path.exists(path):
            result["values"][name] = json.load(open(path, encoding="utf-8"))["values"]
            print("learned values", name, result["values"][name])
    late = {arm: {n: [hit for hit, _ in v[LATE:]] for n, v in per_seed.items()} for arm, per_seed in outcomes.items()}
    versus = {base: compare_arms(late["learn"], late[base]) for base in ("frozen", "scrambled")}
    result["comparison"] = versus
    for base, comparison in versus.items():
        print("learn against", base, comparison)
    holds = {"L1": [], "L2": [], "L3": []}
    for name in names:
        share, loud = late_shares(outcomes["learn"][name])
        holds["L1"].append(share >= SHARE_LATE and versus["frozen"]["p_one_sided"] < 0.05
                           and (versus["frozen"]["per_seed"][name] or 0.0) > 0.0)
        holds["L2"].append(loud >= SHARE_LOUD)
        holds["L3"].append(versus["scrambled"]["p_one_sided"] < 0.05 and (versus["scrambled"]["per_seed"][name] or 0.0) > 0.0)
    result["holds"], result["verdict"] = holds, verdict(holds)
    print("holds:", holds)
    print("VERDICT:", result["verdict"])
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
