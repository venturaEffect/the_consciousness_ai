"""Does the oscillatory binding layer change which module leads, in the dark room?

Read-only probe on session records. It needs no checkpoint and no GPU.

BACKGROUND. The binding layer has 5 oscillators, one for each workspace module. Their
amplitudes are the bids. On DMTS the layer emits one value forever, because only one
module receives sensory input there (`docs/results/oscillator_frozen_2026_09.md`,
`docs/results/modality_starvation_2026_09.md`). With that one value the binding boost is
the same factor for every module and cannot change a rank. `sync_R` carries no stimulus
identity on DMTS, with and without the natural frequency fix
(`docs/results/sync_r_content_2026_09.md`, `docs/results/akorn_omega_fix_2026_09.md`).
In the dark room two senses have input and the layer moves. No document says what the
layer does to the competition there.

WHAT IS READ. For every recorded step except step 0 of an episode: the bids before the
binding (`internals.modulated_bids`), the bids after it (`bound_bids`) and `sync_r`. The
leader is the module with the highest bid. A step with a tie at the top, or with no bid
above 0, is left out and counted. The factor of a module is its bound bid divided by its
bid before the binding.

PREDICTIONS AND RULE, written 2026-10-14 before this probe ran on any record. Data: the
10 runs of Gate B7, `runs/gate_b7_s73` to `runs/gate_b7_s82`. Seen before this text:
one step, step 5 of episode 0 at seed 73. There the factor was 1.487 for vision and
1.493 for audio, 1.000 for memory and 1.083 for body.
  O1. The binding changes the leader at less than 0.05 of the steps.
  O2. A linear fit of `sync_r` on the five bids before the binding has an R squared of
      0.90 or more.
Each prediction is read per seed. The reading "the binding layer follows the bids and
does not decide the competition" is SUPPORTED when O1 and O2 hold at 8 or more of the
10 seeds each. It is NOT SUPPORTED when one of them fails at 3 or more seeds. Between
the two, or with a seed that cannot be measured, it is UNDECIDED. My expectation for O2
is not firm.

This is a statement about bids. It is not evidence about consciousness, and it moves no
indicator.

Usage:
    python -m scripts.analysis.probe_binding_decisive --runs runs/gate_b7_s73 runs/gate_b7_s74
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

MODULES = ("vision", "audio", "memory", "body", "semantic")
CHANGED_MAX, R_SQUARED_MIN, SEEDS_NEEDED, SEEDS_FAILED = 0.05, 0.90, 8, 3


def leader(bids: Dict[str, float]) -> Optional[str]:
    """The module with the highest bid. None for a tie at the top or when no bid is above 0."""
    ranked = sorted(bids.items(), key=lambda item: item[1], reverse=True)
    if not ranked or ranked[0][1] <= 0.0 or (len(ranked) > 1 and ranked[0][1] == ranked[1][1]):
        return None
    return ranked[0][0]


def binding_row(record: dict) -> Optional[dict]:
    """Leader before and after the binding, and the factor of every module with a bid above 0."""
    before_bids, after_bids = record["internals"]["modulated_bids"], record["bound_bids"]
    before, after = leader(before_bids), leader(after_bids)
    if before is None or after is None:
        return None
    factors = {name: after_bids[name] / before_bids[name] for name in before_bids if before_bids[name] > 0.0}
    return {"before": before, "after": after, "changed": before != after, "factors": factors}


def _mean(numbers: List[float]) -> Optional[float]:
    return sum(numbers) / len(numbers) if numbers else None


def summarise(records: List[dict]) -> Dict[str, object]:
    """Steps, share of steps where the leader changed, mean factor per module, and the gap of the two sense factors."""
    rows = [binding_row(record) for record in records]
    kept = [row for row in rows if row is not None]
    both = [row["factors"] for row in kept if "vision" in row["factors"] and "audio" in row["factors"]]
    return {"steps": len(kept), "left_out": len(rows) - len(kept),
            "changed_share": _mean([1.0 if row["changed"] else 0.0 for row in kept]),
            "mean_factor": {name: _mean([row["factors"][name] for row in kept if name in row["factors"]]) for name in MODULES},
            "sense_factor_gap": _mean([abs(f["vision"] - f["audio"]) for f in both])}


def r_squared(bids: np.ndarray, target: np.ndarray) -> Optional[float]:
    """R squared of a least-squares fit of the target on the bids with an intercept. None for a constant target."""
    target = np.asarray(target, dtype=np.float64)
    total = float(np.sum((target - target.mean()) ** 2))
    if len(target) < 3 or total <= 1e-18:
        return None
    design = np.column_stack([np.asarray(bids, dtype=np.float64), np.ones(len(target))])
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    return 1.0 - float(np.sum((target - design @ coefficients) ** 2)) / total


def sync_fit(records: List[dict]) -> Dict[str, object]:
    """R squared of sync_r on the five bids before the binding, and how many values sync_r takes."""
    bids = np.array([[record["internals"]["modulated_bids"][name] for name in MODULES] for record in records])
    sync = np.array([record["sync_r"] for record in records], dtype=np.float64)
    return {"r_squared": r_squared(bids, sync) if len(records) else None,
            "sync_distinct": int(len(np.unique(np.round(sync, 9)))), "sync_sd": float(sync.std()) if len(sync) else None}


def _holds(seeds: List[dict], key: str, test) -> List[Optional[bool]]:
    return [None if seed[key] is None else bool(test(seed[key])) for seed in seeds]


def verdict(seeds: List[dict]) -> str:
    """SUPPORTED, NOT SUPPORTED or UNDECIDED, by the rule of the docstring."""
    readings = [_holds(seeds, "changed_share", lambda share: share < CHANGED_MAX),
                _holds(seeds, "r_squared", lambda value: value >= R_SQUARED_MIN)]
    if any(sum(1 for r in reading if r is False) >= SEEDS_FAILED for reading in readings):
        return "NOT SUPPORTED"
    if all(sum(1 for r in reading if r is True) >= SEEDS_NEEDED for reading in readings):
        return "SUPPORTED"
    return "UNDECIDED"


def main() -> None:
    from scripts.analysis.probe_gate_b import load_steps
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    seeds = []
    for run in args.runs:
        records = [row[1] for row in load_steps(run)]
        seed = {"run": run, **summarise(records), **sync_fit(records)}
        seeds.append(seed)
        factors = " ".join("%s %s" % (name, "none" if value is None else "%.3f" % value) for name, value in seed["mean_factor"].items())
        print("%s steps %d left out %d changed %s R2 %s sync values %d\n   mean factor: %s  sense gap %s" % (
            run, seed["steps"], seed["left_out"],
            "none" if seed["changed_share"] is None else "%.4f" % seed["changed_share"],
            "none" if seed["r_squared"] is None else "%.4f" % seed["r_squared"], seed["sync_distinct"], factors,
            "none" if seed["sense_factor_gap"] is None else "%.4f" % seed["sense_factor_gap"]))
    print("reading: %s" % verdict(seeds))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump({"seeds": seeds, "verdict": verdict(seeds)}, handle, indent=2)


if __name__ == "__main__":
    main()
