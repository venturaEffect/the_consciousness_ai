"""Which sense bid do the reentrant cycles raise, the leader's or the other one's?

Read-only probe on session records. It needs no checkpoint and no GPU.

THE MECHANISM IN THE CODE. `ReentrantProcessor.settle` runs the workspace competition,
then sends the broadcast back to every specialist. `SensoryTectum.receive_broadcast` and
`AuditorySpecialist.receive_broadcast` compute pe = |broadcast - own content| / |own content|.
For pe above 0.3 the bid rises by 0.1 * pe. For pe at or below 0.3 the bid is multiplied by
1 - 0.1 * pe. So the module whose own vector is the broadcast keeps its bid (pe is 0), and
the other module gains. Which module is "the other" depends on `--broadcast-merge`. With
`legacy` the broadcast is the vector of the weakest admitted module, so the leader gains.
With `top_winner` the broadcast is the vector of the strongest admitted module, so the
module behind gains.

WHAT IS READ. For every recorded step except step 0 of an episode: the bids before the
cycles (`raw_bids`) and after them (`internals.settle.final_bids`). The leader is the
sense with the higher raw bid. Steps with equal raw bids, or with a sense bid of 0, are
left out and counted. A flip is a step where the other sense is strictly ahead after the
cycles.

PREDICTIONS, written 2026-10-13 before this probe was run on any record. Data: the runs
with the default merge and with `top_winner` at the same seeds, 73 to 75 (no learned
valence) and 83 to 85 (learned valence with the saturating boost). Already seen before
this text, in an exploration: at these six pairs the share of steps with audio ahead is
about equal for the two merges before the cycles and lower with `top_winner` after
them. The change of each bid by leader, and the flip share, were not computed before.
  D1. With `legacy`, the mean change of the leader's bid is above the mean change of the
      other bid, for both leaders.
  D2. With `top_winner`, the mean change of the other bid is above the mean change of
      the leader's bid, for both leaders.
  D3. The share of audio-led steps that flip to vision is higher with `top_winner` than
      with `legacy`.
Each prediction is read per seed pair as true, false, or not measurable (a leader with
no step). The mechanism reading is SUPPORTED when D1, D2 and D3 are true at 6 of 6
pairs, and NOT SUPPORTED when any of them is false at 2 or more pairs. Between the two
it is UNDECIDED. No threshold on the size of a change is set.

This is a statement about bids. It is not evidence about consciousness, and it moves no
indicator.

Usage:
    python -m scripts.analysis.probe_reentrant_boost --pair runs/gate_b7_s73 runs/carrier_t1_s73 \\
        --pair runs/gate_b7_s74 runs/carrier_t1_s74
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.analysis.probe_gate_b import load_steps  # noqa: E402

SENSES = ("vision", "audio")
PREDICTIONS = ("D1", "D2", "D3")


def bid_change(record: dict) -> Optional[dict]:
    """Leader before the cycles, the change of both sense bids, and if the lead flipped. None when there is no leader."""
    before = record["raw_bids"]
    after = record["internals"]["settle"]["final_bids"]
    if before["vision"] == before["audio"] or min(before["vision"], before["audio"]) <= 0.0:
        return None
    leader = "vision" if before["vision"] > before["audio"] else "audio"
    other = "audio" if leader == "vision" else "vision"
    return {"leader": leader,
            "leader_change": after[leader] - before[leader],
            "other_change": after[other] - before[other],
            "flipped": after[other] > after[leader]}


def _mean(numbers: List[float]) -> Optional[float]:
    return sum(numbers) / len(numbers) if numbers else None


def summarise(rows: list) -> Dict[str, object]:
    """Per leader: steps, mean change of the leader's bid and of the other bid, share of flips. None where no step exists."""
    changes = [bid_change(row[1]) for row in rows]
    summary: Dict[str, object] = {"left_out": sum(1 for c in changes if c is None)}
    for sense in SENSES:
        mine = [c for c in changes if c is not None and c["leader"] == sense]
        summary[sense] = {"steps": len(mine),
                          "leader_change": _mean([c["leader_change"] for c in mine]),
                          "other_change": _mean([c["other_change"] for c in mine]),
                          "flip_share": _mean([1.0 if c["flipped"] else 0.0 for c in mine])}
    return summary


def _above(first: Optional[float], second: Optional[float]) -> Optional[bool]:
    return None if first is None or second is None else first > second


def _all(values: List[Optional[bool]]) -> Optional[bool]:
    return None if any(v is None for v in values) else all(values)


def pair_directions(legacy: dict, top: dict) -> Dict[str, Optional[bool]]:
    """D1, D2 and D3 for one seed pair. None means not measurable."""
    return {
        "D1": _all([_above(legacy[s]["leader_change"], legacy[s]["other_change"]) for s in SENSES]),
        "D2": _all([_above(top[s]["other_change"], top[s]["leader_change"]) for s in SENSES]),
        "D3": _above(top["audio"]["flip_share"], legacy["audio"]["flip_share"]),
    }


def verdict(directions: List[Dict[str, Optional[bool]]]) -> str:
    false_counts = [sum(1 for d in directions if d[name] is False) for name in PREDICTIONS]
    if all(d[name] is True for d in directions for name in PREDICTIONS):
        return "SUPPORTED"
    return "NOT SUPPORTED" if max(false_counts) >= 2 else "UNDECIDED"


def _show(label: str, summary: dict) -> None:
    for sense in SENSES:
        s = summary[sense]
        numbers = " ".join("%s %s" % (k, "none" if s[k] is None else "%+.4f" % s[k]) for k in ("leader_change", "other_change"))
        flips = "none" if s["flip_share"] is None else "%.3f" % s["flip_share"]
        print("  %-10s %s leads at %4d steps: %s flip share %s" % (label, sense, s["steps"], numbers, flips))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pair", nargs=2, action="append", required=True, metavar=("LEGACY_RUN", "TOP_WINNER_RUN"))
    parser.add_argument("--output-json")
    args = parser.parse_args()
    report, directions = [], []
    for legacy_run, top_run in args.pair:
        legacy, top = summarise(load_steps(legacy_run)), summarise(load_steps(top_run))
        direction = pair_directions(legacy, top)
        directions.append(direction)
        report.append({"legacy_run": legacy_run, "top_winner_run": top_run, "legacy": legacy, "top_winner": top, "directions": direction})
        print("%s  against  %s" % (legacy_run, top_run))
        _show("legacy", legacy)
        _show("top_winner", top)
        print("  D1 %s  D2 %s  D3 %s" % tuple(direction[name] for name in PREDICTIONS))
    print("mechanism reading: %s" % verdict(directions))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump({"pairs": report, "verdict": verdict(directions)}, handle, indent=2)


if __name__ == "__main__":
    main()
