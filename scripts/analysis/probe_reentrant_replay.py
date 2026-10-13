"""Does the size of the two sense vectors cause the vision takeover under `top_winner`?

Read-only replay on session records of runs made with `--broadcast-merge top_winner`. It
needs no checkpoint and no GPU.

BACKGROUND. `docs/results/top_winner_diagnosis_2026_10.md` found that the reentrant cycles
flip 53 to 67 percent of the audio leads to vision under `top_winner`. After that result
I saw, in 3 runs, that the audio vector has size 16.00 and the vision vector 1.22 to 1.39.
The rule of `SensoryTectum.receive_broadcast` and `AuditorySpecialist.receive_broadcast`
is pe = |broadcast - own vector| / |own vector|, then bid + 0.1 * pe (at most 1.0) for pe
above 0.3 and bid * (1 - 0.1 * pe) otherwise. With these sizes the vision bid receives a
pe of about 12 and the audio bid a pe of about 1.

THE REPLAY. For each recorded step it starts from the two sense bids before the cycles
(`raw_bids`) and the recorded vectors (`tectum_content`, `audio_content`). In each cycle
the broadcast is the vector of the sense with the higher bid, both senses apply the rule,
and the replay stops when the leader is the same as in the cycle before, or after 5
cycles. It leaves out what it cannot rebuild from a record: the affective modulation,
the binding, the ignition test and the three other modules.

ARMS. "recorded" uses the vectors as recorded. "unit" scales both vectors to size 1
before the rule, so both senses get the same pe.

PREDICTIONS AND RULE, written 2026-10-13 before this probe ran on any record. Data: the
six `top_winner` runs, seeds 73 to 75 (`runs/carrier_t1_s*`) and 83 to 85
(`runs/carrier_t2_s*`). A flip is a step where the other sense is strictly ahead after
the cycles.
  R1, validity. At audio-led steps and at vision-led steps, the "recorded" arm gives the
      same flip or no flip as the record (`settle.final_bids`) at 0.90 or more of the
      steps, in each of the 6 runs. If not, the replay does not describe the cycles and
      the verdict is VOID.
  R2, the size hypothesis. In the "unit" arm the share of audio leads that flip is below
      half of that share in the "recorded" arm. SUPPORTED when this holds at 6 of 6 runs.
      NOT SUPPORTED when it fails at 2 or more runs. UNDECIDED at 1.
My expectation is not firm. With unit size the pe of two unrelated vectors is about 1.4,
which adds 0.14 to the bid of the sense that is behind, and many audio leads are smaller
than that. So the rule itself may flip them at any vector size.

This is a statement about bids. It is not evidence about consciousness, and it moves no
indicator.

Usage:
    python -m scripts.analysis.probe_reentrant_replay --runs runs/carrier_t1_s73 runs/carrier_t1_s74
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

SENSES = ("vision", "audio")
VECTOR_KEYS = {"vision": "tectum_content", "audio": "audio_content"}
PE_LIMIT, GAIN, MAX_CYCLES = 0.3, 0.1, 5
VALID_AGREEMENT = 0.90


def prediction_error(broadcast: np.ndarray, own: np.ndarray, unit: bool) -> float:
    """|broadcast - own| / |own|. With unit, both vectors are scaled to size 1 first."""
    if unit:
        broadcast = broadcast / (np.linalg.norm(broadcast) + 1e-8)
        own = own / (np.linalg.norm(own) + 1e-8)
    return float(np.linalg.norm(broadcast - own) / (np.linalg.norm(own) + 1e-8))


def feedback_bid(bid: float, pe: float) -> float:
    """The bid after one feedback step of the rule in receive_broadcast."""
    updated = min(1.0, bid + GAIN * pe) if pe > PE_LIMIT else bid * (1.0 - GAIN * pe)
    return max(0.0, min(1.0, updated))


def replay_settle(bids: Dict[str, float], vectors: Dict[str, np.ndarray], unit: bool,
                  max_cycles: int = MAX_CYCLES) -> Tuple[Dict[str, float], int]:
    """Final sense bids and the number of cycles, with the leader's vector as the broadcast."""
    current = {sense: float(bids[sense]) for sense in SENSES}
    previous_leader, cycles = None, 0
    for cycle in range(max_cycles):
        cycles = cycle + 1
        leader = max(SENSES, key=lambda sense: current[sense])
        if cycle >= 1 and leader == previous_leader:
            break
        for sense in SENSES:
            current[sense] = feedback_bid(current[sense], prediction_error(vectors[leader], vectors[sense], unit))
        previous_leader = leader
    return current, cycles


def _flipped(bids: Dict[str, float], leader: str) -> bool:
    other = "audio" if leader == "vision" else "vision"
    return bids[other] > bids[leader]


def _share(flags: List[bool]) -> Optional[float]:
    return sum(flags) / len(flags) if flags else None


def run_table(steps: list) -> Dict[str, Dict[str, object]]:
    """Per leader: steps, the recorded, replayed and unit flip shares, and the agreement of replay and record."""
    rows: Dict[str, List[Tuple[bool, bool, bool]]] = {sense: [] for sense in SENSES}
    for record, vision_vector, audio_vector in steps:
        raw = record["raw_bids"]
        if raw["vision"] == raw["audio"] or min(raw["vision"], raw["audio"]) <= 0.0:
            continue
        leader = "vision" if raw["vision"] > raw["audio"] else "audio"
        vectors = {"vision": vision_vector, "audio": audio_vector}
        rows[leader].append((_flipped(record["internals"]["settle"]["final_bids"], leader),
                             _flipped(replay_settle(raw, vectors, unit=False)[0], leader),
                             _flipped(replay_settle(raw, vectors, unit=True)[0], leader)))
    return {sense: {"steps": len(flags),
                    "recorded_flip_share": _share([f[0] for f in flags]),
                    "replay_flip_share": _share([f[1] for f in flags]),
                    "agreement": _share([f[0] == f[1] for f in flags]),
                    "unit_flip_share": _share([f[2] for f in flags])}
            for sense, flags in rows.items()}


def verdict(tables: List[Dict[str, Dict[str, object]]]) -> str:
    """VOID, SUPPORTED, NOT SUPPORTED or UNDECIDED, by the rule of the docstring."""
    agreements = [table[sense]["agreement"] for table in tables for sense in SENSES]
    if any(a is None or a < VALID_AGREEMENT for a in agreements):
        return "VOID"
    failures = sum(1 for table in tables
                   if not table["audio"]["unit_flip_share"] < 0.5 * table["audio"]["replay_flip_share"])
    return "SUPPORTED" if failures == 0 else ("NOT SUPPORTED" if failures >= 2 else "UNDECIDED")


def load_run(run: str) -> list:
    """(step record, vision vector, audio vector) for every recorded step except step 0 of an episode."""
    steps = []
    for folder in sorted(glob.glob(os.path.join(run, "episodes", "ep_*"))):
        records = [json.loads(line) for line in open(os.path.join(folder, "steps.jsonl"), encoding="utf-8")]
        arrays = np.load(os.path.join(folder, "vectors.npz"))
        by_step = {sense: dict(zip(arrays[key + "__steps"].tolist(), arrays[key].reshape(len(arrays[key]), -1)))
                   for sense, key in VECTOR_KEYS.items()}
        for record in records[1:]:
            if all(record["step"] in by_step[sense] for sense in SENSES):
                steps.append((record, by_step["vision"][record["step"]], by_step["audio"][record["step"]]))
    return steps


def _line(sense: str, row: Dict[str, object]) -> str:
    shares = " ".join("%s %s" % (key, "none" if row[key] is None else "%.3f" % row[key])
                      for key in ("recorded_flip_share", "replay_flip_share", "agreement", "unit_flip_share"))
    return "  %s leads at %4d steps: %s" % (sense, row["steps"], shares)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", required=True, help="run folders made with --broadcast-merge top_winner")
    parser.add_argument("--output-json")
    args = parser.parse_args()
    tables = []
    for run in args.runs:
        table = run_table(load_run(run))
        tables.append(table)
        print(run)
        for sense in SENSES:
            print(_line(sense, table[sense]))
    print("size hypothesis: %s" % verdict(tables))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump({"runs": dict(zip(args.runs, tables)), "verdict": verdict(tables)}, handle, indent=2)


if __name__ == "__main__":
    main()
