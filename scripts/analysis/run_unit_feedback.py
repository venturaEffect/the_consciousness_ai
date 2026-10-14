"""Does the size of the sense vectors cause the vision takeover under `top_winner`? A test on the real loop.

BACKGROUND. `docs/results/top_winner_diagnosis_2026_10.md` found that with
`--broadcast-merge top_winner` the reentrant cycles flip 53 to 67 percent of the audio
leads to vision. The rule of `SensoryTectum.receive_broadcast` and
`AuditorySpecialist.receive_broadcast` is pe = |broadcast - own vector| / |own vector|,
then bid + 0.1 * pe for pe above 0.3. In 3 recorded runs the audio vector has size 16.00
and the vision vector 1.22 to 1.39. So the vision bid receives a pe of about 12 when the
audio vector is the broadcast, and the audio bid a pe of about 1 in the other case. A
replay on session records could not test this, because it did not match the record
(verdict VOID in the same document).

THE INTERVENTION. This launcher starts the training loop as it is, with one change made
in memory. Before each of the two rules runs, the broadcast tensor is scaled to the
size of the module's own vector. The rule itself is not copied and not changed. With
equal sizes, pe is the distance of the two vectors at unit size, for both senses. A
call with no broadcast tensor or no own vector goes to the rule unchanged. The launcher
counts the scaled calls of each sense and writes them to `unit_feedback_calls.json` in
the run folder. No production file changes, and a run without this launcher is the
same as before.

This is a test instrument. It is not a proposed mechanism, and no default changes.

PREDICTIONS AND RULE, written 2026-10-14 before the launcher ran on any seed.
Control arm: the recorded `top_winner` runs `runs/carrier_t1_s73`, `_s74`, `_s75`
(Python 3.8.3, no learned valence). Unit arm: the same command and seeds through this
launcher, on Python 3.8.3. A flip is a step where the other sense is strictly ahead
after the cycles (`probe_reentrant_boost.summarise`).
  C0, control check. Seed 73 with the control command, run again at this revision
      without the launcher, gives a `metrics.csv` identical to `runs/carrier_t1_s73`.
      If not, the three control runs are made again at this revision and used.
  S0, the intervention ran. Each unit run has more than 0 scaled calls for vision and
      for audio. If not, the verdict is VOID.
  S1, the size hypothesis. In the unit arm the share of audio leads that flip to
      vision is below half of that share in the control arm, at the same seed.
      SUPPORTED when S1 holds at 3 of 3 seeds. NOT SUPPORTED when it fails at 2 or
      more. UNDECIDED at 1, or when a share cannot be measured.
My expectation is not firm. Two unrelated vectors at unit size have a distance of about
1.4, which adds 0.14 to the bid of the sense that is behind, and many audio leads are
smaller than that. The two arms see different frames after the first changed action,
so this compares run statistics at matched seeds. It does not compare step by step.
Three seeds give a hypothesis-grade result. It changes no default.

This is a statement about bids. It is not evidence about consciousness, and it moves no
indicator.

Usage:
    python -m scripts.analysis.run_unit_feedback <arguments of train_rlhf, with --log-dir>
    python -m scripts.analysis.run_unit_feedback judge --pair runs/carrier_t1_s73 runs/unit_t1_s73 ...
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from typing import Callable, Dict, List, Optional, Sequence

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

CALLS_FILE = "unit_feedback_calls.json"
HALF = 0.5


def extract_tensor(broadcast_content) -> Optional[torch.Tensor]:
    """The tensor that the two rules read from a broadcast, or None. Same order of keys as the rules."""
    if isinstance(broadcast_content, torch.Tensor):
        return broadcast_content
    if isinstance(broadcast_content, dict):
        for key in ("_fused", "tensor"):
            if isinstance(broadcast_content.get(key), torch.Tensor):
                return broadcast_content[key]
    return None


def rescale_to_own_size(broadcast: torch.Tensor, own: torch.Tensor) -> torch.Tensor:
    """The broadcast, flattened and cut as the rules do, scaled to the size of the own vector."""
    flat, own_flat = broadcast.detach().reshape(-1), own.reshape(-1)
    length = min(flat.shape[0], own_flat.shape[0])
    flat, own_flat = flat[:length], own_flat[:length]
    return flat * (torch.norm(own_flat) / (torch.norm(flat) + 1e-8))


def unit_feedback(original: Callable, calls: Dict[str, int], name: str) -> Callable:
    """A rule that scales the broadcast and then calls the original rule of the module."""
    def receive_broadcast(self, broadcast_content, current_bid):
        tensor = extract_tensor(broadcast_content)
        if tensor is None or self._last_content is None:
            return original(self, broadcast_content, current_bid)
        calls[name] = calls.get(name, 0) + 1
        return original(self, rescale_to_own_size(tensor, self._last_content), current_bid)
    return receive_broadcast


def install(calls: Dict[str, int]) -> Callable[[], None]:
    """Replace the rule of both sense classes. Returns a function that puts the rules back."""
    from models.audio.auditory_specialist import AuditorySpecialist
    from models.core.sensory_tectum import SensoryTectum
    originals = {SensoryTectum: SensoryTectum.receive_broadcast, AuditorySpecialist: AuditorySpecialist.receive_broadcast}
    SensoryTectum.receive_broadcast = unit_feedback(originals[SensoryTectum], calls, "vision")
    AuditorySpecialist.receive_broadcast = unit_feedback(originals[AuditorySpecialist], calls, "audio")

    def remove() -> None:
        for module_class, rule in originals.items():
            module_class.receive_broadcast = rule
    return remove


def log_dir_of(arguments: Sequence[str]) -> str:
    """The value of --log-dir. The launcher stops without it, because the call counts need a folder."""
    for index, argument in enumerate(arguments):
        if argument == "--log-dir" and index + 1 < len(arguments):
            return arguments[index + 1]
        if argument.startswith("--log-dir="):
            return argument.split("=", 1)[1]
    raise SystemExit("run_unit_feedback needs --log-dir, so that the call counts are written beside the run")


def write_calls(log_dir: str, calls: Dict[str, int]) -> None:
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, CALLS_FILE), "w", encoding="utf-8") as handle:
        json.dump(dict(sorted(calls.items())), handle, indent=2)


def launch(arguments: List[str]) -> None:
    log_dir, calls = log_dir_of(arguments), {}
    install(calls)
    sys.argv = ["scripts.training.train_rlhf"] + arguments
    try:
        runpy.run_module("scripts.training.train_rlhf", run_name="__main__")
    finally:
        write_calls(log_dir, calls)


# ------------------------------------------------------------------------------------------ judge


def seed_reading(control_flip: Optional[float], unit_flip: Optional[float], calls: Dict[str, int]) -> Optional[bool]:
    """S1 for one seed. None when the intervention did not run or a share cannot be measured."""
    if min(calls.get("vision", 0), calls.get("audio", 0)) <= 0 or control_flip is None or unit_flip is None:
        return None
    return unit_flip < HALF * control_flip


def size_verdict(readings: List[Optional[bool]], calls: List[Dict[str, int]]) -> str:
    """VOID, SUPPORTED, NOT SUPPORTED or UNDECIDED, by the rule of the docstring."""
    if any(min(c.get("vision", 0), c.get("audio", 0)) <= 0 for c in calls):
        return "VOID"
    if all(reading is True for reading in readings):
        return "SUPPORTED"
    return "NOT SUPPORTED" if sum(1 for reading in readings if reading is False) >= 2 else "UNDECIDED"


def judge(arguments: List[str]) -> None:
    from scripts.analysis.probe_gate_b import load_steps
    from scripts.analysis.probe_reentrant_boost import _show, summarise
    parser = argparse.ArgumentParser(description="Judge the size hypothesis on pairs of runs.")
    parser.add_argument("--pair", nargs=2, action="append", required=True, metavar=("CONTROL_RUN", "UNIT_RUN"))
    parser.add_argument("--output-json")
    args = parser.parse_args(arguments)
    report, readings, all_calls = [], [], []
    for control_run, unit_run in args.pair:
        control, unit = summarise(load_steps(control_run)), summarise(load_steps(unit_run))
        with open(os.path.join(unit_run, CALLS_FILE), encoding="utf-8") as handle:
            calls = json.load(handle)
        reading = seed_reading(control["audio"]["flip_share"], unit["audio"]["flip_share"], calls)
        readings.append(reading)
        all_calls.append(calls)
        report.append({"control_run": control_run, "unit_run": unit_run, "control": control, "unit": unit, "calls": calls, "S1": reading})
        print("%s  against  %s   scaled calls %s" % (control_run, unit_run, calls))
        _show("control", control)
        _show("unit", unit)
        print("  S1 %s" % reading)
    print("size hypothesis: %s" % size_verdict(readings, all_calls))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump({"pairs": report, "verdict": size_verdict(readings, all_calls)}, handle, indent=2)


def main() -> None:
    arguments = sys.argv[1:]
    if arguments and arguments[0] == "judge":
        judge(arguments[1:])
    else:
        launch(arguments)


if __name__ == "__main__":
    main()
