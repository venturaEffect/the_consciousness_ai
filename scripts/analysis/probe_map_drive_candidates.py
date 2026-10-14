"""Compare the candidate drives of the map binding offline, on recorded episodes. Read-only.

`docs/decisions/2026_10_14_map_binding_drive.md` states the candidates, the control and the
choice rule. They were committed before this probe was first run.

DATA. The episodes of a run that have saved weights (`weights.pt`). The encoder of the tectum
is rebuilt from those weights and run on the recorded frames. The sound direction is the
recorded one. The weights are saved with the episode, so the features can differ a little
from the ones of the run. The control measures that.

CONTROL. K0 recomputed here must give the recorded group (the same cells) at 0.9 or more of
the steps. If not, the comparison is VOID.

STEPS. As in `probe_map_selection.load`. Step 0 of an episode is left out, the condition of
step t comes from the record of step t - 1, and the light cells from frame t.

Usage:
    python -m scripts.analysis.probe_map_drive_candidates --runs runs/s3b_s132 runs/s3b_s133 runs/s3b_s134 --decoy-levels 0.5,3.0
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from models.core.map_binding import HALF_MAXIMUM, LegionLimit, SegmentLimit, salience_drive  # noqa: E402
from models.core.retinotopic_encoder import RetinotopicEncoder  # noqa: E402
from scripts.analysis.probe_binding_selectivity import condition, episode_level  # noqa: E402
from scripts.analysis.probe_map_selection import light_cells, on_light  # noqa: E402

CONTROLS = ("K0", "K1")
PREFERENCE = ("K2", "K3")          # fewer assumed values first
REPRODUCTION, MULTI, ON_LIGHT, RUN_MINIMUM, MARGIN = 0.9, 0.5, 0.5, 50, 0.05


def candidates() -> dict:
    return {"K0": LegionLimit(), "K1": LegionLimit(threshold=HALF_MAXIMUM),
            "K2": LegionLimit(threshold=HALF_MAXIMUM, drive=salience_drive), "K3": SegmentLimit()}


def summarise(records) -> dict:
    """records: (kind, blocks, on_light) per step. Shares over steps with a group and over related steps."""
    grouped = [r for r in records if r[1] > 0]
    related = [r for r in records if r[0] == "related"]
    return {"steps": len(records), "with_group": len(grouped), "related": len(related),
            "multi": float(np.mean([r[1] >= 2 for r in grouped])) if grouped else None,
            "on_light": float(np.mean([r[2] for r in related])) if related else None}


def qualifies(pooled: dict, by_run: Dict[str, dict]) -> bool:
    """The three conditions of the choice rule."""
    if pooled["multi"] is None or pooled["multi"] < MULTI:
        return False
    if pooled["on_light"] is None or pooled["on_light"] <= ON_LIGHT:
        return False
    return all(s["on_light"] > ON_LIGHT for s in by_run.values() if s["related"] >= RUN_MINIMUM)


def choose(pooled_shares: Dict[str, float]) -> Optional[str]:
    """pooled_shares: share on the light of each qualifying candidate. Controls are never chosen."""
    shares = {name: share for name, share in pooled_shares.items() if name not in CONTROLS}
    if not shares:
        return None
    top = max(shares.values())
    return next(name for name in PREFERENCE if name in shares and shares[name] >= top - MARGIN)


def encoder_for(folder: str) -> Optional[RetinotopicEncoder]:
    path = os.path.join(folder, "weights.pt")
    if not os.path.exists(path):
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)["tectum"]
    prefix = "retinotopic_encoder."
    encoder = RetinotopicEncoder(out_channels=64, target_grid=16, pretrained=False)
    encoder.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
    return encoder.eval()


def by_step(record, name: str) -> dict:
    """{step: array} from a recorded vector and its `__steps` index. A step without an entry is absent."""
    return {int(step): value for step, value in zip(record[name + "__steps"], record[name])}


def episode_steps(folder: str, levels, offset: int, history: list, encoder) -> list:
    """(kind, light cells, recorded group, features, sound direction) for steps 1 and later."""
    steps = [json.loads(line) for line in open(os.path.join(folder, "steps.jsonl"), encoding="utf-8")]
    frames = np.load(os.path.join(folder, "frames.npz"))["frames"]
    sound = by_step(np.load(os.path.join(folder, "vectors.npz")), "audio_spatial")
    level = episode_level(int(folder.rsplit("_", 1)[1]), levels)
    rows = []
    with torch.no_grad():
        for t in range(1, len(steps)):
            info = steps[t - 1]["internals"]["info_after_step"]
            kind = condition(info, bool(info.get("_truth_light_in_view", False)), level)
            pixels = torch.from_numpy(frames[t]).float().div(255.0).permute(2, 0, 1)[None]
            rows.append((kind, light_cells(frames[t]), history[offset + t]["group"],
                         encoder(pixels), torch.from_numpy(sound[t]).float()[None]))
    return rows


def run_steps(run: str, levels) -> list:
    with open(os.path.join(run, "map_binding.json"), encoding="utf-8") as handle:
        history = json.load(handle)["history"]
    rows, offset = [], 0
    for folder in sorted(glob.glob(os.path.join(run, "episodes", "ep_*"))):
        encoder = encoder_for(folder)
        if encoder is not None:
            rows.extend(episode_steps(folder, levels, offset, history, encoder))
        offset += sum(1 for _ in open(os.path.join(folder, "steps.jsonl"), encoding="utf-8"))
    if offset != len(history):
        raise ValueError("the map record has %d entries and the step record %d" % (len(history), offset))
    return rows


def evaluate(rows: list) -> dict:
    """Per candidate: records (kind, blocks, on light), and for K0 the share that equals the recorded group."""
    forms = candidates()
    found = {name: [] for name in forms}
    same = []
    for kind, light, recorded, grid, sound in rows:
        for name, candidate in forms.items():
            _, record = candidate.observe(grid, sound)
            found[name].append((kind, record["blocks"], on_light(record["group"], light)))
            if name == "K0":
                same.append(record["group"] == recorded)
    return {"records": found, "reproduction": float(np.mean(same)) if same else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--decoy-levels", default="1.0")
    parser.add_argument("--output-json")
    args = parser.parse_args()
    levels = tuple(float(part) for part in args.decoy_levels.split(","))
    by_run, pooled_records, reproduction = {}, {name: [] for name in candidates()}, {}
    for run in args.runs:
        evaluated = evaluate(run_steps(run, levels))
        reproduction[run] = evaluated["reproduction"]
        by_run[run] = {name: summarise(records) for name, records in evaluated["records"].items()}
        for name, records in evaluated["records"].items():
            pooled_records[name].extend(records)
    report = {"reproduction": reproduction, "by_run": by_run,
              "pooled": {name: summarise(records) for name, records in pooled_records.items()}}
    shares = [value for value in reproduction.values() if value is not None]
    report["control_holds"] = bool(shares) and min(shares) >= REPRODUCTION
    report["qualifies"] = {name: qualifies(report["pooled"][name], {run: by_run[run][name] for run in by_run})
                           for name in pooled_records}
    report["chosen"] = choose({name: report["pooled"][name]["on_light"] for name, ok in report["qualifies"].items() if ok}) \
        if report["control_holds"] else None
    print(json.dumps(report, indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
