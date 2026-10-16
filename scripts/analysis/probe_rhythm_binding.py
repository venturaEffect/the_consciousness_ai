"""Gate of binding by timing (docs/decisions/2026_10_15_rhythm_binding.md): G1 to G4.

Read-only probe on runs made with `--dark-room-pulse --dark-room-sound-direction off` and
`--map-binding rhythm` (or `rhythm-scrambled`), with the tectal motor map.

Alignment. Map binding entry t of an episode is the tectum's step t, which saw the frame after
environment step t - 1. So its truth is `info_after_step` of step record t - 1, and its image is
frame t. Entry 0 (no sound) is not counted.

Counted steps: t >= 12, the light and the decoy both in view. Object places: cell centres within
40 room pixels of the object's centre (geometry, because the colours pulse). The louder object at
a step: the larger of level x binaural_amplitude(distance), without the rhythm. Body places (centre
closer than 20 room pixels) are left out of every mean.

  G1: per episode, mean coherence on the louder object minus the other object, averaged over the
      counted steps; one-sided Wilcoxon over episodes (alternative: above 0).
  G2: per episode, that difference regressed on the difference in mean peak luminance; the
      intercept above 0, one-sided t-test.
  G3: first contact with the louder object within 60 steps (probe_motor_map), the rhythm arm
      against the scrambled arm and against the active arm.
  G4: share of counted steps with the group on the light in level-0.5 episodes, and on the decoy
      in level-3.0 episodes; each above 0.7.

Usage:
    python -m scripts.analysis.probe_rhythm_binding --rhythm runs/rb_rhythm_s211 ... \
        --scrambled runs/rb_scrambled_s211 ... --active runs/rb_active_s211 ...
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np
import torch
from scipy import stats

from models.core.rhythm_binding import cell_peak_luminance
from models.core.tectal_motor_map import body_mask, cell_offsets
from scripts.analysis.probe_map_selection import on_light
from scripts.analysis.probe_motor_map import compare_arms, first_contact_louder, split
from simulations.environments.audio_mixin import binaural_amplitude

START, CONTACT_STEPS, G4_SHARE = 12, 60, 0.7
_DX, _DY = cell_offsets()
_BODY = body_mask().reshape(-1)


def object_cells(offset) -> np.ndarray:
    """[256] bool: cell centres within 40 room pixels (the object radius) of the object's centre."""
    return (np.hypot(_DX - offset[0], _DY - offset[1]) < 40.0).reshape(-1)


def louder_now(info: dict) -> str:
    light = binaural_amplitude(float(np.hypot(*info["_truth_light_offset"])))
    decoy = float(info["_truth_decoy_level"]) * binaural_amplitude(float(np.hypot(*info["_truth_decoy_offset"])))
    return "light" if light >= decoy else "decoy"


def _mean_on(values: np.ndarray, mask: np.ndarray) -> float:
    kept = mask & ~_BODY
    return float(values[kept].mean()) if kept.any() else 0.0


def coherence_difference(rows: List[dict]) -> float:
    other = {"light": "decoy", "decoy": "light"}
    return float(np.mean([_mean_on(r["coherence"], r[r["louder"]]) - _mean_on(r["coherence"], r[other[r["louder"]]])
                          for r in rows]))


def luminance_difference(rows: List[dict]) -> float:
    other = {"light": "decoy", "decoy": "light"}
    return float(np.mean([_mean_on(r["peak"], r[r["louder"]]) - _mean_on(r["peak"], r[other[r["louder"]]])
                          for r in rows]))


def selection_share(rows: List[dict]) -> float:
    return float(np.mean([on_light(r["group"], r[r["louder"]].reshape(16, 16)) for r in rows]))


def g4_share(rows: List[dict], level: float, name: str) -> float:
    kept = [r for r in rows if r["level"] == level]
    return float(np.mean([on_light(r["group"], r[name].reshape(16, 16)) for r in kept])) if kept else float("nan")


def intercept_test(x, y) -> tuple:
    """(intercept, one-sided p for intercept > 0) of the least-squares line y = a + b x."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (intercept + slope * x)
    s2 = (residual ** 2).sum() / (n - 2)
    se = np.sqrt(s2 * (1.0 / n + x.mean() ** 2 / ((x - x.mean()) ** 2).sum()))
    return float(intercept), float(stats.t.sf(intercept / se, n - 2))


def verdict(holds: Dict[str, List[bool]]) -> str:
    """PASS when every gate holds at every seed; FAIL when one gate fails at 2 or more seeds."""
    if all(all(seeds) for seeds in holds.values()):
        return "PASS"
    if any(sum(1 for ok in seeds if not ok) >= 2 for seeds in holds.values()):
        return "FAIL"
    return "UNDECIDED"


def _episodes(run: str) -> List[List[dict]]:
    return [[json.loads(line) for line in open(os.path.join(folder, "steps.jsonl"), encoding="utf-8")]
            for folder in sorted(glob.glob(os.path.join(run, "episodes", "ep_*")))]


def episode_rows(steps: List[dict], history: List[dict], frames: np.ndarray) -> List[dict]:
    rows = []
    for t in range(START, len(steps)):
        info = steps[t - 1]["internals"]["info_after_step"]
        if not (info.get("_truth_light_in_view") and info.get("_truth_decoy_in_view")):
            continue
        image = torch.from_numpy(frames[t]).float().div(255.0).permute(2, 0, 1)[None]
        rows.append({"coherence": np.asarray(history[t]["coherence"], dtype=float),
                     "peak": cell_peak_luminance(image).reshape(-1), "group": history[t]["group"],
                     "light": object_cells(info["_truth_light_offset"]),
                     "decoy": object_cells(info["_truth_decoy_offset"]),
                     "louder": louder_now(info), "level": float(info["_truth_decoy_level"])})
    return rows


def summarise_rhythm(run: str) -> dict:
    with open(os.path.join(run, "map_binding.json"), encoding="utf-8") as handle:
        history = json.load(handle)["history"]
    episodes = _episodes(run)
    parts = split(history, [len(steps) for steps in episodes])
    folders = sorted(glob.glob(os.path.join(run, "episodes", "ep_*")))
    per_episode, all_rows = [], []
    for steps, part, folder in zip(episodes, parts, folders):
        rows = episode_rows(steps, part, np.load(os.path.join(folder, "frames.npz"))["frames"])
        if rows:
            per_episode.append((coherence_difference(rows), luminance_difference(rows)))
            all_rows.extend(rows)
    diffs = np.array([d for d, _ in per_episode])
    lums = np.array([lum for _, lum in per_episode])
    g1_p = float(stats.wilcoxon(diffs, alternative="greater").pvalue) if len(diffs) else float("nan")
    intercept, g2_p = intercept_test(lums, diffs) if len(diffs) > 2 else (float("nan"), float("nan"))
    return {"episodes_counted": len(per_episode), "counted_steps": len(all_rows),
            "g1_mean_difference": float(diffs.mean()) if len(diffs) else None, "g1_p": g1_p,
            "g2_intercept": intercept, "g2_p": g2_p,
            "g4_light_at_half": g4_share(all_rows, 0.5, "light"),
            "g4_decoy_at_three": g4_share(all_rows, 3.0, "decoy"),
            "selection_on_louder": selection_share(all_rows) if all_rows else None}


def contact_outcomes(run: str) -> list:
    return [first_contact_louder(steps, limit=CONTACT_STEPS) for steps in _episodes(run)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rhythm", nargs="+", required=True)
    parser.add_argument("--scrambled", nargs="+", required=True)
    parser.add_argument("--active", nargs="+", required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    if not len(args.rhythm) == len(args.scrambled) == len(args.active):
        parser.error("each arm needs one run for each seed, in the same order")
    names = [os.path.basename(run) for run in args.rhythm]
    result = {"rhythm": {}, "contact": {}}
    holds = {"G1": [], "G2": [], "G3": [], "G4": []}
    for name, run in zip(names, args.rhythm):
        values = summarise_rhythm(run)
        result["rhythm"][name] = values
        print(name, values)
        holds["G1"].append(values["g1_mean_difference"] is not None and values["g1_p"] < 0.05)
        holds["G2"].append(values["g2_intercept"] > 0 and values["g2_p"] < 0.05)
        holds["G4"].append(values["g4_light_at_half"] > G4_SHARE and values["g4_decoy_at_three"] > G4_SHARE)
    outcomes = {arm: {n: contact_outcomes(r) for n, r in zip(names, getattr(args, arm))}
                for arm in ("rhythm", "scrambled", "active")}
    for arm, per_seed in outcomes.items():
        counts = {n: "%d of %d" % (sum(1 for o in v if o), sum(1 for o in v if o is not None)) for n, v in per_seed.items()}
        result["contact"][arm] = counts
        print("first contact with the louder object,", arm, counts)
    versus = {base: compare_arms(outcomes["rhythm"], outcomes[base]) for base in ("scrambled", "active")}
    result["comparison"] = versus
    for base, comparison in versus.items():
        print("rhythm against", base, comparison)
    pooled_ok = all(c["p_one_sided"] < 0.05 for c in versus.values())
    holds["G3"] = [pooled_ok and all((c["per_seed"][n] or 0.0) > 0.0 for c in versus.values()) for n in names]
    result["holds"], result["verdict"] = holds, verdict(holds)
    print("holds:", holds)
    print("VERDICT:", result["verdict"])
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
