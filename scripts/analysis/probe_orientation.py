"""Orientation test: does the agent turn to the object with the louder tone, binding on against off?

Read-only probe on runs made with `--dark-room-decoy-visible --dark-room-start-between`
(docs/decisions/2026_10_15_orientation_test.md). Each episode starts with the agent midway
between the light and the decoy, so at the start both tones come from the same distance and
the decoy level decides which one is louder (level below 1, the light; above 1, the decoy).

An episode is ORIENTED when, after K steps, the agent is closer to the louder object than to
the quieter one (from the `_truth_` offsets of the step record). Equal distances, or an
episode shorter than K steps, give no outcome. After the orientation test of Stein,
Meredith, Huneycutt and McDade (J. Cogn. Neurosci. 1989).

Usage:
    python -m scripts.analysis.probe_orientation --on runs/or_on_s194 ... --off runs/or_off_s194 ...
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import fisher_exact

K_STEPS = 10


def louder_at_start(level: float) -> Optional[str]:
    """The louder object when both are at the same distance: the decoy level decides."""
    if level > 1.0:
        return "decoy"
    if level < 1.0:
        return "light"
    return None


def oriented(steps: List[dict], k: int = K_STEPS) -> Optional[bool]:
    """True when the agent is closer to the louder object after k steps, False when closer to the other."""
    if len(steps) < k:
        return None
    first = steps[0]["internals"]["info_after_step"]
    louder = louder_at_start(float(first["_truth_decoy_level"]))
    info = steps[k - 1]["internals"]["info_after_step"]
    light = float(np.hypot(*info["_truth_light_offset"]))
    decoy = float(np.hypot(*info["_truth_decoy_offset"]))
    if louder is None or light == decoy:
        return None
    return (light < decoy) == (louder == "light")


def run_outcomes(run: str, k: int = K_STEPS) -> List[Optional[bool]]:
    """One outcome for each recorded episode of a run, in episode order."""
    outcomes = []
    for folder in sorted(glob.glob(os.path.join(run, "episodes", "ep_*"))):
        with open(os.path.join(folder, "steps.jsonl"), encoding="utf-8") as handle:
            steps = [json.loads(line) for _, line in zip(range(k), handle)]
        outcomes.append(oriented(steps, k))
    return outcomes


def _share(outcomes) -> tuple:
    kept = [o for o in outcomes if o is not None]
    return sum(kept), len(kept)


def compare(on: Dict[str, list], off: Dict[str, list]) -> dict:
    """Per seed, the oriented share on minus off; pooled shares and the one-sided Fisher p (on > off)."""
    per_seed, hits_on, n_on, hits_off, n_off = {}, 0, 0, 0, 0
    for seed in on:
        (h_on, c_on), (h_off, c_off) = _share(on[seed]), _share(off[seed])
        per_seed[seed] = h_on / c_on - h_off / c_off if c_on and c_off else None
        hits_on, n_on, hits_off, n_off = hits_on + h_on, n_on + c_on, hits_off + h_off, n_off + c_off
    p = fisher_exact([[hits_on, n_on - hits_on], [hits_off, n_off - hits_off]], alternative="greater")[1]
    return {"per_seed": per_seed, "pooled_on": hits_on / n_on if n_on else None,
            "pooled_off": hits_off / n_off if n_off else None, "episodes_on": n_on,
            "episodes_off": n_off, "p_one_sided": float(p)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--on", nargs="+", required=True)
    parser.add_argument("--off", nargs="+", required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    if len(args.on) != len(args.off):
        parser.error("--on and --off need one run each for the same seeds, in the same order")
    names = [os.path.basename(run) for run in args.on]
    on = {name: run_outcomes(run) for name, run in zip(names, args.on)}
    off = {name: run_outcomes(run) for name, run in zip(names, args.off)}
    result = compare(on, off)
    for name in names:
        print(name, "oriented on %d of %d, off %d of %d" % (*_share(on[name]), *_share(off[name])))
    print(json.dumps(result, indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
