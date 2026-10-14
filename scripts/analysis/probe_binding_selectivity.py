"""Is the binding layer selective? Gate G1 of `docs/decisions/2026_10_14_oscillator_binding_design.md`.

Read-only probe on session records of dark room runs made with `--dark-room-decoy-sound`.

CONDITIONS. A step counts when the light was in view in the frame that the agent saw.
The sound that the agent heard at that step came from the light and from the second
source (the decoy). The step is RELATED when the light tone was the louder of the two at
the agent, so that vision and hearing report the same place. It is UNRELATED when the
decoy tone was louder, so that hearing reports another place. The loudness is
`binaural_amplitude` of the distance, from the ground truth of the step before.

MEASURE. The binding factor of a module is its bound bid over its bid before the
binding, and the code sets it to 1 + 0.5 * alignment. So the alignment of a module with
the mean field is 2 * (factor - 1). The joint alignment of a step is the mean of the
vision and the audio alignment. Steps with a vision or audio bid of 0 are left out. The
joint alignment is fitted on the five bids before the binding (least squares with an
intercept), and the residuals are compared. The statistic is the mean residual of
related steps minus that of unrelated steps. Its one-sided p value comes from 1000
shuffles of the condition labels (seed 0).

PREDICTIONS FOR THE PRESENT LAYER, written 2026-10-14 before this probe read a run
made with the decoy. Data: three fresh seeds, 96, 97 and 98, with the Gate B7 command
plus `--dark-room-decoy-sound`, on Python 3.12.10.
  B1, validity. Each run has 100 or more steps of each condition. If not, the test bed
      does not work and the verdict is VOID.
  B2. The present layer is not selective. The p value is 0.05 or more.
The reading "the present layer is not selective" is CONFIRMED when B2 holds at 2 or 3
of the 3 seeds, and REJECTED when it fails at 2 or 3. The same measure, with p below
0.05 at 3 of 3 fresh seeds, is the pass line of gate G1 for stage 2.

RESULT OF THE FIRST ATTEMPT, 2026-10-14: VOID. B1 failed at seeds 96 and 98 (7 and 48
unrelated steps). At decoy level 1.0 the light tone is the louder one at almost every
step with the light in view.

SECOND ATTEMPT, written 2026-10-14 before its runs. On the records of seeds 96 to 98 a
decoy level of 1.5 would give 114 to 252 unrelated and 199 to 573 related steps. That
choice was made on those seeds, so the second attempt uses fresh seeds 110, 111 and
112, with the same command plus `--dark-room-decoy-level 1.5`. B1, B2 and the rule are
unchanged.

RESULT OF THE SECOND ATTEMPT, 2026-10-14: VOID. B1 failed at seeds 111 and 112 (4
and 25 unrelated steps). A level chosen on earlier seeds did not carry over, because
where the agent goes changes with the seed.

THIRD ATTEMPT, written 2026-10-14 before its runs. The level changes at each reset of
the environment, 0.5 and 3.0 in turn (`--dark-room-decoy-level 0.5,3.0`), so that each
run has a quiet and a loud decoy. On the records of seeds 96 to 98, a level of 3.0 made
the decoy the louder source at 430 of 430, 451 of 451 and 681 of 687 in-view steps.
Fresh seeds 116, 117 and 118. B1, B2 and the rule are unchanged.

FAILED, AN ERROR IN THIS PROBE, found 2026-10-14 after the third attempt. `condition`
compared the two amplitudes from the distances only and left out the level of the
decoy. With a level other than 1.0 it counted the wrong steps. So attempts 2 and 3 were
judged on a wrong condition. The first attempt (level 1.0) is not affected. The probe now
multiplies the decoy amplitude by its level. The level comes from the step record
(`_truth_decoy_level`, recorded from 2026-10-14) or, for older records, from
`--decoy-levels` with the rule of `episode_level`. A new analysis of attempts 2 and 3
with the corrected probe is made after their data was seen, so it is not pre-stated.

With the corrected probe, attempts 2 and 3 stay VOID (seed 112: 52 unrelated steps;
seed 116: 43 unrelated; seed 118: 64 related). The time with the light in view changes
too much from seed to seed for 10 episodes.

FOURTH ATTEMPT, written 2026-10-14 before its runs, with the corrected probe. The same
command as the third attempt (`--dark-room-decoy-level 0.5,3.0`), but 20 episodes for
each run (`--episodes 20`). Fresh seeds 122, 123 and 124. B1, B2 and the rule are
unchanged.

This is a statement about the binding layer. It is not evidence about consciousness,
and it moves no indicator.

Usage:
    python -m scripts.analysis.probe_binding_selectivity --runs runs/decoy_s96 runs/decoy_s97 runs/decoy_s98
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from simulations.environments.audio_mixin import binaural_amplitude  # noqa: E402

MODULES = ("vision", "audio", "memory", "body", "semantic")
MIN_STEPS, P_LINE, SHUFFLES = 100, 0.05, 1000


def episode_level(episode: int, levels) -> float:
    """The decoy level of an episode for records without `_truth_decoy_level`.

    The training script resets the environment once while it seeds it, before episode
    0, and each reset takes the next level. So episode e has levels[(e + 1) % n].
    """
    return float(levels[(episode + 1) % len(levels)])


def condition(previous_info: dict, in_view: bool, level: float = 1.0) -> Optional[str]:
    """"related", "unrelated" or None, from the ground truth of the step before."""
    if not in_view or previous_info.get("_truth_decoy_offset") is None:
        return None
    level = float(previous_info.get("_truth_decoy_level", level))
    light = binaural_amplitude(float(np.hypot(*previous_info["_truth_light_offset"])))
    decoy = level * binaural_amplitude(float(np.hypot(*previous_info["_truth_decoy_offset"])))
    if light == decoy:
        return None
    return "related" if light > decoy else "unrelated"


def joint_alignment(record: dict) -> Optional[float]:
    """Mean alignment of vision and audio with the mean field. None when a sense bid is 0."""
    before, after = record["internals"]["modulated_bids"], record["bound_bids"]
    if before["vision"] <= 0.0 or before["audio"] <= 0.0:
        return None
    return float(np.mean([2.0 * (after[name] / before[name] - 1.0) for name in ("vision", "audio")]))


def residuals(bids: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Residuals of a least-squares fit of the target on the bids with an intercept."""
    design = np.column_stack([np.asarray(bids, dtype=np.float64), np.ones(len(target))])
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    return np.asarray(target, dtype=np.float64) - design @ coefficients


def shuffle_test(values: np.ndarray, related: np.ndarray, shuffles: int = SHUFFLES, seed: int = 0) -> Dict[str, float]:
    """Mean of the related values minus the unrelated ones, and its one-sided p value."""
    def difference(labels):
        return float(values[labels].mean() - values[~labels].mean())
    observed = difference(related)
    rng = np.random.default_rng(seed)
    above = sum(1 for _ in range(shuffles) if difference(rng.permutation(related)) >= observed)
    return {"difference": observed, "p": (above + 1) / (shuffles + 1)}


def load(run: str, levels=(1.0,)) -> List[dict]:
    """One row for every step with a condition and a joint alignment."""
    rows = []
    for folder in sorted(glob.glob(os.path.join(run, "episodes", "ep_*"))):
        level = episode_level(int(folder.rsplit("_", 1)[1]), levels)
        steps = [json.loads(line) for line in open(os.path.join(folder, "steps.jsonl"), encoding="utf-8")]
        for previous, current in zip(steps, steps[1:]):
            info = previous["internals"]["info_after_step"]
            kind = condition(info, bool(info.get("_truth_light_in_view", False)), level)
            joint = joint_alignment(current)
            if kind is not None and joint is not None:
                rows.append({"condition": kind, "joint": joint,
                             "bids": [current["internals"]["modulated_bids"][name] for name in MODULES]})
    return rows


def evaluate(rows: List[dict]) -> Dict[str, object]:
    related = np.array([row["condition"] == "related" for row in rows], dtype=bool)
    counts = {"related": int(related.sum()), "unrelated": int((~related).sum())}
    if min(counts.values()) < MIN_STEPS:
        return {**counts, "valid": False, "difference": None, "p": None}
    left = residuals(np.array([row["bids"] for row in rows]), np.array([row["joint"] for row in rows]))
    return {**counts, "valid": True, **shuffle_test(left, related)}


def verdict(seeds: List[Dict[str, object]]) -> str:
    """VOID, CONFIRMED or REJECTED for the reading that the present layer is not selective."""
    if not all(seed["valid"] for seed in seeds):
        return "VOID"
    return "CONFIRMED" if sum(1 for seed in seeds if seed["p"] >= P_LINE) >= 2 else "REJECTED"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--decoy-levels", default="1.0",
                        help="the --dark-room-decoy-level of the runs, for records without _truth_decoy_level")
    parser.add_argument("--output-json")
    args = parser.parse_args()
    levels = tuple(float(part) for part in args.decoy_levels.split(","))
    seeds = []
    for run in args.runs:
        seed = {"run": run, **evaluate(load(run, levels))}
        seeds.append(seed)
        print("%s related %d unrelated %d difference %s p %s" % (
            run, seed["related"], seed["unrelated"],
            "none" if seed["difference"] is None else "%+.5f" % seed["difference"],
            "none" if seed["p"] is None else "%.4f" % seed["p"]))
    print("present layer not selective: %s" % verdict(seeds))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump({"seeds": seeds, "verdict": verdict(seeds)}, handle, indent=2)


if __name__ == "__main__":
    main()
