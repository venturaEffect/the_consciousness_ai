"""Closed stimulus set: do the earlier thermodynamic readouts generalise to an unseen stimulus?

Why this probe exists. DMTS draws its sample from 72 stimuli (6 shapes x 6 colours x 2 sizes). In the cached recordings
obs_map has 60 distinct rows in 120 trials, and workspace_broadcast has 54 to 62 distinct rows in 600. The earlier
thermodynamic probes (fidelity, encoding, matched readout, transform cost, frontier) cross validated with folds grouped by
TRIAL. A trial fold does not keep a stimulus out of training, so their test rows repeat training inputs exactly for those
two streams. Their clean accuracies therefore measure separation of a known 72-item set. They do not show that the class
is read from an unseen stimulus. This probe repeats the reference readout and the 64-axis equalised p-bit curve with folds
that hold out whole stimuli. It is a software emulation. It says nothing about hardware, consciousness or Phi.

DEFINITIONS, fixed before the run.
    stimulus     the frame shown in the sample phase. obs_map is computed from the current frame only, so its distinct rows
                 identify the stimulus. Each trial gets the id of its obs_map row.
    trial folds  5 folds grouped by trial, as in the earlier probes.
    stimulus folds  5 folds grouped by stimulus id, so no test input occurs in training. The penalty is chosen by inner folds
                 grouped the same way.
    arms         plain  x = sigmoid(z), the tuned ridge readout of the fidelity probe, clean vectors.
                 full64 the top 64 training principal axes, variance-equalised (the frontier probe), clean and through the
                 p-bit channel at LADDER with a readout trained on the channel's own outputs.
    chance       1/6, and the majority-class share per model, because the class counts are unequal.
DATA. The 30 cached recordings (models 42, 43, 44; stimulus draws 300 to 309; 1200 trials per model). Nothing new is recorded.

PREDICTIONS, written before the run, with no thresholds.
  S1. Clean accuracy falls under stimulus folds for obs_map and workspace_broadcast, whose rows are functions of the stimulus.
  S2. tectum_content and z_state, which also depend on the recurrent state, change less.
  S3. If obs_map stays well above chance under stimulus folds, the class is read across colour and size and the earlier
      noise results carry over. If it falls to chance, the earlier results describe a lookup of 72 items.

WHAT IS REPORTED, fixed before the run, no pass line.
  1. Distinct rows per stream and recording, and distinct stimuli per model.
  2. Clean accuracy of plain and full64 with trial folds and with stimulus folds, the paired difference, and 95% intervals from
     a bootstrap that resamples STIMULI (clusters), not trials, because trials of one stimulus are not independent.
  3. The majority-class share per model.
  4. With stimulus folds: accuracy of full64 at each dose, rho* against the plain clean accuracy of the same fold scheme, and
     T90 and T95. These are reported only where the lower bound of that clean accuracy exceeds 1/6 + 0.05. Otherwise the
     stream is NOT ESTIMABLE, because a ratio of two values near chance has no meaning.
Nothing is pooled across models.

PILOT NOTE. A pilot on one new recording (model 42, draw 999, not part of the data, discarded) ran in 37 s. With stimulus folds
the plain clean accuracy fell from 0.78 to 0.09 for obs_map, from 0.45 to 0.06 for tectum_content and from 0.78 to 0.23 for
z_state. workspace_broadcast fell from 0.77 to 0.58. This contradicts S2 on one recording. The pilot printed budget values for
streams whose clean readout was at chance, which is meaningless, so the estimable rule of item 4 was added before the data run.
Nothing else was changed.

Usage:
    python -m scripts.analysis.probe_thermodynamic_stimulus evaluate
    python -m scripts.analysis.probe_thermodynamic_stimulus summarise --output-json runs/thermo_fidelity/stimulus_summary.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.probe_thermodynamic_cost import Recording, make_folds, normalise  # noqa: E402
from scripts.analysis.probe_thermodynamic_encoding import binomial_estimates, score_trials  # noqa: E402
from scripts.analysis.probe_thermodynamic_fidelity import CHANCE, LADDER, REPEATS, Readout, estimable, first_dose  # noqa: E402
from scripts.analysis.probe_thermodynamic_frontier import encode_rank  # noqa: E402
from scripts.analysis.probe_thermodynamic_matched import COPIES, select_penalty_fast  # noqa: E402

SCHEMES = ("trial", "stimulus")
AXES = 64
BOOTSTRAPS = 2000
STATUS = "UNPROVEN"


# ------------------------------------------------------------------------------- stimulus ids


def stimulus_ids(obs_map: torch.Tensor) -> torch.Tensor:
    """Id of each row from the distinct rows of obs_map. Equal rows get the same id."""
    return torch.unique(obs_map, dim=0, return_inverse=True)[1]


def stimulus_key(row: torch.Tensor) -> str:
    """A key that is the same for the same obs_map row in any recording of one model."""
    return hashlib.md5(row.detach().cpu().numpy().astype(np.float32).tobytes()).hexdigest()[:12]


def distinct_rows(stream: torch.Tensor) -> int:
    return int(torch.unique(stream, dim=0).shape[0])


# ------------------------------------------------------------------------------ one recording


def evaluate_scheme(z: torch.Tensor, labels: torch.Tensor, trials: torch.Tensor, groups: torch.Tensor, seed: int,
                    scheme: str, hits: Dict[str, Dict[int, List[float]]]) -> None:
    """Score plain and full64 (clean and matched p-bit) with folds grouped by `groups`."""
    for fold, held in enumerate(make_folds(groups, seed)):
        ztr, zte = normalise(z[~held], z[held])
        ytr, gtr = labels[~held], groups[~held]
        xtr, xte = torch.sigmoid(ztr), torch.sigmoid(zte)
        plain = Readout(xtr, ytr, select_penalty_fast(xtr, ytr, gtr, seed + fold))
        score_trials(hits, f"{scheme}|plain|clean", plain.predict(xte), held, labels, trials)
        utr, ute = encode_rank(ztr, zte, 1.0, AXES)
        score_full64(hits, scheme, (torch.sigmoid(utr), torch.sigmoid(ute)), (ytr, gtr), held, (labels, trials),
                     100000 * seed + 1000 * fold + (0 if scheme == "trial" else 500), seed + fold)


def score_full64(hits, scheme, features, train, held, everything, base, seed) -> None:
    """Clean and matched-trained p-bit accuracy of the 64-axis equalised encoding for one fold."""
    xtr, xte = features
    ytr, gtr = train
    labels, trials = everything
    clean = Readout(xtr, ytr, select_penalty_fast(xtr, ytr, gtr, seed))
    score_trials(hits, f"{scheme}|full64|clean", clean.predict(xte), held, labels, trials)
    tests = [binomial_estimates(xte, LADDER, torch.Generator().manual_seed(base + 3 * r), "pbit") for r in range(REPEATS)]
    copies = [binomial_estimates(xtr, LADDER, torch.Generator().manual_seed(base + 10 + c), "pbit") for c in range(COPIES)]
    for i, dose in enumerate(LADDER):
        x_noisy = torch.cat([c[i] for c in copies])
        readout = Readout(x_noisy, ytr.repeat(COPIES), select_penalty_fast(x_noisy, ytr.repeat(COPIES), gtr.repeat(COPIES), seed))
        for r in range(REPEATS):
            score_trials(hits, f"{scheme}|full64|matched|{dose}", readout.predict(tests[r][i]), held, labels, trials)


def evaluate_recording(rec: Recording, seed: int) -> Dict[str, object]:
    ids = stimulus_ids(rec.streams["obs_map"])
    order = sorted(int(t) for t in rec.trials.unique())
    first_row = {t: int((rec.trials == t).nonzero()[0]) for t in order}
    out: Dict[str, object] = {"trial_labels": [int(rec.labels[first_row[t]]) for t in order],
                              "trial_stimulus": [stimulus_key(rec.streams["obs_map"][first_row[t]]) for t in order],
                              "distinct_rows": {name: distinct_rows(z) for name, z in rec.streams.items()}, "streams": {}}
    for name, z in rec.streams.items():
        hits: Dict[str, Dict[int, List[float]]] = {}
        evaluate_scheme(z, rec.labels, rec.trials, rec.trials, seed, "trial", hits)
        evaluate_scheme(z, rec.labels, rec.trials, ids, seed, "stimulus", hits)
        out["streams"][name] = {key: [float(np.mean(c[t])) for t in sorted(c)] for key, c in hits.items()}
    return out


# ------------------------------------------------------------------------------- statistics


def cluster_weights(clusters: np.ndarray, rng: np.random.Generator, n: int = BOOTSTRAPS) -> np.ndarray:
    """Bootstrap weights per trial, [n, trials], from resampling whole clusters with replacement."""
    _, index = np.unique(clusters, return_inverse=True)
    k = index.max() + 1
    counts = rng.multinomial(k, np.full(k, 1.0 / k), size=n)
    return counts[:, index].astype(float)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return (weights * values).sum(1) / weights.sum(1)


def interval(values: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
    boot = weighted_mean(values, weights)
    return {"mean": float(values.mean()), "low": float(np.quantile(boot, 0.025)), "high": float(np.quantile(boot, 0.975))}


def majority_share(labels: np.ndarray) -> float:
    return float(np.bincount(labels).max() / len(labels))


def summarise_stream(per_trial: Dict[str, np.ndarray], weights: np.ndarray) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for arm in ("plain", "full64"):
        trial, stim = per_trial[f"trial|{arm}|clean"], per_trial[f"stimulus|{arm}|clean"]
        out[arm] = {"trial_folds": interval(trial, weights), "stimulus_folds": interval(stim, weights),
                    "difference": interval(stim - trial, weights)}
    clean = per_trial["stimulus|plain|clean"]
    denom = weighted_mean(clean, weights) - CHANCE
    rows = []
    for dose in LADDER:
        a = per_trial[f"stimulus|full64|matched|{dose}"]
        with np.errstate(divide="ignore", invalid="ignore"):
            rho = (weighted_mean(a, weights) - CHANCE) / denom
        rho = rho[np.isfinite(rho)]
        point = (float(a.mean()) - CHANCE) / (float(clean.mean()) - CHANCE) if clean.mean() != CHANCE else float("nan")
        rows.append({"accuracy": float(a.mean()), "rho": point,
                     "rho_low": float(np.quantile(rho, 0.025)) if len(rho) else float("nan"),
                     "rho_high": float(np.quantile(rho, 0.975)) if len(rho) else float("nan")})
    out["curve"] = rows
    out["estimable"] = estimable(out["plain"]["stimulus_folds"]["low"])
    reach = out["estimable"]
    out["T90"] = first_dose(rows, 0.90, "rho") if reach else None
    out["T95"] = first_dose(rows, 0.95, "rho") if reach else None
    out["T95_lower"] = first_dose(rows, 0.95, "rho_low") if reach else None
    return out


# ------------------------------------------------------------------------------------------ IO


def stimulus_path(out_dir: Path, model: int, draw: int) -> Path:
    return out_dir / f"sti_m{model}_s{draw}.json"


def command_evaluate(args: argparse.Namespace) -> None:
    out_dir, started = Path(args.out_dir), time.time()
    for path in sorted(out_dir.glob("rec_m*_s*.pt")):
        model, draw = [int(p[1:]) for p in path.stem[4:].split("_")]
        target = stimulus_path(out_dir, model, draw)
        if target.exists():
            continue
        data = torch.load(path, weights_only=False)
        rec = Recording(data["streams"], data["labels"], data["trials"])
        target.write_text(json.dumps(evaluate_recording(rec, seed=1000 * model + draw)))
        print(f"[{time.time() - started:7.0f} s] stimulus folds model {model} draw {draw}", file=sys.stderr, flush=True)


def load_model(out_dir: Path, model: int) -> Dict[str, object]:
    """Pooled per-trial arrays, labels, stimulus keys and distinct-row counts for one model."""
    pooled: Dict[str, Dict[str, List[float]]] = {}
    labels, keys, rows = [], [], {}
    for path in sorted(out_dir.glob(f"sti_m{model}_s*.json")):
        res = json.loads(path.read_text())
        labels.extend(res["trial_labels"])
        keys.extend(res["trial_stimulus"])
        for stream, count in res["distinct_rows"].items():
            rows.setdefault(stream, []).append(count)
        for stream, conds in res["streams"].items():
            for name, values in conds.items():
                pooled.setdefault(stream, {}).setdefault(name, []).extend(values)
    arrays = {s: {c: np.array(v) for c, v in conds.items()} for s, conds in pooled.items()}
    return {"arrays": arrays, "labels": np.array(labels), "keys": np.array(keys), "rows": rows}


def command_summarise(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    models = sorted({int(p.stem.split("_")[1][1:]) for p in out_dir.glob("sti_m*_s*.json")})
    rng = np.random.default_rng(0)
    summary = {"status": STATUS, "ladder": list(LADDER), "models": {}}
    for model in models:
        data = load_model(out_dir, model)
        weights = cluster_weights(data["keys"], rng)
        summary["models"][str(model)] = {
            "trials": int(len(data["labels"])), "distinct_stimuli": int(len(set(data["keys"]))),
            "majority_share": majority_share(data["labels"]),
            "distinct_rows_per_recording": {s: [int(min(v)), int(max(v))] for s, v in data["rows"].items()},
            "streams": {s: summarise_stream(a, weights) for s, a in data["arrays"].items()}}
    print_summary(summary)
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2))


def print_summary(summary: Dict[str, object]) -> None:
    print(f"closed stimulus set, instruments {summary['status']}")
    for model, m in summary["models"].items():
        print(f"model {model}: trials {m['trials']}, distinct stimuli {m['distinct_stimuli']}, majority share {m['majority_share']:.3f}, "
              f"distinct rows per recording {m['distinct_rows_per_recording']}")
        for stream, e in m["streams"].items():
            for arm in ("plain", "full64"):
                t, s, d = e[arm]["trial_folds"], e[arm]["stimulus_folds"], e[arm]["difference"]
                print(f"   {stream:20s} {arm:6s} trial folds {t['mean']:.3f}  stimulus folds {s['mean']:.3f} ({s['low']:.3f} to {s['high']:.3f})  "
                      f"difference {d['mean']:+.3f} ({d['low']:+.3f} to {d['high']:+.3f})")
            acc = " ".join(f"{r['accuracy']:.2f}" for r in e["curve"])
            budget = f"T90 {e['T90']}  T95 {e['T95']} (lower {e['T95_lower']})" if e["estimable"] else "NOT ESTIMABLE (clean readout at chance)"
            print(f"   {stream:20s} full64 p-bit, stimulus folds A(T) {acc}  {budget}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("evaluate", "summarise"):
        p = sub.add_parser(name)
        p.add_argument("--out-dir", default="runs/thermo_fidelity")
        if name == "summarise":
            p.add_argument("--output-json")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    torch.set_num_threads(1)
    {"evaluate": command_evaluate, "summarise": command_summarise}[args.command](args)


if __name__ == "__main__":
    main()
