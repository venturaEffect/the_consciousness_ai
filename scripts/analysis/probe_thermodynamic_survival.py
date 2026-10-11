"""Do the architecture's verified properties survive a stochastic p-bit channel, measured the same way?

The verified property. scripts/analysis/probe_collapse_locus.py records one vector per DMTS trial (the last sample step) from
obs_map, z_state, capsule_poses and tectum_content, and decodes sample shape and colour with a linear probe and a PCA-80 + MLP
probe on one random 70/30 split. On runs/capfix_alllevels it gives obs_map 0.988 / 0.976, z_state 0.988 / 0.762 and
tectum_content 0.524 / 0.833 for shape (linear / pca+mlp). This probe keeps that instrument and puts a channel in front of it.
It adds nothing else. It is a software emulation. It says nothing about hardware, about consciousness or about Phi, and it does
not test section 13 of docs/preregistered_predictions.md.

THE CHANNEL. Each recorded vector is encoded, sent through independent p-bits (T samples per dimension, the exact binomial
law of probe_thermodynamic_encoding.binomial_estimates) and decoded to an estimate of the encoded value. The decoder of the
project is then fitted and tested on the channel outputs. Two encodings, both fitted on the training rows of the split:
    plain   every raw dimension z-scored, then sigmoid. Events per decision: (raw dimension) x T.
    full64  the top 64 principal axes, variance-equalised (probe_thermodynamic_frontier.encode_rank), then sigmoid.
            Events per decision: 64 x T.
Doses T: 16, 64, 256, 1024, 4096. "clean" is the encoding with no channel. "raw" is the project's own input, untouched.

THREE READINGS, reported side by side. DMTS has 72 stimuli and every presentation is pixel-identical, so they differ.
    R1 project   the project's split (train_test_split, 30% test, stratified, seeded) on shape and colour. This is "measured
                 the same way". Test stimuli also occur in training, and the output says so.
    R2 identity  the 72-way label (shape, colour, size), chance 1/72, same kind of split by trial. Test rows whose stimulus has no
                 training row are dropped and counted. This is what match-to-sample needs: a known stimulus kept apart from the others.
    R3 unseen    shape and colour with whole stimuli held out (one seeded 70/30 split of the stimuli).
The classifiers are the project's two pipelines in every reading: StandardScaler + LogisticRegression(max_iter=1000), and
StandardScaler + PCA(80) + MLPClassifier((64,), max_iter=2000). For R1 on raw input the result is checked against the
project's own linear_decode and pca_mlp_decode, and the probe raises on any difference.

SENSORY-NOISE ARM. The same on recordings made with DMTSEnv(sample_noise=8), where two presentations of a stimulus are no
longer pixel-identical. It separates input noise from substrate noise. The trial sequence is unchanged by the noise level.

DATA. Checkpoints capfix_alllevels, capfix_seed43, capfix_seed44. 14 episodes of 20 trials, probe seed 42, as the project's
probe (n = 280). Nothing is pooled across checkpoints. Intervals come from a bootstrap that resamples test STIMULI.

PREDICTIONS, written before the data run, with no thresholds.
  V1. R1 on raw input reproduces the documented numbers for capfix_alllevels. (Checked in Step 1 with the project's code.)
  V2. R1 survives the channel for obs_map at a moderate dose with full64, and needs a higher dose for z_state. tectum_content
      shape with the linear probe (0.524 clean) has little to lose and is not informative.
  V3. R2 is near 1.0 on raw obs_map, because equal frames give equal vectors. It is lower for z_state and tectum_content, which
      also depend on the recurrent state.
  V4. R3 is at or below chance for shape on obs_map, z_state and tectum_content, as in the closed-stimulus study. Colour may
      generalise better than shape.
  V5. Sensory noise lowers every reading a little and does not change their order.
What would count against my expectations: R1 collapsing at every dose, R2 far below 1.0 on raw obs_map, or R3 well above chance.

COST LIMITS, fixed before the data run from a pilot on a discarded configuration (model 42, probe seed 999, 3 episodes,
60 trials). One linear fit on the 262144-D z_state took 22 to 35 s at 60 trials and takes minutes at 280. So:
    z_state, plain encoding, linear probe: reading R1 only, arms clean, 64 and 1024 only. Every other stream, reading, arm and
    classifier runs in full. The raw z_state linear fit runs for every reading.
    Sensory-noise recordings: raw and full64 only. The plain encoding is not run on them.
Nothing else was changed after the pilot. The pilot numbers are not reported, because 60 trials are too few.

WHAT IS REPORTED, fixed before the run, no pass line. For each checkpoint, noise arm, stream, reading, label and classifier:
accuracy on raw, clean and at each dose, with a 95% interval from the stimulus bootstrap; the smallest dose that reaches 90%
and 95% of the clean above-chance accuracy (only where the clean lower bound is above chance + 0.05); events per decision; and the
audit (rows, distinct stimuli, share of test rows whose stimulus is in training, class counts).

Usage:
    python -m scripts.analysis.probe_thermodynamic_survival record
    python -m scripts.analysis.probe_thermodynamic_survival evaluate
    python -m scripts.analysis.probe_thermodynamic_survival summarise --output-json runs/thermo_survival/summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.probe_thermodynamic_cost import normalise  # noqa: E402
from scripts.analysis.probe_thermodynamic_encoding import binomial_estimates  # noqa: E402
from scripts.analysis.probe_thermodynamic_frontier import encode_rank  # noqa: E402
from scripts.analysis.probe_thermodynamic_stimulus import cluster_weights, interval  # noqa: E402

CHECKPOINTS = {42: "runs/capfix_alllevels/tectum.pt", 43: "runs/capfix_seed43/tectum.pt", 44: "runs/capfix_seed44/tectum.pt"}
STREAMS = ("obs_map", "z_state", "tectum_content")
DOSES = (16, 64, 256, 1024, 4096)
ENCODINGS = ("plain", "full64")
CLASSIFIERS = ("linear", "pca_mlp")
NOISE_ARMS = (0.0, 8.0)
EPISODES, PROBE_SEED, TEST_SIZE, AXES = 14, 42, 0.3, 64
STATUS = "UNPROVEN"


# ---------------------------------------------------------------------------------- recording


def collect_records(checkpoint: str, episodes: int, seed: int, sample_noise: float) -> Dict[str, np.ndarray]:
    """The loop of probe_collapse_locus.collect, plus the size label and an optional sensory noise level."""
    from scripts.analysis.probe_perception_decodability import _build_components, frame_to_tensor
    from simulations.environments.dmts_env import DMTSEnv

    cfg, tectum, *_ = _build_components("dmts", action_dim=5, seed=seed, mock_semantic=False, load_tectum=checkpoint,
                                        latent_mode="continuous", capsule_workspace_source="all_levels")
    env = DMTSEnv(num_trials=20, sample_noise=sample_noise)
    taps: Dict[str, list] = {s: [] for s in STREAMS}
    labels: Dict[str, list] = {"shape": [], "color": [], "size": []}
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            tectum.reset_state(1)
            current, done, steps = None, False, 0
            while not done and steps < 4000:
                phase = info.get("phase")
                frame = frame_to_tensor(obs, cfg["device"])
                content, _ = tectum(frame, torch.zeros(1, cfg["tectum_feature_dim"], 2, device=cfg["device"]))
                if phase == "sample":
                    current = _tap(tectum, content, info)
                elif phase == "choice" and current is not None:
                    _commit(taps, labels, current)
                    current = None
                action = 0 if phase != "choice" else int(rng.integers(1, env.num_choices + 1))
                obs, _, term, trunc, info = env.step(action)
                done, steps = term or trunc, steps + 1
    out = {s: np.stack(v).astype(np.float32) for s, v in taps.items()}
    out.update({k: np.array(v) for k, v in labels.items()})
    return out


def _tap(tectum, content, info) -> Dict[str, object]:
    return {"obs_map": tectum._last_obs_map.reshape(-1).cpu().numpy().astype(np.float32),
            "z_state": tectum.z_state.reshape(-1).cpu().numpy().astype(np.float32),
            "tectum_content": content.reshape(-1).cpu().numpy().astype(np.float32),
            "shape": info["sample_shape"], "color": info["sample_color"], "size": info["sample_size"]}


def _commit(taps, labels, current) -> None:
    for stream in taps:
        taps[stream].append(current[stream])
    for name in labels:
        labels[name].append(current[name])


def identity_labels(records: Dict[str, np.ndarray]) -> np.ndarray:
    """The 72-way stimulus label, shape|colour|size."""
    return np.array([f"{a}|{b}|{c}" for a, b, c in zip(records["shape"], records["color"], records["size"])])


# ------------------------------------------------------------------------------------- splits


def project_split(y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """The split of the project's decoders: 30% test, stratified when every class has two rows."""
    _, index = np.unique(y, return_inverse=True)
    stratify = index if np.bincount(index).min() >= 2 else None
    return train_test_split(np.arange(len(y)), test_size=TEST_SIZE, random_state=seed, stratify=stratify)


def identity_split(identity: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """A trial split for the 72-way label. Test rows whose stimulus has no training row are dropped and counted."""
    train, test = project_split(identity, seed)
    known = np.isin(identity[test], identity[train])
    return train, test[known], int((~known).sum())


def stimulus_split(identity: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Whole stimuli held out: 30% of the stimuli go to the test side."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    return next(splitter.split(np.zeros(len(identity)), groups=identity))


def seen_share(identity: np.ndarray, train: np.ndarray, test: np.ndarray) -> float:
    """Share of test rows whose stimulus also occurs in training."""
    return float(np.isin(identity[test], identity[train]).mean()) if len(test) else float("nan")


# ----------------------------------------------------------------------------------- decoders


def build_classifier(kind: str, seed: int, n_train: int, n_features: int):
    """The project's two pipelines (probe_perception_decodability.linear_decode and probe_collapse_locus.pca_mlp_decode)."""
    if kind == "linear":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    k = min(80, n_train - 1, n_features)
    return make_pipeline(StandardScaler(), PCA(n_components=k, random_state=seed),
                         MLPClassifier(hidden_layer_sizes=(64,), max_iter=2000, random_state=seed))


def decode_rows(X: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray, kind: str, seed: int) -> np.ndarray:
    """Correctness of every test row. X is float64 for the linear probe and float32 for pca+mlp, as in the project."""
    if kind == "linear":
        features, target = np.asarray(X, dtype=np.float64), np.unique(y, return_inverse=True)[1]
    else:
        features, target = np.asarray(X, dtype=np.float32), np.asarray(y)
    clf = build_classifier(kind, seed, len(train), features.shape[1])
    clf.fit(features[train], target[train])
    return clf.predict(features[test]) == target[test]


def check_against_project(X: np.ndarray, y: np.ndarray, seed: int) -> None:
    """Raise if decode_rows on the project's split differs from the project's own decoders."""
    from scripts.analysis.probe_collapse_locus import pca_mlp_decode
    from scripts.analysis.probe_perception_decodability import linear_decode

    train, test = project_split(y, seed)
    mine = {k: float(decode_rows(X, y, train, test, k, seed).mean()) for k in CLASSIFIERS}
    theirs = {"linear": float(linear_decode(X, y, seed=seed)["test_acc"]), "pca_mlp": float(pca_mlp_decode(X, y, seed=seed)[0])}
    for kind in CLASSIFIERS:
        if abs(mine[kind] - theirs[kind]) > 1e-9:
            raise RuntimeError(f"{kind}: this probe gives {mine[kind]:.6f}, the project's decoder gives {theirs[kind]:.6f}")


# ----------------------------------------------------------------------------------- encodings


def encode(kind: str, X: np.ndarray, train: np.ndarray) -> torch.Tensor:
    """Encode every row with statistics fitted on the training rows only. Returns values in (0, 1)."""
    data = torch.from_numpy(np.asarray(X, dtype=np.float32))
    if kind == "plain":
        mean, scale = data[train].mean(0), data[train].std(0).clamp_min(1e-8)
        return torch.sigmoid((data - mean) / scale)
    ztr, zall = normalise(data[train], data)
    _, uall = encode_rank(ztr, zall, 1.0, AXES)
    return torch.sigmoid(uall).float()


def channel(x: torch.Tensor, dose: int, seed: int) -> np.ndarray:
    """One p-bit channel use per row and dimension at the given dose, decoded to an estimate of x."""
    generator = torch.Generator().manual_seed(seed)
    return binomial_estimates(x, (dose,), generator, "pbit")[0].numpy()


def events(kind: str, raw_dim: int, dose: int) -> int:
    return (raw_dim if kind == "plain" else AXES) * dose


# ------------------------------------------------------------------------------ one recording


def readings(records: Dict[str, np.ndarray], seed: int) -> Dict[str, Dict[str, object]]:
    """The splits and labels of the three readings."""
    identity = identity_labels(records)
    out: Dict[str, Dict[str, object]] = {}
    for label in ("shape", "color"):
        train, test = project_split(records[label], seed)
        out[f"R1|{label}"] = {"y": records[label], "train": train, "test": test, "dropped": 0}
        train3, test3 = stimulus_split(identity, seed)
        out[f"R3|{label}"] = {"y": records[label], "train": train3, "test": test3, "dropped": 0}
    train2, test2, dropped = identity_split(identity, seed)
    out["R2|identity"] = {"y": identity, "train": train2, "test": test2, "dropped": dropped}
    return out


SLOW_ARMS = ("clean", "64", "1024")


def planned(stream: str, reading: str, encoding: str, arm: str, kind: str) -> bool:
    """False for the fits dropped by the cost limits of the docstring."""
    if stream == "z_state" and encoding == "plain" and kind == "linear":
        return reading.startswith("R1") and arm in SLOW_ARMS
    return True


def evaluate_stream(X: np.ndarray, plan: Dict[str, Dict[str, object]], seed: int, base: int, stream: str = "obs_map",
                    encodings: Sequence[str] = ENCODINGS) -> Dict[str, List[int]]:
    """Per-test-row correctness for raw, clean and every dose, for every reading, encoding and classifier."""
    out: Dict[str, List[int]] = {}
    for name, r in plan.items():
        y, train, test = r["y"], r["train"], r["test"]
        for kind in CLASSIFIERS:
            out[f"{name}|raw|{kind}"] = decode_rows(X, y, train, test, kind, seed).astype(int).tolist()
        for encoding in encodings:
            x = encode(encoding, X, train)
            arms = {"clean": None}
            arms.update({str(d): base + 100 * ENCODINGS.index(encoding) + i for i, d in enumerate(DOSES)})
            for arm, noise_seed in arms.items():
                kinds = [k for k in CLASSIFIERS if planned(stream, name, encoding, arm, k)]
                if not kinds:
                    continue
                features = x.numpy() if arm == "clean" else channel(x, int(arm), noise_seed)
                for kind in kinds:
                    out[f"{name}|{encoding}|{arm}|{kind}"] = decode_rows(features, y, train, test, kind, seed).astype(int).tolist()
    return out


def audit(records: Dict[str, np.ndarray], plan: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    identity = identity_labels(records)
    return {"rows": int(len(identity)), "distinct_stimuli": int(len(set(identity))),
            "distinct_rows": {s: int(np.unique(records[s], axis=0).shape[0]) for s in STREAMS},
            "class_counts": {k: np.unique(records[k], return_counts=True)[1].tolist() for k in ("shape", "color", "size")},
            "seen_share": {n: seen_share(identity, r["train"], r["test"]) for n, r in plan.items()},
            "test_rows": {n: int(len(r["test"])) for n, r in plan.items()}, "dropped": {n: r["dropped"] for n, r in plan.items()},
            "test_stimulus": {n: identity[r["test"]].tolist() for n, r in plan.items()}}


def evaluate_records(records: Dict[str, np.ndarray], seed: int, verify: bool,
                     encodings: Sequence[str] = ENCODINGS) -> Dict[str, object]:
    plan = readings(records, seed)
    if verify:
        for stream in STREAMS:
            for label in ("shape", "color"):
                check_against_project(records[stream], records[label], seed)
    return {"audit": audit(records, plan), "raw_dims": {s: int(records[s].shape[1]) for s in STREAMS},
            "streams": {s: evaluate_stream(records[s], plan, seed, 1000 * i, s, encodings) for i, s in enumerate(STREAMS)}}


# ------------------------------------------------------------------------------------ summary


def first_dose_reaching(curve: Dict[str, Dict[str, float]], clean: Dict[str, float], chance: float, level: float) -> Optional[int]:
    """Smallest evaluated dose whose accuracy reaches `level` of the clean above-chance accuracy. None if not estimable or not reached."""
    if clean["low"] <= chance + 0.05:
        return None
    for dose in DOSES:
        if str(dose) in curve and curve[str(dose)]["mean"] - chance >= level * (clean["mean"] - chance):
            return dose
    return None


def summarise_condition(rows: Dict[str, List[int]], prefix: str, kind: str, weights: np.ndarray, chance: float) -> Dict[str, object]:
    def stat(key: str) -> Dict[str, float]:
        return interval(np.array(rows[key], dtype=float), weights)

    out: Dict[str, object] = {"raw": stat(f"{prefix}|raw|{kind}")}
    for encoding in ENCODINGS:
        arms = [arm for arm in ("clean",) + tuple(str(d) for d in DOSES) if f"{prefix}|{encoding}|{arm}|{kind}" in rows]
        if "clean" not in arms:
            continue
        curve = {arm: stat(f"{prefix}|{encoding}|{arm}|{kind}") for arm in arms}
        out[encoding] = {"curve": curve, "T90": first_dose_reaching(curve, curve["clean"], chance, 0.90),
                         "T95": first_dose_reaching(curve, curve["clean"], chance, 0.95)}
    return out


def summarise_recording(result: Dict[str, object], rng: np.random.Generator) -> Dict[str, object]:
    out: Dict[str, object] = {"audit": {k: v for k, v in result["audit"].items() if k != "test_stimulus"}, "streams": {}}
    for stream, rows in result["streams"].items():
        entry = {}
        for name, stimuli in result["audit"]["test_stimulus"].items():
            weights = cluster_weights(np.array(stimuli), rng)
            chance = 1.0 / 72 if name.startswith("R2") else 1.0 / 6
            entry[name] = {kind: summarise_condition(rows, name, kind, weights, chance) for kind in CLASSIFIERS}
        out["streams"][stream] = entry
    return out


# ------------------------------------------------------------------------------------------ IO


def record_path(out_dir: Path, model: int, noise: float) -> Path:
    return out_dir / f"rec_m{model}_n{int(noise)}.npz"


def result_path(out_dir: Path, model: int, noise: float) -> Path:
    return out_dir / f"res_m{model}_n{int(noise)}.json"


def progress(message: str, started: float) -> None:
    print(f"[{time.time() - started:7.0f} s] {message}", file=sys.stderr, flush=True)


def command_record(args: argparse.Namespace) -> None:
    out_dir, started = Path(args.out_dir), time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        for noise in args.noise:
            path = record_path(out_dir, model, noise)
            if path.exists():
                continue
            np.savez(path, **collect_records(CHECKPOINTS[model], args.episodes, args.seed, noise))
            progress(f"recorded model {model} noise {noise}", started)


def command_evaluate(args: argparse.Namespace) -> None:
    out_dir, started = Path(args.out_dir), time.time()
    for path in sorted(out_dir.glob("rec_m*_n*.npz")):
        model, noise = int(path.stem.split("_")[1][1:]), float(path.stem.split("_")[2][1:])
        target = result_path(out_dir, model, noise)
        if target.exists():
            continue
        records = dict(np.load(path, allow_pickle=False))
        encodings = ENCODINGS if noise == 0.0 else ("full64",)
        target.write_text(json.dumps(evaluate_records(records, args.seed, verify=(noise == 0.0), encodings=encodings)))
        progress(f"evaluated model {model} noise {noise}", started)


def command_summarise(args: argparse.Namespace) -> None:
    out_dir, rng = Path(args.out_dir), np.random.default_rng(0)
    summary: Dict[str, object] = {"status": STATUS, "doses": list(DOSES), "recordings": {}}
    for path in sorted(out_dir.glob("res_m*_n*.json")):
        summary["recordings"][path.stem[4:]] = summarise_recording(json.loads(path.read_text()), rng)
    print_summary(summary)
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2))


def print_summary(summary: Dict[str, object]) -> None:
    print(f"survival on the project's instrument, instruments {summary['status']}, doses {summary['doses']}")
    for name, rec in summary["recordings"].items():
        a = rec["audit"]
        print(f"\n{name}: rows {a['rows']}, distinct stimuli {a['distinct_stimuli']}, distinct rows {a['distinct_rows']}")
        print(f"   test rows {a['test_rows']}  seen share {({k: round(v, 3) for k, v in a['seen_share'].items()})}  dropped {a['dropped']}")
        for stream, entry in rec["streams"].items():
            for reading, kinds in entry.items():
                for kind, c in kinds.items():
                    for encoding in ENCODINGS:
                        if encoding not in c:
                            continue
                        curve = c[encoding]["curve"]
                        doses = " ".join(f"{curve[str(d)]['mean']:.2f}" if str(d) in curve else " -- " for d in DOSES)
                        print(f"   {stream:15s} {reading:12s} {kind:7s} {encoding:6s} raw {c['raw']['mean']:.3f} clean {curve['clean']['mean']:.3f} "
                              f"| {doses} | T90 {c[encoding]['T90']} T95 {c[encoding]['T95']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("record", "evaluate", "summarise"):
        p = sub.add_parser(name)
        p.add_argument("--out-dir", default="runs/thermo_survival")
        p.add_argument("--seed", type=int, default=PROBE_SEED)
        if name == "record":
            p.add_argument("--models", type=int, nargs="+", default=sorted(CHECKPOINTS))
            p.add_argument("--noise", type=float, nargs="+", default=list(NOISE_ARMS))
            p.add_argument("--episodes", type=int, default=EPISODES)
        if name == "summarise":
            p.add_argument("--output-json")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    torch.set_num_threads(1)
    {"record": command_record, "evaluate": command_evaluate, "summarise": command_summarise}[args.command](args)


if __name__ == "__main__":
    main()
