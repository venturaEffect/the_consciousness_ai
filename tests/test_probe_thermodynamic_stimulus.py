"""Tests for the closed-stimulus probe, on synthetic tensors, CPU only.

The defining case. Take a few fixed stimulus vectors and give each a RANDOM label. Trials repeat the stimuli. A readout
tested with folds grouped by trial has seen every test input in training, so it scores far above chance by lookup. A readout
tested with folds that hold out whole stimuli cannot, so it scores at chance. A probe that cannot tell these apart is wrong.
"""
import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.analysis import probe_thermodynamic_fidelity as fid  # noqa: E402
from scripts.analysis import probe_thermodynamic_stimulus as sti  # noqa: E402
from scripts.analysis.probe_thermodynamic_cost import Recording, make_folds  # noqa: E402


def _lookup_recording(seed: int = 0, n_stimuli: int = 48, trials: int = 144, rows_per_trial: int = 2, dim: int = 80,
                      structured: bool = False) -> Recording:
    """Trials repeat a closed set of stimuli. Labels are random per stimulus unless `structured`."""
    gen = torch.Generator().manual_seed(seed)
    stimuli = torch.randn(n_stimuli, dim, generator=gen)
    labels_of = torch.randint(0, 6, (n_stimuli,), generator=gen)
    if structured:
        centres = 2.0 * torch.randn(6, dim, generator=gen)
        stimuli = centres[labels_of] + 0.5 * stimuli
    which = torch.arange(trials) % n_stimuli
    rows = stimuli[which].repeat_interleave(rows_per_trial, dim=0)
    labels = labels_of[which].repeat_interleave(rows_per_trial)
    trial_ids = torch.arange(trials).repeat_interleave(rows_per_trial)
    return Recording({"obs_map": rows.clone(), "other": rows + 0.0}, labels, trial_ids)


class TestStimulusIds:
    def test_equal_rows_share_an_id_and_different_rows_do_not(self):
        rows = torch.tensor([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [5.0, 6.0], [3.0, 4.0]])
        ids = sti.stimulus_ids(rows)
        assert ids[0] == ids[2] and ids[1] == ids[4] and len(ids.unique()) == 3

    def test_key_is_stable_for_the_same_row_and_differs_for_another(self):
        a, b = torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.5])
        assert sti.stimulus_key(a) == sti.stimulus_key(a.clone()) and sti.stimulus_key(a) != sti.stimulus_key(b)

    def test_distinct_rows_counts_the_closed_set(self):
        rec = _lookup_recording()
        assert sti.distinct_rows(rec.streams["obs_map"]) == 48

    def test_stimulus_folds_keep_every_test_input_out_of_training(self):
        rec = _lookup_recording()
        ids = sti.stimulus_ids(rec.streams["obs_map"])
        for held in make_folds(ids, 0):
            train = {sti.stimulus_key(r) for r in rec.streams["obs_map"][~held]}
            test = {sti.stimulus_key(r) for r in rec.streams["obs_map"][held]}
            assert train.isdisjoint(test)

    def test_trial_folds_do_not_keep_test_inputs_out_of_training(self):
        rec = _lookup_recording()
        leaks = 0
        for held in make_folds(rec.trials, 0):
            train = {sti.stimulus_key(r) for r in rec.streams["obs_map"][~held]}
            leaks += sum(sti.stimulus_key(r) in train for r in rec.streams["obs_map"][held])
        assert leaks > 0


class TestLookupAgainstGeneralisation:
    def _clean(self, rec):
        out = sti.evaluate_recording(rec, seed=1)["streams"]["obs_map"]
        return float(np.mean(out["trial|plain|clean"])), float(np.mean(out["stimulus|plain|clean"]))

    def test_random_labels_are_looked_up_with_trial_folds_and_at_chance_with_stimulus_folds(self):
        trial, stim = self._clean(_lookup_recording(structured=False))
        assert trial > 0.6 and stim < 0.35

    def test_structured_labels_survive_stimulus_folds(self):
        trial, stim = self._clean(_lookup_recording(structured=True))
        assert trial > 0.9 and stim > 0.9

    def test_every_condition_is_scored_once_per_trial(self):
        rec = _lookup_recording(trials=96)
        res = sti.evaluate_recording(rec, seed=1)
        expected = {f"{s}|plain|clean" for s in sti.SCHEMES} | {f"{s}|full64|clean" for s in sti.SCHEMES} \
            | {f"{s}|full64|matched|{d}" for s in sti.SCHEMES for d in fid.LADDER}
        for stream in ("obs_map", "other"):
            assert set(res["streams"][stream]) == expected and all(len(v) == 96 for v in res["streams"][stream].values())
        assert len(res["trial_labels"]) == len(res["trial_stimulus"]) == 96


class TestStatistics:
    def test_cluster_weights_are_equal_within_a_cluster(self):
        clusters = np.array(["a", "a", "b", "c", "c", "c"])
        w = sti.cluster_weights(clusters, np.random.default_rng(0), n=50)
        assert w.shape == (50, 6) and np.all(w[:, 0] == w[:, 1]) and np.all(w[:, 3] == w[:, 5])
        assert np.all(w.sum(1) > 0)

    def test_interval_is_zero_width_for_constant_values(self):
        rng = np.random.default_rng(1)
        flat = sti.interval(np.full(400, 0.5), sti.cluster_weights(np.arange(400) // 40, rng))
        assert flat["high"] - flat["low"] == pytest.approx(0.0)

    def test_resampling_clusters_is_wider_than_resampling_trials_when_trials_of_a_cluster_agree(self):
        rng = np.random.default_rng(1)
        per_cluster = rng.binomial(1, 0.6, 10).astype(float)
        values = np.repeat(per_cluster, 40)  # every trial of a stimulus gets the same score
        by_trial = sti.interval(values, sti.cluster_weights(np.arange(400), rng))
        by_cluster = sti.interval(values, sti.cluster_weights(np.arange(400) // 40, rng))
        assert by_cluster["high"] - by_cluster["low"] > 2 * (by_trial["high"] - by_trial["low"])

    def test_weighted_mean_matches_the_plain_mean_for_unit_weights(self):
        values = np.array([0.0, 1.0, 1.0, 0.0])
        assert sti.weighted_mean(values, np.ones((3, 4)))[0] == pytest.approx(0.5)

    def test_majority_share(self):
        assert sti.majority_share(np.array([0, 0, 0, 1, 2, 2])) == pytest.approx(0.5)


class TestCli:
    def test_round_trip(self, tmp_path):
        rec = _lookup_recording(trials=96)
        torch.save({"streams": rec.streams, "labels": rec.labels, "trials": rec.trials}, fid.recording_path(tmp_path, 42, 300))
        sti.main(["evaluate", "--out-dir", str(tmp_path)])
        out = tmp_path / "s.json"
        sti.main(["summarise", "--out-dir", str(tmp_path), "--output-json", str(out)])
        model = json.loads(out.read_text())["models"]["42"]
        assert model["distinct_stimuli"] == 48 and model["trials"] == 96
        entry = model["streams"]["obs_map"]
        assert {"plain", "full64", "curve", "T95"} <= set(entry) and len(entry["curve"]) == len(fid.LADDER)
        assert entry["plain"]["difference"]["mean"] < 0
        assert entry["estimable"] is False and entry["T95"] is None
