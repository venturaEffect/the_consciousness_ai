"""Tests for the survival probe, on synthetic arrays, CPU only. No model is loaded.

What must hold.
    The probe's decoders are the project's decoders: on the project's split they give the same accuracy as
    linear_decode and pca_mlp_decode, to the last digit.
    A stimulus split shares no stimulus between its two sides. A trial split of repeated stimuli does.
    A closed set with random labels per stimulus is looked up under the project's split (R1) and is at chance with whole
    stimuli held out (R3). A probe that cannot tell these apart is wrong.
    The channel estimate tends to the clean encoding as the dose grows, with variance x (1 - x) / T.
The evaluation tests use two doses and compute each synthetic case once, because every condition is a classifier fit.
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.analysis import probe_thermodynamic_survival as sv  # noqa: E402

SHAPES = ["circle", "square", "triangle", "star", "cross", "diamond"]
COLORS = ["red", "green", "blue", "yellow", "magenta", "cyan"]
FAST_DOSES = (16, 4096)


def _records(seed: int = 0, trials: int = 280, dim: int = 60, structured: bool = True, jitter: float = 0.0):
    """A closed set of 72 stimuli. Each stimulus has one fixed vector per stream, as in DMTS without sensory noise."""
    rng = np.random.default_rng(seed)
    grid = [(a, b, c) for a in SHAPES for b in COLORS for c in ("small", "large")]
    base = rng.normal(size=(72, dim)).astype(np.float32)
    if structured:
        shape_code = rng.normal(size=(6, dim)).astype(np.float32) * 2.0
        color_code = rng.normal(size=(6, dim)).astype(np.float32) * 2.0
        base = np.stack([shape_code[SHAPES.index(a)] + color_code[COLORS.index(b)] + 0.3 * base[i] for i, (a, b, _) in enumerate(grid)])
    which = rng.integers(0, 72, size=trials)
    noise = jitter * rng.normal(size=(trials, dim)).astype(np.float32)
    out = {s: (base[which] + noise).astype(np.float32) for s in sv.STREAMS}
    out["shape"] = np.array([grid[i][0] for i in which])
    out["color"] = np.array([grid[i][1] for i in which])
    out["size"] = np.array([grid[i][2] for i in which])
    return out


@pytest.fixture(scope="module")
def evaluated():
    """Both synthetic cases, evaluated once at two doses: {name: (records, plan, rows, accuracy)}."""
    old, sv.DOSES = sv.DOSES, FAST_DOSES
    out = {}
    try:
        for name, structured in (("lookup", False), ("structured", True)):
            rec = _records(structured=structured)
            plan = sv.readings(rec, 42)
            rows = sv.evaluate_stream(rec["obs_map"], plan, 42, 0)
            out[name] = (rec, plan, rows, {k: float(np.mean(v)) for k, v in rows.items()})
    finally:
        sv.DOSES = old  # restore before any other test runs
    return out


class TestSameInstrument:
    @pytest.mark.parametrize("label", ["shape", "color"])
    def test_decoders_match_the_project_decoders_exactly(self, label):
        rec = _records(jitter=0.5)
        sv.check_against_project(rec["obs_map"], rec[label], seed=42)  # raises on any difference

    def test_check_raises_when_the_split_differs(self, monkeypatch):
        rec = _records(jitter=3.0, structured=False)
        monkeypatch.setattr(sv, "TEST_SIZE", 0.5)
        with pytest.raises(RuntimeError):
            sv.check_against_project(rec["obs_map"], rec["shape"], seed=42)

    def test_classifier_pipelines_are_the_documented_ones(self):
        linear = sv.build_classifier("linear", 0, 196, 60)
        mlp = sv.build_classifier("pca_mlp", 0, 196, 60)
        assert [type(s).__name__ for s in linear] == ["StandardScaler", "LogisticRegression"] and linear[-1].max_iter == 1000
        assert [type(s).__name__ for s in mlp] == ["StandardScaler", "PCA", "MLPClassifier"]
        assert mlp[1].n_components == 60 and mlp[-1].hidden_layer_sizes == (64,) and mlp[-1].max_iter == 2000


class TestSplits:
    def test_identity_label_is_the_72_way_combination(self):
        rec = _records()
        identity = sv.identity_labels(rec)
        assert identity[0] == f"{rec['shape'][0]}|{rec['color'][0]}|{rec['size'][0]}" and len(set(identity)) <= 72

    def test_project_split_tests_on_stimuli_seen_in_training(self):
        rec = _records()
        identity = sv.identity_labels(rec)
        train, test = sv.project_split(rec["shape"], 42)
        assert len(test) == 84 and sv.seen_share(identity, train, test) > 0.9

    def test_stimulus_split_shares_no_stimulus(self):
        identity = sv.identity_labels(_records())
        train, test = sv.stimulus_split(identity, 42)
        assert set(identity[train]).isdisjoint(set(identity[test])) and sv.seen_share(identity, train, test) == 0.0

    def test_identity_split_drops_test_rows_with_an_unseen_stimulus(self):
        identity = np.array(["a"] * 20 + ["b"] * 20 + ["c"])  # c occurs once
        train, test, dropped = sv.identity_split(identity, 0)
        assert set(identity[test]) <= set(identity[train])
        assert dropped == (1 if "c" not in identity[train] else 0)


class TestReadings:
    def test_lookup_set_is_read_under_the_project_split_and_not_with_stimuli_held_out(self, evaluated):
        acc = evaluated["lookup"][3]  # a random vector per stimulus: nothing to generalise from
        assert acc["R1|shape|raw|linear"] > 0.8
        assert acc["R3|shape|raw|linear"] < 0.4
        assert acc["R2|identity|raw|linear"] > 0.9

    def test_structured_code_generalises_to_unseen_stimuli(self, evaluated):
        acc = evaluated["structured"][3]
        assert acc["R3|shape|raw|linear"] > 0.8 and acc["R3|color|raw|linear"] > 0.8

    def test_every_condition_is_present_with_one_value_per_test_row(self, evaluated):
        _, plan, rows, _ = evaluated["structured"]
        arms = ["clean"] + [str(d) for d in FAST_DOSES]
        for name, r in plan.items():
            expected = {f"{name}|raw|{k}" for k in sv.CLASSIFIERS} | {f"{name}|{e}|{a}|{k}" for e in sv.ENCODINGS for a in arms for k in sv.CLASSIFIERS}
            assert expected <= set(rows) and all(len(rows[key]) == len(r["test"]) for key in expected)

    def test_accuracy_rises_with_dose_towards_the_clean_value(self, evaluated):
        acc = evaluated["structured"][3]
        low, high, clean = acc["R1|shape|full64|16|linear"], acc["R1|shape|full64|4096|linear"], acc["R1|shape|full64|clean|linear"]
        assert high >= low and high >= clean - 0.05


class TestChannel:
    def test_encodings_stay_inside_the_unit_interval_and_use_training_rows_only(self):
        rec = _records()
        train, test = sv.project_split(rec["shape"], 42)
        changed = rec["obs_map"].copy()
        changed[test] += 5.0
        for kind in sv.ENCODINGS:
            a, b = sv.encode(kind, rec["obs_map"], train), sv.encode(kind, changed, train)
            assert float(a.min()) > 0.0 and float(a.max()) < 1.0
            assert torch.allclose(a[train], b[train], atol=1e-5)

    def test_full64_has_64_axes_and_plain_keeps_the_raw_dimension(self):
        rec = _records(dim=100)
        train, _ = sv.project_split(rec["shape"], 42)
        assert sv.encode("full64", rec["obs_map"], train).shape == (280, 64)
        assert sv.encode("plain", rec["obs_map"], train).shape == (280, 100)

    def test_channel_is_unbiased_with_the_bernoulli_variance(self):
        x = torch.full((4000, 3), 0.3)
        est = sv.channel(x, 64, seed=1)
        assert est.mean() == pytest.approx(0.3, abs=0.01)
        assert est.var() == pytest.approx(0.3 * 0.7 / 64, rel=0.1)

    def test_events_count_dimensions_times_dose(self):
        assert sv.events("plain", 16384, 64) == 16384 * 64 and sv.events("full64", 16384, 64) == 64 * 64


class TestSummary:
    def test_first_dose_needs_a_clean_value_above_chance(self):
        curve = {str(d): {"mean": 0.5} for d in sv.DOSES}
        assert sv.first_dose_reaching(curve, {"mean": 0.2, "low": 0.18}, 1 / 6, 0.9) is None
        assert sv.first_dose_reaching(curve, {"mean": 0.5, "low": 0.4}, 1 / 6, 0.9) == sv.DOSES[0]

    def test_first_dose_is_none_when_no_dose_reaches_the_level(self):
        curve = {str(d): {"mean": 0.2} for d in sv.DOSES}
        assert sv.first_dose_reaching(curve, {"mean": 0.9, "low": 0.8}, 1 / 6, 0.9) is None

    def test_first_dose_skips_doses_that_were_not_evaluated(self):
        curve = {"64": {"mean": 0.2}, "1024": {"mean": 0.9}}
        assert sv.first_dose_reaching(curve, {"mean": 0.9, "low": 0.8}, 1 / 6, 0.9) == 1024

    def test_audit_reports_the_closed_set(self, evaluated):
        rec, plan, _, _ = evaluated["structured"]
        a = sv.audit(rec, plan)
        assert a["rows"] == 280 and a["distinct_stimuli"] <= 72 and a["distinct_rows"]["obs_map"] == a["distinct_stimuli"]
        assert a["seen_share"]["R1|shape"] > 0.9 and a["seen_share"]["R3|shape"] == 0.0 and a["seen_share"]["R2|identity"] == 1.0

    def test_recording_summary_has_three_readings_and_both_classifiers(self, evaluated, monkeypatch):
        monkeypatch.setattr(sv, "DOSES", FAST_DOSES)
        rec, plan, rows, _ = evaluated["structured"]
        result = {"audit": sv.audit(rec, plan), "streams": {"obs_map": rows}}
        entry = sv.summarise_recording(result, np.random.default_rng(0))["streams"]["obs_map"]
        assert set(entry) == {"R1|shape", "R1|color", "R3|shape", "R3|color", "R2|identity"}
        assert set(entry["R1|shape"]) == set(sv.CLASSIFIERS) and "T95" in entry["R1|shape"]["linear"]["full64"]
        assert entry["R1|shape"]["linear"]["raw"]["low"] <= entry["R1|shape"]["linear"]["raw"]["mean"]


class TestCostLimits:
    def test_only_the_slow_z_state_linear_fits_are_dropped(self):
        assert sv.planned("z_state", "R1|shape", "plain", "64", "linear")
        assert not sv.planned("z_state", "R1|shape", "plain", "256", "linear")
        assert not sv.planned("z_state", "R3|shape", "plain", "64", "linear")
        assert sv.planned("z_state", "R3|shape", "plain", "256", "pca_mlp")
        assert sv.planned("z_state", "R2|identity", "full64", "256", "linear")
        assert sv.planned("obs_map", "R3|shape", "plain", "256", "linear")

    def test_dropped_fits_are_absent_and_the_summary_still_works(self, monkeypatch):
        monkeypatch.setattr(sv, "DOSES", (64, 256))
        rec = _records(trials=140, dim=30)
        plan = sv.readings(rec, 42)
        rows = sv.evaluate_stream(rec["z_state"], plan, 42, 0, "z_state")
        assert "R1|shape|plain|64|linear" in rows and "R1|shape|plain|256|linear" not in rows
        assert "R3|shape|plain|clean|linear" not in rows and "R3|shape|plain|256|pca_mlp" in rows
        result = {"audit": sv.audit(rec, plan), "streams": {"z_state": rows}}
        entry = sv.summarise_recording(result, np.random.default_rng(0))["streams"]["z_state"]
        assert set(entry["R1|shape"]["linear"]["plain"]["curve"]) == {"clean", "64"}
        assert "plain" not in entry["R3|shape"]["linear"] and "full64" in entry["R3|shape"]["linear"]

    def test_noise_arm_runs_full64_only(self, monkeypatch):
        monkeypatch.setattr(sv, "DOSES", (64,))
        monkeypatch.setattr(sv, "STREAMS", ("obs_map",))
        rec = _records(trials=140, dim=30)
        rows = sv.evaluate_records(rec, 42, verify=False, encodings=("full64",))["streams"]["obs_map"]
        assert not any("|plain|" in k for k in rows) and any("|full64|" in k for k in rows)
