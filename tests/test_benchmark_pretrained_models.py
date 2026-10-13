"""Helpers of scripts/analysis/benchmark_pretrained_models.py. No model is loaded here.

Every expected number is counted by hand.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.analysis import benchmark_pretrained_models as bench


class TestCountParameters:
    def test_linear_layer(self):
        assert bench.count_parameters(torch.nn.Linear(3, 2)) == 3 * 2 + 2

    def test_a_shared_parameter_is_counted_once(self):
        shared = torch.nn.Linear(4, 4)
        assert bench.count_parameters(torch.nn.Sequential(shared, shared)) == 4 * 4 + 4


class TestTimes:
    def test_summary_of_an_odd_number_of_times(self):
        assert bench.summarise_times([0.3, 0.1, 0.2]) == {"runs": 3, "min_s": 0.1, "median_s": 0.2, "max_s": 0.3}

    def test_no_time_gives_none_and_not_zero(self):
        assert bench.summarise_times([]) == {"runs": 0, "min_s": None, "median_s": None, "max_s": None}

    def test_timed_loop_runs_at_least_the_minimum_number_of_times(self):
        calls = []
        times = bench.timed_loop(lambda: calls.append(1), seconds=0.0, minimum_runs=3)
        assert len(times) == 3 and len(calls) == 3
        assert all(t >= 0.0 for t in times)


class TestPower:
    def test_reads_the_first_line_as_watts(self, monkeypatch):
        monkeypatch.setattr(bench.subprocess, "check_output", lambda *a, **k: b"23.50\n")
        assert bench.read_gpu_power() == pytest.approx(23.5)

    def test_a_failed_call_gives_none(self, monkeypatch):
        def fail(*args, **kwargs):
            raise OSError("no nvidia-smi")
        monkeypatch.setattr(bench.subprocess, "check_output", fail)
        assert bench.read_gpu_power() is None

    def test_mean_power_ignores_missing_samples_and_needs_one(self):
        assert bench.mean_power([20.0, None, 30.0]) == pytest.approx(25.0)
        assert bench.mean_power([None, None]) is None


class TestPooling:
    def test_mean_pool_and_grid_pool_of_a_patch_grid(self):
        grid = np.arange(2 * 4 * 4 * 3, dtype=np.float64).reshape(2, 4, 4, 3)   # [rows, 4, 4, channels]
        pooled = bench.pool_patch_grid(grid, cells=1)
        assert pooled.shape == (2, 3)
        assert pooled[0, 0] == pytest.approx(np.mean(grid[0, :, :, 0]))
        quarters = bench.pool_patch_grid(grid, cells=2)
        assert quarters.shape == (2, 2 * 2 * 3)
        assert quarters[0, 0] == pytest.approx(np.mean(grid[0, :2, :2, 0]))

    def test_a_grid_that_does_not_divide_is_refused(self):
        with pytest.raises(ValueError):
            bench.pool_patch_grid(np.zeros((1, 5, 5, 2)), cells=2)


class TestUnseenStimulusDecode:
    def test_a_feature_that_holds_the_class_is_read_on_unseen_stimuli(self):
        rng = np.random.default_rng(0)
        shapes = np.repeat(np.arange(4), 30)                       # 4 classes
        stimulus = np.array(["%d|%d" % (s, i % 6) for i, s in enumerate(shapes)])   # 6 stimuli per class
        features = np.eye(4)[shapes] + 0.01 * rng.standard_normal((120, 4))
        scores = bench.unseen_stimulus_accuracy(features, shapes, stimulus, splits=3)
        assert len(scores) == 3 and min(scores) == 1.0

    def test_noise_features_stay_near_chance(self):
        rng = np.random.default_rng(1)
        shapes = np.repeat(np.arange(4), 60)
        stimulus = np.array(["%d|%d" % (s, i % 6) for i, s in enumerate(shapes)])
        scores = bench.unseen_stimulus_accuracy(rng.standard_normal((240, 8)), shapes, stimulus, splits=5)
        assert max(scores) < 0.5

    def test_no_test_stimulus_is_in_training(self):
        stimulus = np.array(["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"])
        for train, test in bench.stimulus_splits(stimulus, splits=4):
            assert not set(stimulus[train]) & set(stimulus[test])
