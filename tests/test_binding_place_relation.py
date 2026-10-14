"""Stage 2 of the binding design: the agreement of place enters the coupling.

Values computed by hand. The sound cell uses the rule of TopographicMap: column
(az + 1) / 2 * 15 and row (el + 1) / 2 * 15 on a 16 by 16 grid, cut to an integer.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from unittest import mock

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.core.global_workspace import GlobalWorkspace
from models.core.oscillatory_binding import KuramotoLayer, WorkspaceBindingSystem
from models.core.sensory_tectum import place_agreement


def grid_with_peak(row, column):
    grid = torch.zeros(1, 4, 16, 16)
    grid[0, :, row, column] = 1.0
    return grid


def sound_at(column, row):
    spatial = torch.zeros(1, 4, 2)
    # The middle of the cell, so that float rounding cannot move it to the cell before.
    spatial[0, 0, 0] = (column + 0.5) / 15 * 2 - 1
    spatial[0, 0, 1] = (row + 0.5) / 15 * 2 - 1
    return spatial


class TestPlaceAgreement:
    def test_the_same_cell_agrees_fully(self):
        assert place_agreement(grid_with_peak(3, 12), sound_at(12, 3)) == pytest.approx(1.0)

    def test_two_cells_apart_gives_the_gaussian_value(self):
        assert place_agreement(grid_with_peak(3, 12), sound_at(10, 3)) == pytest.approx(math.exp(-4 / 8), abs=1e-6)

    def test_no_sound_gives_none(self):
        assert place_agreement(grid_with_peak(3, 12), torch.zeros(1, 4, 2)) is None


class TestCoupling:
    @staticmethod
    def gap_after(relation_value):
        torch.manual_seed(0)
        layer = KuramotoLayer(num_oscillators=2, dimensions=2)
        with torch.no_grad():
            layer.coupling_weights.fill_(1.0)
        phases = torch.tensor([[[1.0, 0.0], [math.cos(0.5), math.sin(0.5)]]])
        relation = torch.full((1, 2, 2), float(relation_value))
        new, _ = layer(phases, amplitudes=torch.ones(1, 2), iterations=5, relation=relation)
        return float(torch.acos(torch.clamp((new[0, 0] * new[0, 1]).sum(), -1.0, 1.0)))

    def test_a_related_pair_comes_closer_and_an_unrelated_pair_moves_apart(self):
        assert self.gap_after(1.0) < 0.5 < self.gap_after(-1.0)

    def test_no_relation_is_the_same_as_the_call_without_it(self):
        torch.manual_seed(0)
        layer = KuramotoLayer(num_oscillators=3, dimensions=2)
        phases = layer.init_phases(batch_size=1)
        amplitudes = torch.tensor([[0.5, 0.2, 0.9]])
        plain, _ = layer(phases, amplitudes=amplitudes, iterations=5)
        ones, _ = layer(phases, amplitudes=amplitudes, iterations=5, relation=torch.ones(1, 3, 3))
        assert torch.equal(plain, ones)

    def test_the_matrix_holds_2r_minus_1_for_the_pair(self):
        system = WorkspaceBindingSystem(num_modules=3)
        system.register_modules(["vision", "audio", "memory"])
        matrix = system.relation_matrix({("vision", "audio"): 0.25})
        assert matrix[0, 0, 1] == pytest.approx(-0.5) and matrix[0, 1, 0] == pytest.approx(-0.5)
        assert matrix[0, 0, 2] == 1.0 and matrix[0, 2, 1] == 1.0


class TestWorkspace:
    def test_the_default_is_none_and_an_unknown_value_is_refused(self):
        assert GlobalWorkspace({}).binding_relation == "none"
        with pytest.raises(ValueError):
            GlobalWorkspace({"binding_relation": "other"})

    def test_place_needs_the_akorn_binding(self):
        with pytest.raises(ValueError):
            GlobalWorkspace({"binding_relation": "place", "binding_mechanism": "komplex"})


class TestTrainingFlag:
    def test_the_flag_is_off_by_default(self):
        from scripts.training import train_rlhf
        captured = {}

        def capture(self, *args, **kwargs):
            captured["parser"] = self
            raise RuntimeError("captured")

        with mock.patch.object(argparse.ArgumentParser, "parse_args", capture):
            with pytest.raises(RuntimeError):
                train_rlhf.main()
        assert captured["parser"].parse_args([]).binding_relation == "none"
