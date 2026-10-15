"""Stage 3 of the binding design: one oscillator for each place on the tectum map.

`docs/decisions/2026_10_14_binding_stage3_map_oscillators.md`, design A with the three
corrections of its review. Expected values are computed by hand where a closed form
exists. The dynamics are checked by their stated outcome: active neighbours synchronise,
separate active regions do not, and the read-out takes the region with the most drive.
"""
from __future__ import annotations

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.core.global_workspace import GlobalWorkspace
from models.core.map_binding import MapOscillators, audio_drive, combine, visual_drive


def blobs(*regions, size=16):
    """A drive map with value 1 inside each (row0, row1, col0, col1) region and 0 elsewhere."""
    drive = torch.zeros(size, size)
    for r0, r1, c0, c1 in regions:
        drive[r0:r1, c0:c1] = 1.0
    return drive


def mean_cosine(phases, first, second):
    a = phases[first[0]:first[1], first[2]:first[3]].reshape(-1, phases.shape[-1])
    b = phases[second[0]:second[1], second[2]:second[3]].reshape(-1, phases.shape[-1])
    return float((a @ b.T).mean())


class TestDrives:
    def test_visual_drive_is_zero_on_the_background_and_one_at_the_peak(self):
        grid = torch.ones(1, 1, 16, 16)
        grid[0, 0, 2:6, 10:14] = 4.0
        grid[0, 0, 3, 12] = 7.0
        drive = visual_drive(grid)
        assert drive[0, 0] == pytest.approx(0.0)
        assert drive[3, 12] == pytest.approx(1.0)
        assert drive[5, 10] == pytest.approx(0.5)

    def test_a_flat_map_has_no_visual_drive(self):
        assert float(visual_drive(torch.ones(1, 3, 16, 16)).max()) == 0.0

    def test_audio_drive_is_a_gaussian_area_around_the_cell_of_the_sound(self):
        spatial = torch.zeros(1, 4, 2)
        spatial[0, 0, 0] = (12 + 0.5) / 15 * 2 - 1
        spatial[0, 0, 1] = (3 + 0.5) / 15 * 2 - 1
        drive = audio_drive(spatial, 16, 16)
        assert drive[3, 12] == pytest.approx(1.0)
        assert drive[3, 13] == pytest.approx(math.exp(-1 / 8))
        assert drive[5, 12] == pytest.approx(math.exp(-4 / 8))

    def test_no_sound_gives_no_audio_drive(self):
        assert float(audio_drive(torch.zeros(1, 4, 2), 16, 16).max()) == 0.0

    def test_the_two_drives_combine_as_either_one(self):
        assert combine(torch.tensor(0.5), torch.tensor(0.5)) == pytest.approx(0.75)
        assert combine(torch.tensor(0.0), torch.tensor(0.3)) == pytest.approx(0.3)


class TestDynamics:
    def test_one_connected_region_synchronises(self):
        layer = MapOscillators(seed=0)
        drive = blobs((2, 8, 2, 8))
        for _ in range(100):
            layer.step(drive)
        assert mean_cosine(layer.phases, (2, 8, 2, 8), (2, 8, 2, 8)) > 0.95

    def test_two_separate_regions_do_not_synchronise_with_each_other(self):
        layer = MapOscillators(seed=0)
        drive = blobs((1, 5, 1, 5), (10, 14, 10, 14))
        for _ in range(100):
            layer.step(drive)
        assert mean_cosine(layer.phases, (1, 5, 1, 5), (1, 5, 1, 5)) > 0.95
        assert mean_cosine(layer.phases, (10, 14, 10, 14), (10, 14, 10, 14)) > 0.95
        assert mean_cosine(layer.phases, (1, 5, 1, 5), (10, 14, 10, 14)) < 0.5

    def test_a_cell_whose_drive_rises_from_zero_restarts_its_phase(self):
        layer = MapOscillators(seed=0)
        layer.step(torch.zeros(16, 16))
        before = layer.phases[4, 4].clone()
        layer.step(blobs((4, 5, 4, 5)))
        assert not torch.allclose(layer.phases[4, 4], before)


class TestReadOut:
    def test_the_region_with_the_most_drive_is_selected(self):
        layer = MapOscillators(seed=0)
        drive = blobs((1, 5, 1, 5), (8, 15, 8, 15))       # 16 cells against 49 cells
        for _ in range(100):
            layer.step(drive)
        group = layer.selected_group(drive)
        assert bool(group[10, 10]) and not bool(group[2, 2])

    def test_a_sense_gets_the_share_of_its_drive_inside_the_selected_group(self):
        group = blobs((0, 8, 0, 16)).bool()               # the upper half
        vision = blobs((2, 6, 2, 6))                      # all inside
        audio = blobs((6, 10, 0, 4))                      # half inside
        factors = MapOscillators.factors(group, vision, audio)
        assert factors["vision"] == pytest.approx(1.5)
        assert factors["audio"] == pytest.approx(1.25)


class TestWorkspace:
    def test_map_factors_replace_the_factors_of_vision_and_audio_only(self):
        workspace = GlobalWorkspace({"map_binding": True})
        workspace.map_factor_source = lambda: {"vision": 1.5, "audio": 1.0}
        bound, _ = workspace.bind_with_map({"vision": 0.4, "audio": 0.3, "memory": 0.1})
        assert bound["vision"] == pytest.approx(0.6)
        assert bound["audio"] == pytest.approx(0.3)

    def test_the_default_is_off_and_map_binding_excludes_a_relation(self):
        assert GlobalWorkspace({}).map_binding is False
        with pytest.raises(ValueError):
            GlobalWorkspace({"map_binding": True, "binding_relation": "place"})


class TestLegionLimit:
    """Option B: the end result that Terman and Wang prove for LEGION (4 nearest neighbours).

    Each connected area of stimulated cells is one group, different areas are different
    groups, and the group with the most drive is selected.
    """

    def test_two_separate_areas_are_two_groups_and_the_stronger_is_selected(self):
        from models.core.map_binding import LegionLimit
        vision = blobs((1, 5, 1, 5))                     # 16 cells
        audio = blobs((8, 15, 8, 15))                    # 49 cells
        factors, record = LegionLimit().update(vision, audio)
        assert record["blocks"] == 2
        assert factors["audio"] == pytest.approx(1.5) and factors["vision"] == pytest.approx(1.0)

    def test_overlapping_senses_form_one_group(self):
        from models.core.map_binding import LegionLimit
        vision = blobs((2, 8, 2, 8))
        audio = blobs((6, 10, 6, 10))
        factors, record = LegionLimit().update(vision, audio)
        assert record["blocks"] == 1
        assert factors == {"vision": pytest.approx(1.5), "audio": pytest.approx(1.5)}

    def test_cells_that_touch_only_at_a_corner_are_separate(self):
        from models.core.map_binding import LegionLimit
        vision = blobs((0, 2, 0, 2), (2, 4, 2, 4))        # corner contact at (1,1)-(2,2)
        _, record = LegionLimit().update(vision, torch.zeros(16, 16))
        assert record["blocks"] == 2

    def test_no_drive_gives_no_group_and_factors_of_one(self):
        from models.core.map_binding import LegionLimit
        factors, record = LegionLimit().update(torch.zeros(16, 16), torch.zeros(16, 16))
        assert record["blocks"] == 0 and record["group"] == []
        assert factors == {"vision": 1.0, "audio": 1.0}

    def test_the_record_lists_the_cells_of_the_selected_group(self):
        from models.core.map_binding import LegionLimit
        _, record = LegionLimit().update(blobs((0, 1, 0, 2)), torch.zeros(16, 16))
        assert record["group"] == [0, 1]


class TestWiring:
    def test_the_tectum_has_no_map_by_default(self):
        from models.core.sensory_tectum import SensoryTectum
        tectum = SensoryTectum({})
        assert tectum.map_binding is None and tectum.map_factors() == {}

    def test_the_training_flag_is_off_by_default(self):
        import argparse
        from unittest import mock
        from scripts.training import train_rlhf
        captured = {}

        def capture(self, *args, **kwargs):
            captured["parser"] = self
            raise RuntimeError("captured")

        with mock.patch.object(argparse.ArgumentParser, "parse_args", capture):
            with pytest.raises(RuntimeError):
                train_rlhf.main()
        assert captured["parser"].parse_args([]).map_binding == "off"
        assert captured["parser"].parse_args(["--map-binding", "limit"]).map_binding == "limit"
        assert captured["parser"].parse_args(["--map-binding", "salience"]).map_binding == "salience"
        assert captured["parser"].parse_args(["--map-binding", "adapted"]).map_binding == "adapted"

    @pytest.mark.parametrize("mode, threshold, drive, adapted", [("limit", 0.05, "visual_drive", False),
                                                                  ("salience", 0.5, "salience_drive", False),
                                                                  ("adapted", 0.5, "salience_drive", True)])
    def test_the_wiring_builds_the_form_of_the_flag(self, mode, threshold, drive, adapted):
        from models.core.sensory_tectum import SensoryTectum
        from scripts.training import train_rlhf
        tectum, workspace = SensoryTectum({}), GlobalWorkspace({"map_binding": True})
        train_rlhf._wire_map_binding(tectum, workspace, None, mode)
        assert tectum.map_binding.threshold == threshold
        assert tectum.map_binding.drive.__name__ == drive
        assert (tectum.map_binding.adaptation is not None) == adapted
        assert workspace.map_factor_source == tectum.map_factors

    def test_the_tectum_forward_feeds_the_workspace_factors(self):
        """The real call path: a tectum forward pass, then the workspace binds with the factors it left."""
        from models.core.sensory_tectum import SensoryTectum
        from scripts.training import train_rlhf
        torch.manual_seed(0)
        tectum, workspace = SensoryTectum({}), GlobalWorkspace({"map_binding": True})
        train_rlhf._wire_map_binding(tectum, workspace, None, "salience")
        frame = torch.zeros(1, 3, 224, 224)
        frame[0, :, 20:60, 150:200] = 1.0
        tectum(frame, torch.zeros(1, tectum.feature_dim, 2))
        factors = tectum.map_factors()
        bound, _ = workspace.bind_with_map({"vision": 0.4, "audio": 0.2, "memory": 0.1})
        assert len(tectum.map_history) == 1
        assert bound["vision"] == pytest.approx(0.4 * factors["vision"])
        assert bound["audio"] == pytest.approx(0.2 * factors["audio"])


def gaussian_weight(size, sigma, source, target):
    """Normalised weight of `source` in the Gaussian mean at `target`, summed by hand over the map."""
    def raw(a, b):
        return math.exp(-((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) / (2.0 * sigma ** 2))
    total = sum(raw((r, c), target) for r in range(size) for c in range(size))
    return raw(source, target) / total


def grid_from(pattern, channels=3):
    """[1, C, H, W] features, the one-hot vector of each cell's label. Equal labels have cosine 1, others 0."""
    rows, cols = len(pattern), len(pattern[0])
    grid = torch.zeros(1, channels, rows, cols)
    for r in range(rows):
        for c in range(cols):
            grid[0, pattern[r][c], r, c] = 1.0
    return grid


class TestSalienceDrive:
    """Candidate K2 of `docs/decisions/2026_10_14_map_binding_drive.md`: centre against surround."""

    def test_a_flat_map_has_no_salience(self):
        from models.core.map_binding import salience_drive
        assert float(salience_drive(torch.ones(1, 4, 16, 16)).max()) == 0.0

    def test_one_odd_cell_gives_the_hand_computed_contrast(self):
        from models.core.map_binding import CENTRE_SIGMA, SURROUND_RATIO, salience_drive
        grid = torch.zeros(1, 1, 8, 8)
        grid[0, 0, 3, 4] = 1.0
        drive = salience_drive(grid)
        surround = CENTRE_SIGMA * SURROUND_RATIO

        def contrast(target):
            return abs(gaussian_weight(8, CENTRE_SIGMA, (3, 4), target) - gaussian_weight(8, surround, (3, 4), target))
        assert drive[3, 4] == pytest.approx(1.0)
        assert drive[3, 6] == pytest.approx(contrast((3, 6)) / contrast((3, 4)), rel=1e-4)
        assert drive[7, 0] == pytest.approx(contrast((7, 0)) / contrast((3, 4)), rel=1e-4)

    def test_the_drive_is_computed_on_the_cpu_for_any_device_input(self):
        from models.core.map_binding import salience_drive
        assert salience_drive(torch.rand(1, 4, 16, 16)).device.type == "cpu"


class TestFeatureSegments:
    """Candidate K3: grey-level LEGION (Chen, Wang and Liu 2000), links by feature similarity."""

    def test_two_uniform_areas_are_two_segments(self):
        from models.core.map_binding import feature_segments
        pattern = [[0] * 3 + [1] * 3 for _ in range(4)]
        labels = feature_segments(grid_from(pattern))
        assert labels.max() == 2
        assert len({int(labels[r, c]) for r in range(4) for c in range(3)}) == 1
        assert labels[0, 0] != labels[0, 5]

    def test_an_area_without_a_leader_is_background(self):
        from models.core.map_binding import feature_segments
        pattern = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 2],
                   [0, 0, 0, 2, 1]]
        labels = feature_segments(grid_from(pattern))
        assert labels.max() == 1                      # the four odd cells have no leader
        assert labels[3, 3] == 0 and labels[4, 4] == 0 and labels[0, 0] == 1

    def test_diagonal_neighbours_link(self):
        from models.core.map_binding import feature_segments
        pattern = [[1, 1, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0],
                   [0, 0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0, 0]]
        labels = feature_segments(grid_from(pattern))
        assert labels[0, 0] == labels[3, 3] != 0      # corner contact links with 8 neighbours


class TestSelection:
    def test_the_group_with_the_most_drive_is_selected(self):
        from models.core.map_binding import select_group
        labels = torch.zeros(16, 16, dtype=torch.long)
        labels[0:2, 0:2], labels[8:16, 8:16] = 1, 2
        vision = torch.zeros(16, 16)
        vision[0, 0], vision[8:16, 8:16] = 1.0, 0.2
        factors, record = select_group(labels.numpy(), vision, torch.zeros(16, 16))
        assert record["blocks"] == 2
        assert factors["vision"] == pytest.approx(1.0 + 0.5 * 12.8 / 13.8)

    def test_audio_on_the_background_counts_for_no_group(self):
        from models.core.map_binding import select_group
        labels = torch.zeros(16, 16, dtype=torch.long)
        labels[0:4, 0:4] = 1
        vision = torch.zeros(16, 16)
        vision[0:4, 0:4] = 1.0
        audio = torch.zeros(16, 16)
        audio[10, 10] = 1.0
        factors, _ = select_group(labels.numpy(), vision, audio)
        assert factors == {"vision": pytest.approx(1.5), "audio": pytest.approx(1.0)}


class TestObserve:
    def test_observe_equals_update_on_the_drives_it_computes(self):
        from models.core.map_binding import LegionLimit
        grid = torch.ones(1, 1, 16, 16)
        grid[0, 0, 2:6, 2:6] = 4.0
        spatial = torch.zeros(1, 4, 2)
        spatial[0, 0, 0] = spatial[0, 0, 1] = (3 + 0.5) / 15 * 2 - 1
        limit = LegionLimit()
        assert limit.observe(grid, spatial) == limit.update(visual_drive(grid), audio_drive(spatial, 16, 16))

    def test_the_segment_form_selects_the_area_with_the_sound(self):
        """A checkerboard background has no leader. Two squares at mirror places have equal salience,
        so the sound on one of them decides."""
        from models.core.map_binding import SegmentLimit
        pattern = [[3 + (r + c) % 2 for c in range(16)] for r in range(16)]
        for r in range(2, 7):
            for c in range(2, 7):
                pattern[r][c], pattern[15 - r][15 - c] = 1, 2
        spatial = torch.zeros(1, 4, 2)
        spatial[0, 0, 0] = spatial[0, 0, 1] = (12 + 0.5) / 15 * 2 - 1
        factors, record = SegmentLimit().observe(grid_from(pattern, channels=5), spatial)
        assert record["blocks"] == 2
        assert 12 * 16 + 12 in record["group"] and 4 * 16 + 4 not in record["group"]
        assert factors["audio"] > 1.25


class TestCellAdaptation:
    """Design A of `docs/decisions/2026_10_14_map_binding_adaptation.md`."""

    def test_a_constant_drive_falls_as_the_closed_form(self):
        from models.core.map_binding import ADAPT_DEPTH, ADAPT_RATE, CellAdaptation
        adaptation = CellAdaptation()
        drive = torch.ones(16, 16)
        outputs = [float(adaptation.step("vision", drive)[3, 3]) for _ in range(4)]
        # mean before step n+1 is 1 - (1 - rate)^n, so the output is 1 - depth * (1 - (1 - rate)^n)
        expected = [1.0 - ADAPT_DEPTH * (1.0 - (1.0 - ADAPT_RATE) ** n) for n in range(4)]
        assert outputs == pytest.approx(expected)

    def test_the_fall_is_bounded_by_the_depth(self):
        from models.core.map_binding import ADAPT_DEPTH, CellAdaptation
        adaptation = CellAdaptation()
        for _ in range(500):
            last = adaptation.step("vision", torch.ones(16, 16))
        assert float(last[0, 0]) == pytest.approx(1.0 - ADAPT_DEPTH, abs=1e-6)

    def test_adaptation_is_specific_to_the_cell_and_to_the_sense(self):
        from models.core.map_binding import CellAdaptation
        adaptation = CellAdaptation()
        one_cell = torch.zeros(16, 16)
        one_cell[2, 2] = 1.0
        for _ in range(10):
            adaptation.step("vision", one_cell)
        other_cell = torch.zeros(16, 16)
        other_cell[9, 9] = 1.0
        assert float(adaptation.step("vision", other_cell)[9, 9]) == pytest.approx(1.0)
        assert float(adaptation.step("audio", one_cell)[2, 2]) == pytest.approx(1.0)

    def test_a_moving_area_wins_over_an_equal_area_that_stays(self):
        from models.core.map_binding import CellAdaptation, LegionLimit
        limit = LegionLimit(threshold=0.5, adaptation=CellAdaptation())
        places = [(0, 0), (0, 4), (0, 8), (0, 12), (12, 0), (12, 4), (12, 8), (12, 12)]
        for r, c in places:
            vision = blobs((7, 10, 7, 10), (r, r + 3, c, c + 3))
            _, record = limit.update(vision, torch.zeros(16, 16))
        assert record["blocks"] == 2
        assert 12 * 16 + 12 in record["group"] and 8 * 16 + 8 not in record["group"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_a_visual_grid_on_the_gpu_works_with_an_audio_drive_on_the_cpu():
    """Regression: the training loop passes the encoder output on the GPU (first option B runs, 2026-10-14)."""
    from models.core.map_binding import LegionLimit
    grid = torch.ones(1, 1, 16, 16, device="cuda")
    grid[0, 0, 2:6, 2:6] = 4.0
    spatial = torch.zeros(1, 4, 2, device="cuda")
    spatial[0, 0, 0] = (3 + 0.5) / 15 * 2 - 1
    spatial[0, 0, 1] = (3 + 0.5) / 15 * 2 - 1
    factors, record = LegionLimit().update(visual_drive(grid), audio_drive(spatial, 16, 16))
    assert record["blocks"] == 1 and factors["vision"] == pytest.approx(1.5)
