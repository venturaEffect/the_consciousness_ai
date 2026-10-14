"""The launcher that scales the broadcast to the size of each module's own vector.

Every expected number is computed by hand. The own vector is (3, 4), size 5. The
broadcast is (0, 10), size 10. Scaled to size 5 it is (0, 5). The rule of the modules is
pe = |broadcast - own| / |own|, so pe = |(-3, 1)| / 5 = sqrt(10) / 5. With both vectors
at size 1 the distance is |(0, 1) - (0.6, 0.8)| = sqrt(0.4), the same number.
"""
from __future__ import annotations

import json
import math
import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.audio.auditory_specialist import AuditorySpecialist
from models.core.sensory_tectum import SensoryTectum
from scripts.analysis import run_unit_feedback as launcher

OWN = torch.tensor([3.0, 4.0])
BROADCAST = torch.tensor([0.0, 10.0])
UNIT_PE = math.sqrt(10.0) / 5.0


class TestExtractTensor:
    def test_a_tensor_is_returned_as_it_is(self):
        assert launcher.extract_tensor(BROADCAST) is BROADCAST

    def test_the_fused_key_comes_before_the_tensor_key(self):
        fused, other = torch.ones(2), torch.zeros(2)
        assert launcher.extract_tensor({"_fused": fused, "tensor": other}) is fused
        assert launcher.extract_tensor({"tensor": other}) is other

    def test_no_tensor_gives_none(self):
        assert launcher.extract_tensor(None) is None
        assert launcher.extract_tensor({"winner": "audio"}) is None
        assert launcher.extract_tensor("text") is None


class TestRescale:
    def test_the_broadcast_gets_the_size_of_the_own_vector(self):
        scaled = launcher.rescale_to_own_size(BROADCAST, OWN)
        assert torch.allclose(scaled, torch.tensor([0.0, 5.0]), atol=1e-6)

    def test_a_longer_broadcast_is_cut_before_the_sizes_are_taken(self):
        scaled = launcher.rescale_to_own_size(torch.tensor([0.0, 10.0, 99.0]), OWN)
        assert scaled.shape == (2,)
        assert torch.allclose(scaled, torch.tensor([0.0, 5.0]), atol=1e-6)

    def test_a_longer_own_vector_is_cut_before_the_sizes_are_taken(self):
        scaled = launcher.rescale_to_own_size(BROADCAST, torch.tensor([3.0, 4.0, 99.0]))
        assert torch.allclose(scaled, torch.tensor([0.0, 5.0]), atol=1e-6)

    def test_a_batch_dimension_is_flattened(self):
        scaled = launcher.rescale_to_own_size(BROADCAST.view(1, 2), OWN.view(1, 2))
        assert scaled.shape == (2,)


class TestRuleOfTheRealModules:
    """The wrapper calls the rule of the module itself. Only the broadcast changes."""

    @pytest.mark.parametrize("module_class", [SensoryTectum, AuditorySpecialist])
    def test_without_the_wrapper_the_size_of_the_broadcast_sets_the_error(self, module_class):
        module = SimpleNamespace(_last_content=OWN)
        raw_pe = math.sqrt(9.0 + 36.0) / 5.0          # |(0,10) - (3,4)| / 5 = 1.342
        assert module_class.receive_broadcast(module, BROADCAST, 0.40) == pytest.approx(0.40 + 0.1 * raw_pe, abs=1e-6)

    @pytest.mark.parametrize("module_class", [SensoryTectum, AuditorySpecialist])
    def test_with_the_wrapper_the_error_is_the_distance_at_unit_size(self, module_class):
        calls = {}
        wrapped = launcher.unit_feedback(module_class.receive_broadcast, calls, "module")
        module = SimpleNamespace(_last_content=OWN)
        assert wrapped(module, BROADCAST, 0.40) == pytest.approx(0.40 + 0.1 * UNIT_PE, abs=1e-6)
        assert wrapped(module, BROADCAST * 100.0, 0.40) == pytest.approx(0.40 + 0.1 * UNIT_PE, abs=1e-5)
        assert calls == {"module": 2}

    @pytest.mark.parametrize("module_class", [SensoryTectum, AuditorySpecialist])
    def test_the_own_vector_as_broadcast_gives_no_error_at_any_size(self, module_class):
        wrapped = launcher.unit_feedback(module_class.receive_broadcast, {}, "module")
        module = SimpleNamespace(_last_content=OWN)
        assert wrapped(module, OWN * 7.0, 0.40) == pytest.approx(0.40, abs=1e-6)

    def test_no_tensor_goes_to_the_rule_unchanged_and_is_not_counted(self):
        calls = {}
        module = SimpleNamespace(_last_content=OWN)
        tectum = launcher.unit_feedback(SensoryTectum.receive_broadcast, calls, "vision")
        audio = launcher.unit_feedback(AuditorySpecialist.receive_broadcast, calls, "audio")
        assert tectum(module, None, 0.40) == pytest.approx(0.40 * 0.95)
        assert audio(module, None, 0.40) == pytest.approx(0.40)
        assert calls == {}

    def test_no_own_vector_goes_to_the_rule_unchanged_and_is_not_counted(self):
        calls = {}
        tectum = launcher.unit_feedback(SensoryTectum.receive_broadcast, calls, "vision")
        assert tectum(SimpleNamespace(_last_content=None), BROADCAST, 0.40) == pytest.approx(0.40 * 0.95)
        assert calls == {}


class TestInstallAndRecord:
    def test_install_replaces_the_rule_of_both_classes_and_remove_puts_it_back(self):
        before = (SensoryTectum.receive_broadcast, AuditorySpecialist.receive_broadcast)
        calls = {}
        remove = launcher.install(calls)
        try:
            assert SensoryTectum.receive_broadcast is not before[0]
            assert AuditorySpecialist.receive_broadcast is not before[1]
            module = SimpleNamespace(_last_content=OWN)
            SensoryTectum.receive_broadcast(module, BROADCAST, 0.40)
            AuditorySpecialist.receive_broadcast(module, BROADCAST, 0.40)
            assert calls == {"vision": 1, "audio": 1}
        finally:
            remove()
        assert (SensoryTectum.receive_broadcast, AuditorySpecialist.receive_broadcast) == before

    def test_the_log_folder_is_read_from_the_arguments(self):
        assert launcher.log_dir_of(["--env", "dark_room", "--log-dir", "runs/x"]) == "runs/x"
        assert launcher.log_dir_of(["--log-dir=runs/y", "--seed", "3"]) == "runs/y"

    def test_a_command_without_a_log_folder_is_refused(self):
        with pytest.raises(SystemExit):
            launcher.log_dir_of(["--env", "dark_room"])

    def test_the_call_counts_are_written_to_the_run_folder(self, tmp_path):
        launcher.write_calls(str(tmp_path), {"vision": 3, "audio": 2})
        with open(os.path.join(str(tmp_path), launcher.CALLS_FILE), encoding="utf-8") as handle:
            assert json.load(handle) == {"audio": 2, "vision": 3}


class TestSizeVerdict:
    CALLS = {"vision": 10, "audio": 10}

    def test_a_seed_holds_when_the_unit_share_is_below_half_of_the_control_share(self):
        assert launcher.seed_reading(0.60, 0.29, self.CALLS) is True
        assert launcher.seed_reading(0.60, 0.30, self.CALLS) is False

    def test_a_seed_is_not_measurable_without_scaled_calls_or_without_a_share(self):
        assert launcher.seed_reading(0.60, 0.10, {"vision": 10, "audio": 0}) is None
        assert launcher.seed_reading(None, 0.10, self.CALLS) is None
        assert launcher.seed_reading(0.60, None, self.CALLS) is None

    def test_supported_needs_every_seed(self):
        assert launcher.size_verdict([True, True, True], [self.CALLS] * 3) == "SUPPORTED"

    def test_two_failed_seeds_give_not_supported(self):
        assert launcher.size_verdict([True, False, False], [self.CALLS] * 3) == "NOT SUPPORTED"

    def test_one_failed_or_one_unmeasured_seed_gives_undecided(self):
        assert launcher.size_verdict([True, True, False], [self.CALLS] * 3) == "UNDECIDED"
        assert launcher.size_verdict([True, True, None], [self.CALLS] * 3) == "UNDECIDED"

    def test_a_run_without_scaled_calls_gives_void(self):
        assert launcher.size_verdict([True, True, None], [self.CALLS, self.CALLS, {"vision": 5}]) == "VOID"


class TestWinnerVerdict:
    CALLS = {"vision": 10, "audio": 10}

    def test_the_line_is_the_limit_of_the_gates(self):
        assert launcher.winner_reading(0.95, 0.949) == {"W1": True, "W2": True}
        assert launcher.winner_reading(0.949, 0.95) == {"W1": False, "W2": False}

    def test_a_share_that_cannot_be_measured_gives_none(self):
        assert launcher.winner_reading(None, 0.5) == {"W1": None, "W2": True}

    def test_supported_needs_both_predictions_at_every_seed(self):
        readings = [launcher.winner_reading(0.99, 0.60)] * 3
        assert launcher.winner_verdict(readings, [self.CALLS] * 3) == "SUPPORTED"

    def test_two_unit_runs_with_a_takeover_give_not_supported(self):
        readings = [launcher.winner_reading(0.99, 0.60)] + [launcher.winner_reading(0.99, 0.97)] * 2
        assert launcher.winner_verdict(readings, [self.CALLS] * 3) == "NOT SUPPORTED"

    def test_no_takeover_in_two_control_runs_gives_void(self):
        readings = [launcher.winner_reading(0.99, 0.60)] + [launcher.winner_reading(0.70, 0.60)] * 2
        assert launcher.winner_verdict(readings, [self.CALLS] * 3) == "VOID"

    def test_one_failed_prediction_gives_undecided(self):
        readings = [launcher.winner_reading(0.99, 0.60)] * 2 + [launcher.winner_reading(0.99, 0.97)]
        assert launcher.winner_verdict(readings, [self.CALLS] * 3) == "UNDECIDED"
        readings = [launcher.winner_reading(0.99, 0.60)] * 2 + [launcher.winner_reading(0.70, 0.60)]
        assert launcher.winner_verdict(readings, [self.CALLS] * 3) == "UNDECIDED"

    def test_a_unit_run_without_scaled_calls_gives_void(self):
        readings = [launcher.winner_reading(0.99, 0.60)] * 3
        assert launcher.winner_verdict(readings, [self.CALLS, self.CALLS, {"audio": 4}]) == "VOID"
