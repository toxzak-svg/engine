from __future__ import annotations

import copy
import unittest

from exp.io import read_yaml_or_json
from exp.models import ExperimentSpec
from exp.simulator import simulate_run


class FailureModeTests(unittest.TestCase):
    def _spec_with_params(self, path: str, params: dict) -> ExperimentSpec:
        payload = copy.deepcopy(read_yaml_or_json(path))
        payload["params"] = params
        return ExperimentSpec.from_dict(payload)

    def test_track1_analog_drift(self) -> None:
        spec = self._spec_with_params("specs/stage1/t1_e2.yaml", {"recalibration_interval": 4096})
        run = simulate_run(spec, seed=101, run_id="t1", commit_sha="abc")
        self.assertIn("analog_drift", run.failure_flags)

    def test_track2_reversibility_break(self) -> None:
        spec = self._spec_with_params(
            "specs/stage1/t2_e1.yaml",
            {"anchor_frequency": "1/2", "disable_norm_constraints": True},
        )
        run = simulate_run(spec, seed=101, run_id="t2", commit_sha="abc")
        self.assertIn("reversibility_break", run.failure_flags)

    def test_track3_critical_fact_loss(self) -> None:
        spec = self._spec_with_params("specs/stage1/t3_e3.yaml", {"compression_ratio": 0.95})
        run = simulate_run(spec, seed=101, run_id="t3", commit_sha="abc")
        self.assertIn("critical_fact_loss", run.failure_flags)

    def test_track4_entity_role_swap(self) -> None:
        spec = self._spec_with_params("specs/stage1/t4_e2.yaml", {"role_permutation_noise": 0.9})
        run = simulate_run(spec, seed=101, run_id="t4", commit_sha="abc")
        self.assertIn("entity_role_swap", run.failure_flags)

    def test_track5_invalid_circuit(self) -> None:
        spec = self._spec_with_params("specs/stage1/t5_e2.yaml", {"max_nodes": 20})
        run = simulate_run(spec, seed=101, run_id="t5", commit_sha="abc")
        self.assertIn("invalid_circuit", run.failure_flags)

    def test_track6_repetitive_text(self) -> None:
        spec = self._spec_with_params("specs/stage1/t6_e2.yaml", {"anneal_temp": 0.1})
        run = simulate_run(spec, seed=101, run_id="t6", commit_sha="abc")
        self.assertIn("repetitive_text", run.failure_flags)


if __name__ == "__main__":
    unittest.main()
