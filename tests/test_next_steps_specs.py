from __future__ import annotations

import unittest

from exp.io import read_yaml_or_json
from exp.models import ExperimentSpec


class NextStepSpecTests(unittest.TestCase):
    def test_stage4_t3_spec_is_valid(self) -> None:
        spec = ExperimentSpec.from_dict(read_yaml_or_json("specs/stage4/t3_e2.yaml"))
        self.assertEqual(spec.stage, 4)
        self.assertEqual(spec.track_id, "T3")
        self.assertEqual(len(spec.seeds), 3)

    def test_recovery_spec_is_valid(self) -> None:
        spec = ExperimentSpec.from_dict(read_yaml_or_json("specs/recovery/stage3/t5_verified_planner.yaml"))
        self.assertEqual(spec.stage, 3)
        self.assertEqual(spec.track_id, "T5")
        self.assertLess(spec.train_budget_gpu_h + spec.infer_budget_gpu_h, 1200.0)

    def test_anchor_spec_exists_each_stage(self) -> None:
        for stage in (0, 1, 2, 3, 4):
            spec = ExperimentSpec.from_dict(read_yaml_or_json(f"specs/stage{stage}/anchor_baseline.yaml"))
            self.assertEqual(spec.track_id, "ANCHOR")


if __name__ == "__main__":
    unittest.main()
