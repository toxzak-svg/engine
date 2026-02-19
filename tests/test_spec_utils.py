import unittest
from exp.spec_utils import build_spec_template, build_cost_matched_baseline_spec
from exp.models import ExperimentSpec

class TestSpecUtils(unittest.TestCase):

    def test_build_spec_template(self):
        base_spec = {"key1": "value1", "key2": "value2"}
        overrides = {"key2": "new_value2", "key3": "value3"}
        result = build_spec_template(base_spec, **overrides)

        expected = {"key1": "value1", "key2": "new_value2", "key3": "value3"}
        self.assertEqual(result, expected)

    def test_build_cost_matched_baseline_spec(self):
        candidate = ExperimentSpec.from_dict({
            "id": "candidate1",
            "train_budget_gpu_h": 10,
            "infer_budget_gpu_h": 5,
            "max_context": 1024,
            "seeds": [1, 2, 3],
            "params": {},
        })
        baseline = ExperimentSpec.from_dict({
            "id": "baseline1",
            "params": {},
        })

        result = build_cost_matched_baseline_spec(candidate, baseline)

        self.assertEqual(result.id, "candidate1-matched-baseline")
        self.assertEqual(result.train_budget_gpu_h, 10)
        self.assertEqual(result.infer_budget_gpu_h, 5)
        self.assertEqual(result.max_context, 1024)
        self.assertEqual(result.seeds, [1, 2, 3])

if __name__ == "__main__":
    unittest.main()