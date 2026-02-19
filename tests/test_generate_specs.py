import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from scripts.generate_specs import (
    build_spec, build_anchor_spec, build_recovery_spec, write_spec,
    generate_main_stages, generate_stage4_t3, generate_recovery_specs
)

class TestGenerateSpecs(unittest.TestCase):

    @patch("scripts.generate_specs.build_spec_template")
    def test_build_spec(self, mock_build_spec_template):
        mock_build_spec_template.return_value = {"id": "test-spec"}
        result = build_spec("track1", 1, "variant1", "baseline1", {"param1": "value1"})

        self.assertEqual(result["id"], "test-spec")
        mock_build_spec_template.assert_called_once()

    @patch("scripts.generate_specs.build_spec_template")
    def test_build_anchor_spec(self, mock_build_spec_template):
        mock_build_spec_template.return_value = {"id": "anchor-spec"}
        result = build_anchor_spec(1)

        self.assertEqual(result["id"], "anchor-spec")
        mock_build_spec_template.assert_called_once()

    @patch("scripts.generate_specs.build_spec_template")
    def test_build_recovery_spec(self, mock_build_spec_template):
        mock_build_spec_template.return_value = {"id": "recovery-spec"}
        item = {"stage": 1, "track_id": "track1", "id": "recovery1", "params": {}}
        result = build_recovery_spec(item)

        self.assertEqual(result["id"], "recovery-spec")
        mock_build_spec_template.assert_called_once()

    @patch("scripts.generate_specs.Path.write_text")
    def test_write_spec(self, mock_write_text):
        path = Path("specs/test.yaml")
        payload = {"key": "value"}
        write_spec(path, payload)

        mock_write_text.assert_called_once()

    @patch("scripts.generate_specs.write_spec")
    @patch("scripts.generate_specs.build_anchor_spec")
    def test_generate_main_stages(self, mock_build_anchor_spec, mock_write_spec):
        mock_build_anchor_spec.return_value = {"id": "anchor-spec"}
        generate_main_stages()

        mock_build_anchor_spec.assert_called()
        mock_write_spec.assert_called()

    @patch("scripts.generate_specs.write_spec")
    @patch("scripts.generate_specs.build_anchor_spec")
    def test_generate_stage4_t3(self, mock_build_anchor_spec, mock_write_spec):
        mock_build_anchor_spec.return_value = {"id": "anchor-spec"}
        generate_stage4_t3()

        mock_build_anchor_spec.assert_called()
        mock_write_spec.assert_called()

    @patch("scripts.generate_specs.write_spec")
    @patch("scripts.generate_specs.build_recovery_spec")
    def test_generate_recovery_specs(self, mock_build_recovery_spec, mock_write_spec):
        mock_build_recovery_spec.return_value = {"id": "recovery-spec"}
        generate_recovery_specs()

        mock_build_recovery_spec.assert_called()
        mock_write_spec.assert_called()

if __name__ == "__main__":
    unittest.main()