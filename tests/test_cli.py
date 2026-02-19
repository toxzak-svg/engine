import unittest
from unittest.mock import patch, MagicMock
from exp.cli import _cmd_generate, _cmd_package, _cmd_test, _cmd_publish

class TestCLICommands(unittest.TestCase):

    @patch("exp.cli.Path")
    @patch("exp.cli.ExperimentSpec.from_dict")
    def test_generate_command(self, mock_from_dict, mock_path):
        args = MagicMock()
        args.stage = 1
        args.type = "main"
        mock_path.return_value = MagicMock()
        mock_from_dict.return_value = MagicMock()

        result = _cmd_generate(args)

        self.assertEqual(result, 0)
        mock_from_dict.assert_called_once()
        mock_path.assert_called_once()

    @patch("builtins.print")
    def test_package_command(self, mock_print):
        args = MagicMock()
        args.version = "1.0.0"
        args.output = "dist/"

        result = _cmd_package(args)

        self.assertEqual(result, 0)
        mock_print.assert_called_with("Packaging version 1.0.0 to dist/")

    @patch("builtins.print")
    def test_test_command(self, mock_print):
        args = MagicMock()
        args.ci = True

        result = _cmd_test(args)

        self.assertEqual(result, 0)
        mock_print.assert_called_with("Running tests...")

    @patch("builtins.print")
    def test_publish_command(self, mock_print):
        args = MagicMock()
        args.repository = "pypi"

        result = _cmd_publish(args)

        self.assertEqual(result, 0)
        mock_print.assert_called_with("Publishing package...")

if __name__ == "__main__":
    unittest.main()