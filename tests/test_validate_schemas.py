import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools" / "validate_schemas.py"

sys.path.insert(0, str(ROOT))

from tools.validate_schemas import ValidationError, load_yaml, validate_case, validate_result


class SchemaValidationTests(unittest.TestCase):
    def test_all_example_cases_validate(self):
        for path in sorted((ROOT / "cases").glob("*/*.yaml")):
            with self.subTest(path=path):
                validate_case(path)

    def test_case_semantics_reject_unknown_dimension_parameter(self):
        source = ROOT / "cases" / "matmul" / "square.yaml"
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            text = source.read_text(encoding="utf-8").replace("i: N", "i: Missing", 1)
            handle.write(text)
            temp_path = Path(handle.name)
        try:
            with self.assertRaises(ValidationError):
                validate_case(temp_path)
        finally:
            temp_path.unlink()

    def test_result_schema_accepts_valid_record(self):
        record = {
            "case_name": "rank4_two_bond",
            "einsum": "ijab,abkl->ijkl",
            "backend": "cpp:reference",
            "language": "cpp",
            "device": "cpu",
            "dtype": "float64",
            "parameters": {"D": 4, "X": 8},
            "input_layouts": {
                "A": {
                    "labels": ["i", "j", "a", "b"],
                    "physical_order": ["i", "j", "a", "b"],
                    "strides": [512, 128, 8, 1],
                },
                "B": {
                    "labels": ["a", "b", "k", "l"],
                    "physical_order": ["a", "b", "k", "l"],
                    "strides": [128, 16, 4, 1],
                },
            },
            "output_layout": {
                "labels": ["i", "j", "k", "l"],
                "physical_order": ["i", "j", "k", "l"],
                "strides": [64, 16, 4, 1],
            },
            "timing": {
                "warmup": 1,
                "repeat": 3,
                "times_sec": [0.3, 0.2, 0.4],
                "min_sec": 0.2,
                "median_sec": 0.3,
                "mean_sec": 0.3,
                "stddev_sec": 0.0816496580927726,
            },
            "performance": {
                "flops": 4096,
                "gflops": 0.000013653333333333334,
                "bytes_estimated": 8192,
                "bandwidth_gbs_estimated": 0.000027306666666666668,
            },
            "validation": {
                "status": "passed",
                "max_abs_error": 0.0,
                "max_rel_error": 0.0,
            },
            "system": {
                "cpu": "unknown",
                "gpu": None,
                "num_threads": 1,
                "compiler": "unknown",
                "compiler_version": None,
                "library_version": {"reference": "unknown"},
            },
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(record, handle)
            temp_path = Path(handle.name)
        try:
            validate_result(temp_path)
        finally:
            temp_path.unlink()

    def test_result_semantics_reject_repeat_mismatch(self):
        record = {
            "case_name": "matmul_square",
            "einsum": "ik,kj->ij",
            "backend": "python:numpy",
            "language": "python",
            "device": "cpu",
            "dtype": "float64",
            "parameters": {"N": 16},
            "input_layouts": {
                "A": {"labels": ["i", "k"], "physical_order": ["i", "k"], "strides": [16, 1]},
                "B": {"labels": ["k", "j"], "physical_order": ["k", "j"], "strides": [16, 1]},
            },
            "output_layout": {"labels": ["i", "j"], "physical_order": ["i", "j"], "strides": [16, 1]},
            "timing": {
                "warmup": 0,
                "repeat": 2,
                "times_sec": [0.1],
                "min_sec": 0.1,
                "median_sec": 0.1,
                "mean_sec": 0.1,
                "stddev_sec": 0.0,
            },
            "performance": {"flops": 8192, "gflops": 0.00008192},
            "validation": {"status": "skipped"},
            "system": {"cpu": "unknown", "gpu": None, "num_threads": 1},
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(record, handle)
            temp_path = Path(handle.name)
        try:
            with self.assertRaises(ValidationError):
                validate_result(temp_path)
        finally:
            temp_path.unlink()

    def test_cli_validates_case_files(self):
        result = subprocess.run(
            [
                sys.executable,
                str(VALIDATOR),
                "--kind",
                "case",
                str(ROOT / "cases" / "matmul" / "square.yaml"),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)

    def test_yaml_loader_parses_inline_lists_and_numbers(self):
        data = load_yaml(ROOT / "cases" / "rank4" / "two_bond.yaml")
        self.assertEqual(data["parameters"]["X"], [4, 8, 16, 32, 64, 128])
        self.assertEqual(data["validation"]["tolerance"]["rtol"], 1.0e-10)
        self.assertEqual(data["variants"]["contraction_permutations"][0], "ijab,abkl->ijkl")


if __name__ == "__main__":
    unittest.main()
