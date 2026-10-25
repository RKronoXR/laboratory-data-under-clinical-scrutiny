"""End-to-end checks on the supplied scientific reference results."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_workspace = ROOT / "results/test_workspace"
        cls.test_workspace.mkdir(parents=True, exist_ok=True)

    def command(self, *args):
        result = subprocess.run([sys.executable, str(ROOT / "run.py"), *args], cwd=ROOT,
                                text=True, encoding="utf-8", capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_main_clinical_tables_from_regional_predictions(self):
        with tempfile.TemporaryDirectory(dir=self.test_workspace) as folder:
            self.command("reproduce", "--out", folder)
            report = json.loads((Path(folder) / "validation.json").read_text())
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["records"], 88560)
            self.assertEqual(report["fitted_models"], 120)
            self.assertEqual(report["unique_regions"], 369)
            self.assertEqual(report["positive_regions"], 184)
            self.assertEqual(report["negative_regions"], 185)
            self.assertEqual(report["published_values_checked"], 544)
            self.assertLess(report["maximum_published_difference"], 1e-10)

    def test_plan_entrypoints_do_not_require_gpu_frameworks(self):
        self.command("main", "plan")
        self.command("benchmark", "plan")
        self.command("additional", "plan")

    def test_benchmark_reference_reproduction(self):
        with tempfile.TemporaryDirectory(dir=self.test_workspace) as folder:
            self.command("benchmark", "reproduce", "--work-dir", folder)
            self.assertTrue(any(Path(folder).rglob("*.csv")))

    def test_released_file_integrity(self):
        output = self.command("verify")
        self.assertEqual(json.loads(output)["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
