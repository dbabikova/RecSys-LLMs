"""Tests of standalone verification and its failure behaviour; no model calls."""

import ast
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_example.py"
EVIDENCE = ROOT / "data" / "ranker_example" / "expected.json"


class IndependentVerifierTests(unittest.TestCase):
    def test_only_standard_library_imports(self):
        source = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(source):
            if isinstance(node, ast.Import):
                imports.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module.split(".")[0])
        self.assertTrue(set(imports) <= sys.stdlib_module_names)
        self.assertNotIn("importlib", imports)
        self.assertNotIn("subprocess", imports)

    def test_corrupted_expectation_exits_nonzero_then_restored_source_passes(self):
        if not EVIDENCE.exists():
            self.skipTest("Generate the real example first; no models/data generation in this test")
        original = EVIDENCE.read_bytes()
        try:
            broken = json.loads(original)
            broken["metrics"]["cbf"] += .01
            EVIDENCE.write_text(json.dumps(broken, ensure_ascii=False), encoding="utf-8")
            failed = subprocess.run(
                [sys.executable, "-B", str(SCRIPT)], cwd=ROOT,
                text=True, encoding="utf-8", capture_output=True,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("CBF nDCG@10", failed.stderr)
            report = (ROOT / "reports/verify_example.md").read_text(encoding="utf-8")
            self.assertIn("**FAIL", report)
            self.assertNotIn("**PASS", report)
        finally:
            EVIDENCE.write_bytes(original)
            restored = subprocess.run(
                [sys.executable, "-B", str(SCRIPT)], cwd=ROOT,
                text=True, encoding="utf-8", capture_output=True,
            )
            self.assertEqual(restored.returncode, 0, restored.stderr)
            result = json.loads(restored.stdout)
            self.assertEqual(result["failed"], 0)
            self.assertGreater(result["passed"], 0)
        # Record the negative control as aggregate test evidence, never a model result.
        (ROOT / "data/ranker_example/failure_test.json").write_text(json.dumps({
            "corrupted_field": "metrics.cbf", "perturbation": .01,
            "corrupt_exit_code": failed.returncode, "restored_exit_code": restored.returncode,
            "raw_csv_modified": False,
        }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
