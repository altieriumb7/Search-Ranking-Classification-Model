import unittest
import shutil
from pathlib import Path

from src.benchmark import BENCHMARK_CASES_PATH, generate_demo_benchmark, load_benchmark_cases, summarize_cases
from src.config import REPORTS_DIR, load_runtime_settings
from src.train_ranker import train
from src.runtime_settings import RuntimeSettings


class BenchmarkDemoTest(unittest.TestCase):
    def test_benchmark_cases_load(self):
        cases = load_benchmark_cases(BENCHMARK_CASES_PATH)
        self.assertGreaterEqual(len(cases), 4)
        self.assertIn("case_id", cases[0])

    def test_demo_benchmark_generation_is_deterministic(self):
        train()
        output_a = REPORTS_DIR / "test_demo_benchmark_a"
        output_b = REPORTS_DIR / "test_demo_benchmark_b"
        try:
            first = generate_demo_benchmark(output_dir=output_a, cases_path=BENCHMARK_CASES_PATH)
            second = generate_demo_benchmark(output_dir=output_b, cases_path=BENCHMARK_CASES_PATH)
            self.assertEqual(first["summary"], second["summary"])
            self.assertEqual(first["results"], second["results"])
        finally:
            shutil.rmtree(output_a, ignore_errors=True)
            shutil.rmtree(output_b, ignore_errors=True)

    def test_summary_metrics_shape(self):
        sample = [
            {"status": "pass", "category": "a"},
            {"status": "fail", "category": "a"},
            {"status": "warning", "category": "b"},
        ]
        summary = summarize_cases(sample)
        self.assertEqual(summary["total_cases"], 3)
        self.assertEqual(summary["pass_count"], 1)
        self.assertEqual(summary["fail_count"], 1)
        self.assertEqual(summary["warning_count"], 1)
        self.assertEqual(summary["category_breakdown"]["a"], 2)

    def test_reports_written_to_output_directory(self):
        train()
        target = REPORTS_DIR / "test_demo_benchmark_output"
        try:
            generate_demo_benchmark(output_dir=target, cases_path=BENCHMARK_CASES_PATH)
            self.assertTrue((target / "demo_results.json").exists())
            self.assertTrue((target / "demo_summary.csv").exists())
            self.assertTrue((target / "qualitative_case_gallery.md").exists())
        finally:
            shutil.rmtree(target, ignore_errors=True)


class RuntimeSettingsTest(unittest.TestCase):
    def test_demo_mode_disables_live_execution(self):
        settings = RuntimeSettings(
            demo_mode=True,
            allow_live_runs=True,
            require_session_api_key=False,
            benchmark_mode="demo",
            openai_api_key_present=False,
            default_config_path=Path("x"),
            reports_dir=Path("y"),
        )
        self.assertFalse(settings.live_runs_enabled)

    def test_missing_openai_key_does_not_crash_settings(self):
        settings = load_runtime_settings()
        self.assertIsInstance(settings.openai_api_key_present, bool)


if __name__ == "__main__":
    unittest.main()
