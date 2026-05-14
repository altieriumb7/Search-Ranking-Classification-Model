import os
import unittest

from src.live_benchmark import can_execute_live_runs, resolve_live_api_key


class LiveBenchmarkGuardsTest(unittest.TestCase):
    def setUp(self):
        self.previous = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.previous)

    def test_demo_mode_blocks_live_runs(self):
        os.environ["DEMO_MODE"] = "true"
        os.environ["ALLOW_LIVE_RUNS"] = "true"
        allowed, reason = can_execute_live_runs("sk-test")
        self.assertFalse(allowed)
        self.assertIn("DEMO_MODE=true", reason)

    def test_allow_live_runs_false_blocks_execution(self):
        os.environ["DEMO_MODE"] = "false"
        os.environ["ALLOW_LIVE_RUNS"] = "false"
        allowed, reason = can_execute_live_runs("sk-test")
        self.assertFalse(allowed)
        self.assertIn("ALLOW_LIVE_RUNS=false", reason)

    def test_missing_api_key_blocks_execution(self):
        os.environ["DEMO_MODE"] = "false"
        os.environ["ALLOW_LIVE_RUNS"] = "true"
        allowed, reason = can_execute_live_runs(None)
        self.assertFalse(allowed)
        self.assertIn("No API key", reason)

    def test_valid_live_settings_enable_execution(self):
        os.environ["DEMO_MODE"] = "false"
        os.environ["ALLOW_LIVE_RUNS"] = "true"
        allowed, reason = can_execute_live_runs("sk-test")
        self.assertTrue(allowed)
        self.assertEqual(reason, "ok")

    def test_session_key_has_priority(self):
        os.environ["OPENAI_API_KEY"] = "env-key"
        key = resolve_live_api_key("session-key")
        self.assertEqual(key, "session-key")

    def test_require_session_key_disables_env_fallback(self):
        os.environ["OPENAI_API_KEY"] = "env-key"
        key = resolve_live_api_key(None, require_session_key=True)
        self.assertIsNone(key)


if __name__ == "__main__":
    unittest.main()
