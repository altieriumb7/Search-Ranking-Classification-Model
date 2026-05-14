import os
from pathlib import Path

from src.runtime_settings import RuntimeSettings, env_bool, env_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = env_path("REPORTS_DIR", PROJECT_ROOT / "reports", PROJECT_ROOT)

QUERIES_PATH = DATA_DIR / "queries.csv"
DOCUMENTS_PATH = DATA_DIR / "documents.csv"
QRELS_PATH = DATA_DIR / "qrels.csv"
PIPELINE_PATH = MODELS_DIR / "ranking_pipeline.pkl"
METRICS_PATH = REPORTS_DIR / "metrics.json"
SUMMARY_PATH = REPORTS_DIR / "evaluation_summary.md"
RUN_HISTORY_PATH = REPORTS_DIR / "run_history.jsonl"

RANDOM_SEED = 42

DEMO_MODE = env_bool("DEMO_MODE", True)
ALLOW_LIVE_RUNS = env_bool("ALLOW_LIVE_RUNS", False)
REQUIRE_SESSION_API_KEY = env_bool("REQUIRE_SESSION_API_KEY", False)
BENCHMARK_MODE = os.getenv("BENCHMARK_MODE", "demo")
DEFAULT_CONFIG_PATH = env_path(
    "DEFAULT_CONFIG_PATH", PROJECT_ROOT / "evals" / "config.yaml", PROJECT_ROOT
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


def load_runtime_settings() -> RuntimeSettings:
    return RuntimeSettings(
        demo_mode=env_bool("DEMO_MODE", DEMO_MODE),
        allow_live_runs=env_bool("ALLOW_LIVE_RUNS", ALLOW_LIVE_RUNS),
        require_session_api_key=env_bool("REQUIRE_SESSION_API_KEY", REQUIRE_SESSION_API_KEY),
        benchmark_mode=os.getenv("BENCHMARK_MODE", BENCHMARK_MODE),
        openai_api_key_present=bool(os.getenv("OPENAI_API_KEY", OPENAI_API_KEY).strip()),
        default_config_path=env_path("DEFAULT_CONFIG_PATH", DEFAULT_CONFIG_PATH, PROJECT_ROOT),
        reports_dir=env_path("REPORTS_DIR", REPORTS_DIR, PROJECT_ROOT),
    )
