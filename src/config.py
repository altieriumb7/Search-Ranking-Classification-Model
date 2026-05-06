from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

QUERIES_PATH = DATA_DIR / "queries.csv"
DOCUMENTS_PATH = DATA_DIR / "documents.csv"
QRELS_PATH = DATA_DIR / "qrels.csv"
PIPELINE_PATH = MODELS_DIR / "ranking_pipeline.pkl"
METRICS_PATH = REPORTS_DIR / "metrics.json"
SUMMARY_PATH = REPORTS_DIR / "evaluation_summary.md"
RUN_HISTORY_PATH = REPORTS_DIR / "run_history.jsonl"

RANDOM_SEED = 42
