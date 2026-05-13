from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.benchmark import DEMO_BENCHMARK_DIR, BENCHMARK_CASES_PATH, generate_demo_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic demo benchmark artifacts.")
    parser.add_argument(
        "--output",
        default=str(DEMO_BENCHMARK_DIR),
        help="Output directory for demo benchmark artifacts.",
    )
    parser.add_argument(
        "--cases",
        default=str(BENCHMARK_CASES_PATH),
        help="Benchmark case YAML file path.",
    )
    args = parser.parse_args()
    payload = generate_demo_benchmark(output_dir=Path(args.output), cases_path=Path(args.cases))
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
