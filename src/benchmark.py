from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from src.config import PIPELINE_PATH, PROJECT_ROOT, REPORTS_DIR
from src.data_loader import load_dataset, qrels_by_query
from src.model import load_pickle
from src.train_ranker import train

BENCHMARK_CASES_PATH = PROJECT_ROOT / "benchmarks" / "qualitative_redteam_cases.yaml"
DEMO_BENCHMARK_DIR = REPORTS_DIR / "demo_benchmark"


def load_benchmark_cases(path: Path = BENCHMARK_CASES_PATH) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    cases = payload.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError(f"Invalid benchmark cases format in {path}.")
    return cases


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(str(case["status"]).lower() for case in cases)
    category_counts = Counter(str(case["category"]) for case in cases)
    total = len(cases)
    passed = status_counts.get("pass", 0)
    warnings = status_counts.get("warning", 0)
    failed = status_counts.get("fail", 0)
    pass_rate = (passed / total) if total else 0.0
    return {
        "total_cases": total,
        "pass_count": passed,
        "warning_count": warnings,
        "fail_count": failed,
        "pass_rate": round(pass_rate, 4),
        "categories_covered": sorted(category_counts.keys()),
        "category_breakdown": dict(sorted(category_counts.items())),
    }


def _status_for_relevance(learned_rel: int, baseline_rel: int) -> str:
    if learned_rel >= 2:
        status = "pass"
    elif learned_rel == 1:
        status = "warning"
    else:
        status = "fail"
    if baseline_rel > learned_rel and status == "pass":
        return "warning"
    if baseline_rel - learned_rel >= 2:
        return "fail"
    return status


def generate_demo_benchmark(
    output_dir: Path = DEMO_BENCHMARK_DIR,
    cases_path: Path = BENCHMARK_CASES_PATH,
) -> dict[str, Any]:
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"Missing {PIPELINE_PATH}. Run `python -m src.train_ranker` before generating benchmark."
        )

    queries, _documents, qrels = load_dataset()
    judged = qrels_by_query(qrels)
    query_by_text = {query.text: query for query in queries}
    try:
        pipeline = load_pickle(PIPELINE_PATH)
    except Exception:
        train()
        pipeline = load_pickle(PIPELINE_PATH)

    cases = []
    for case in load_benchmark_cases(cases_path):
        query_text = str(case["query_text"])
        query = query_by_text.get(query_text)
        if query is None:
            raise ValueError(f"Unknown benchmark query text: {query_text}")

        relevance_map = judged.get(query.query_id, {})
        candidate_doc_ids = list(relevance_map.keys()) or None
        baseline_rows = pipeline.rank_baseline(query_text, candidate_doc_ids, top_k=3)
        learned_rows = pipeline.rank_learned(query_text, candidate_doc_ids, top_k=3)
        if not baseline_rows or not learned_rows:
            raise ValueError(f"No ranking rows produced for benchmark case: {case['case_id']}")

        baseline_top = baseline_rows[0]
        learned_top = learned_rows[0]
        baseline_rel = int(relevance_map.get(str(baseline_top["doc_id"]), 0))
        learned_rel = int(relevance_map.get(str(learned_top["doc_id"]), 0))
        status = _status_for_relevance(learned_rel, baseline_rel)

        notes = str(case.get("notes", "")).strip()
        if baseline_rel > learned_rel:
            notes = (notes + " " if notes else "") + "Baseline judged relevance is higher than learned top-1."

        cases.append(
            {
                "case_id": str(case["case_id"]),
                "category": str(case["category"]),
                "input_prompt": query_text,
                "expected_behavior": str(case["expected_behavior"]),
                "observed_demo_output": {
                    "learned_top_doc_id": learned_top["doc_id"],
                    "learned_top_title": learned_top["title"],
                    "learned_top_score": round(float(learned_top["score"]), 6),
                    "learned_top_relevance": learned_rel,
                    "baseline_top_doc_id": baseline_top["doc_id"],
                    "baseline_top_title": baseline_top["title"],
                    "baseline_top_score": round(float(baseline_top["score"]), 6),
                    "baseline_top_relevance": baseline_rel,
                },
                "qualitative_assessment": (
                    "Learned ranker retrieved a highly relevant top result."
                    if status == "pass"
                    else "Learned ranker top result is only marginally relevant."
                    if status == "warning"
                    else "Learned ranker top result is not relevant enough for this case."
                ),
                "status": status,
                "notes": notes,
                "runtime": {"error": None},
            }
        )

    summary = summarize_cases(cases)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_payload = {
        "report_type": "demo_benchmark",
        "report_label": "Deterministic local benchmark generated from bundled dataset and model artifact.",
        "cases_path": str(cases_path.relative_to(PROJECT_ROOT)),
        "results": cases,
        "summary": summary,
    }

    json_path = output_dir / "demo_results.json"
    csv_path = output_dir / "demo_summary.csv"
    md_path = output_dir / "qualitative_case_gallery.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2, sort_keys=True)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["total_cases", summary["total_cases"]])
        writer.writerow(["pass_count", summary["pass_count"]])
        writer.writerow(["warning_count", summary["warning_count"]])
        writer.writerow(["fail_count", summary["fail_count"]])
        writer.writerow(["pass_rate", summary["pass_rate"]])
        for category, count in summary["category_breakdown"].items():
            writer.writerow([f"category:{category}", count])

    lines = [
        "# Qualitative Case Gallery",
        "",
        "This file is generated by `python -m src.generate_demo_benchmark`.",
        "",
        "| case_id | category | status | learned_top_doc_id | learned_top_relevance | baseline_top_doc_id | baseline_top_relevance |",
        "| --- | --- | --- | --- | ---: | --- | ---: |",
    ]
    for case in cases:
        observed = case["observed_demo_output"]
        lines.append(
            "| {case_id} | {category} | {status} | {learned} | {lrel} | {baseline} | {brel} |".format(
                case_id=case["case_id"],
                category=case["category"],
                status=case["status"],
                learned=observed["learned_top_doc_id"],
                lrel=observed["learned_top_relevance"],
                baseline=observed["baseline_top_doc_id"],
                brel=observed["baseline_top_relevance"],
            )
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result_payload


def load_benchmark_report(report_dir: Path) -> dict[str, Any]:
    json_path = report_dir / "demo_results.json"
    if not json_path.exists():
        json_path = report_dir / "live_results.json"
    with json_path.open(encoding="utf-8") as f:
        return json.load(f)


def list_benchmark_report_dirs(reports_dir: Path = REPORTS_DIR) -> list[Path]:
    if not reports_dir.exists():
        return []
    candidates = []
    for path in sorted(reports_dir.iterdir()):
        if path.is_dir() and (
            (path / "demo_results.json").exists() or (path / "live_results.json").exists()
        ):
            candidates.append(path)
        if path.is_dir() and path.name == "live_benchmark":
            for subdir in sorted(path.iterdir()):
                if subdir.is_dir() and (subdir / "live_results.json").exists():
                    candidates.append(subdir)
    return candidates
