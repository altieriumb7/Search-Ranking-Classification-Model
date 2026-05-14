from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from src.benchmark import BENCHMARK_CASES_PATH, load_benchmark_cases, summarize_cases
from src.config import PIPELINE_PATH, PROJECT_ROOT, REPORTS_DIR, load_runtime_settings
from src.data_loader import load_dataset, qrels_by_query
from src.model import load_pickle

_LAST_LIVE_RUN_TS = 0.0
_LIVE_RUN_COUNT = 0


def resolve_live_api_key(
    session_key: str | None = None, require_session_key: bool = False
) -> str | None:
    key = (session_key or "").strip()
    if key:
        return key
    if require_session_key:
        return None
    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    return env_key or None


def can_execute_live_runs(api_key: str | None) -> tuple[bool, str]:
    settings = load_runtime_settings()
    if settings.demo_mode:
        return False, "Live runs are disabled because DEMO_MODE=true."
    if not settings.allow_live_runs:
        return False, "Live runs are disabled because ALLOW_LIVE_RUNS=false."
    if not api_key:
        return False, "No API key available. Set OPENAI_API_KEY or provide a session key."
    return True, "ok"


def _check_live_limits() -> tuple[bool, str]:
    global _LAST_LIVE_RUN_TS, _LIVE_RUN_COUNT
    now = time.time()
    cooldown = int(os.getenv("LIVE_RUN_COOLDOWN_SECONDS", "60"))
    max_runs = int(os.getenv("LIVE_RUN_MAX_RUNS_PER_PROCESS", "3"))
    if _LIVE_RUN_COUNT >= max_runs:
        return False, f"Live run limit reached for this process ({max_runs})."
    if (now - _LAST_LIVE_RUN_TS) < cooldown:
        wait_seconds = int(cooldown - (now - _LAST_LIVE_RUN_TS))
        return False, f"Live run cooldown active. Retry in {wait_seconds}s."
    _LAST_LIVE_RUN_TS = now
    _LIVE_RUN_COUNT += 1
    return True, "ok"


def _extract_output_text(response_json: dict[str, Any]) -> str:
    output_text = response_json.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    for item in response_json.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                text = content.get("text", "")
                if isinstance(text, str) and text.strip():
                    return text.strip()
    raise ValueError("Could not extract output text from OpenAI response payload.")


def _call_openai_live_judge(
    api_key: str,
    model: str,
    query_text: str,
    learned_top: dict[str, Any],
    baseline_top: dict[str, Any],
) -> dict[str, Any]:
    system_text = (
        "You are a strict ranking evaluator. Return only valid JSON with keys: "
        "winner (learned|baseline|tie), rationale (string), confidence (0..1), risk_note (string)."
    )
    user_text = (
        f"Query: {query_text}\n\n"
        f"Learned top result:\nTitle: {learned_top['title']}\nBody: {learned_top['body']}\n"
        f"Score: {learned_top['score']}\n\n"
        f"Baseline top result:\nTitle: {baseline_top['title']}\nBody: {baseline_top['body']}\n"
        f"Score: {baseline_top['score']}\n\n"
        "Which top result is better for the query intent?"
    )
    payload = {
        "model": model,
        "input": user_text,
        "instructions": system_text,
        "temperature": 0,
        "max_output_tokens": 220,
    }
    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=45,
    )
    response.raise_for_status()
    raw = _extract_output_text(response.json())
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Judge response was not valid JSON: {raw}") from exc
    winner = str(parsed.get("winner", "")).lower()
    if winner not in {"learned", "baseline", "tie"}:
        raise ValueError(f"Unexpected winner value: {winner}")
    return {
        "winner": winner,
        "rationale": str(parsed.get("rationale", "")),
        "confidence": float(parsed.get("confidence", 0.0)),
        "risk_note": str(parsed.get("risk_note", "")),
    }


def run_live_benchmark(
    api_key: str,
    model: str = "gpt-4o-mini",
    output_root: Path = REPORTS_DIR / "live_benchmark",
    cases_path: Path = BENCHMARK_CASES_PATH,
) -> dict[str, Any]:
    allowed, reason = can_execute_live_runs(api_key)
    if not allowed:
        raise RuntimeError(reason)

    limit_ok, limit_reason = _check_live_limits()
    if not limit_ok:
        raise RuntimeError(limit_reason)

    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"Missing {PIPELINE_PATH}. Run `python -m src.train_ranker` before live benchmark."
        )

    max_cases = int(os.getenv("LIVE_RUN_MAX_CASES", "3"))
    cases = load_benchmark_cases(cases_path)[:max_cases]
    queries, _documents, qrels = load_dataset()
    query_by_text = {query.text: query for query in queries}
    judged = qrels_by_query(qrels)
    pipeline = load_pickle(PIPELINE_PATH)

    evaluated_cases: list[dict[str, Any]] = []
    for case in cases:
        query_text = str(case["query_text"])
        query = query_by_text.get(query_text)
        if query is None:
            evaluated_cases.append(
                {
                    "case_id": str(case["case_id"]),
                    "category": str(case["category"]),
                    "input_prompt": query_text,
                    "expected_behavior": str(case["expected_behavior"]),
                    "observed_live_output": {},
                    "qualitative_assessment": "Missing query mapping in dataset.",
                    "status": "warning",
                    "notes": str(case.get("notes", "")),
                    "runtime": {"error": "Unknown query text in benchmark case."},
                }
            )
            continue

        relevance_map = judged.get(query.query_id, {})
        candidate_doc_ids = list(relevance_map.keys()) or None
        baseline_rows = pipeline.rank_baseline(query_text, candidate_doc_ids, top_k=3)
        learned_rows = pipeline.rank_learned(query_text, candidate_doc_ids, top_k=3)
        baseline_top = baseline_rows[0] if baseline_rows else None
        learned_top = learned_rows[0] if learned_rows else None
        if not baseline_top or not learned_top:
            evaluated_cases.append(
                {
                    "case_id": str(case["case_id"]),
                    "category": str(case["category"]),
                    "input_prompt": query_text,
                    "expected_behavior": str(case["expected_behavior"]),
                    "observed_live_output": {},
                    "qualitative_assessment": "No ranking rows produced.",
                    "status": "warning",
                    "notes": str(case.get("notes", "")),
                    "runtime": {"error": "No ranking rows produced."},
                }
            )
            continue

        runtime_error = None
        judge_result: dict[str, Any] | None = None
        try:
            judge_result = _call_openai_live_judge(
                api_key=api_key,
                model=model,
                query_text=query_text,
                learned_top=learned_top,
                baseline_top=baseline_top,
            )
        except Exception as exc:
            runtime_error = str(exc)

        if runtime_error:
            status = "warning"
            assessment = "Judge call failed; case continued with runtime error."
        elif judge_result["winner"] == "learned":
            status = "pass"
            assessment = "Live judge preferred learned top result."
        elif judge_result["winner"] == "baseline":
            status = "fail"
            assessment = "Live judge preferred baseline top result."
        else:
            status = "warning"
            assessment = "Live judge marked the case as tie."

        evaluated_cases.append(
            {
                "case_id": str(case["case_id"]),
                "category": str(case["category"]),
                "input_prompt": query_text,
                "expected_behavior": str(case["expected_behavior"]),
                "observed_live_output": {
                    "learned_top_doc_id": learned_top["doc_id"],
                    "learned_top_title": learned_top["title"],
                    "baseline_top_doc_id": baseline_top["doc_id"],
                    "baseline_top_title": baseline_top["title"],
                    "judge_result": judge_result,
                },
                "qualitative_assessment": assessment,
                "status": status,
                "notes": str(case.get("notes", "")),
                "runtime": {"error": runtime_error},
            }
        )

    summary = summarize_cases(evaluated_cases)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = output_root / timestamp
    report_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "report_type": "live_benchmark",
        "timestamp_utc": timestamp,
        "model": model,
        "max_cases": max_cases,
        "cases_path": str(cases_path.relative_to(PROJECT_ROOT)),
        "results": evaluated_cases,
        "summary": summary,
    }

    json_path = report_dir / "live_results.json"
    csv_path = report_dir / "live_summary.csv"
    md_path = report_dir / "qualitative_case_gallery.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["total_cases", summary["total_cases"]])
        writer.writerow(["pass_count", summary["pass_count"]])
        writer.writerow(["warning_count", summary["warning_count"]])
        writer.writerow(["fail_count", summary["fail_count"]])
        writer.writerow(["pass_rate", summary["pass_rate"]])
    md_path.write_text(
        "\n".join(
            [
                "# Live Qualitative Case Gallery",
                "",
                f"Generated at: {timestamp}",
                "",
                "| case_id | category | status | runtime_error |",
                "| --- | --- | --- | --- |",
            ]
            + [
                f"| {case['case_id']} | {case['category']} | {case['status']} | {case['runtime']['error'] or ''} |"
                for case in evaluated_cases
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return payload
