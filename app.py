from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from src.benchmark import (
    DEMO_BENCHMARK_DIR,
    generate_demo_benchmark,
    list_benchmark_report_dirs,
    load_benchmark_report,
)
from src.config import (
    METRICS_PATH,
    PIPELINE_PATH,
    PROJECT_ROOT,
    REPORTS_DIR,
    load_runtime_settings,
)
from src.data_loader import load_dataset, qrels_by_query
from src.model import load_pickle

st.set_page_config(page_title="Search Ranking LTR Demo", layout="wide")


def _relative_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def _config_candidates(default_path: Path) -> list[Path]:
    candidates = [default_path]
    candidates.extend(sorted((PROJECT_ROOT / "benchmarks").glob("*.yaml")))
    evals_dir = PROJECT_ROOT / "evals"
    if evals_dir.exists():
        candidates.extend(sorted(evals_dir.rglob("*.yaml")))
    unique = []
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


@st.cache_resource
def load_pipeline(path_str: str):
    path = Path(path_str)
    if not path.exists():
        return None
    return load_pickle(path)


@st.cache_data
def load_metrics(path_str: str) -> dict[str, Any] | None:
    path = Path(path_str)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_query_data():
    queries, _documents, qrels = load_dataset()
    return queries, qrels


def metric_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for label, key in [("BM25 baseline", "bm25_baseline"), ("Learned ranker", "learned_ranker")]:
        model_metrics = metrics["metrics"][key]
        rows.append(
            {
                "model": label,
                "NDCG@10": model_metrics["ndcg@10"],
                "MAP": model_metrics["map"],
                "MRR": model_metrics["mrr"],
            }
        )
    return rows


def render_ranked_results(title: str, rows: list[dict[str, Any]], relevance: dict[str, int]) -> None:
    st.subheader(title)
    for row in rows:
        rel = relevance.get(row["doc_id"])
        label = f"Rank {row['rank']} | {row['doc_id']} | score {row['score']:.3f}"
        if rel is not None:
            label += f" | relevance {rel}"
        with st.container(border=True):
            st.caption(label)
            st.markdown(f"**{row['title']}**")
            st.write(row["body"])


settings = load_runtime_settings()

st.sidebar.title("Runtime Settings")
st.sidebar.write(f"Current mode: `{settings.mode_label}`")
st.sidebar.write(f"`DEMO_MODE`: `{settings.demo_mode}`")
st.sidebar.write(f"`ALLOW_LIVE_RUNS`: `{settings.allow_live_runs}`")
st.sidebar.write(f"`BENCHMARK_MODE`: `{settings.benchmark_mode}`")
st.sidebar.write(f"`OPENAI_API_KEY present`: `{settings.openai_api_key_present}`")
st.sidebar.write(f"`DEFAULT_CONFIG_PATH`: `{_relative_label(settings.default_config_path)}`")
st.sidebar.write(f"`REPORTS_DIR`: `{_relative_label(settings.reports_dir)}`")

if settings.live_runs_enabled and not settings.openai_api_key_present:
    session_key = st.sidebar.text_input(
        "Session OPENAI_API_KEY (not persisted)",
        type="password",
        help="Stored only in this browser session.",
    )
    if session_key:
        st.session_state["session_api_key"] = session_key
    st.sidebar.write(
        f"`Session key present`: `{bool(st.session_state.get('session_api_key'))}`"
    )
    if not st.session_state.get("session_api_key"):
        st.sidebar.warning(
            "Live mode is enabled but no API key is available. Add `OPENAI_API_KEY` as an env var "
            "or provide a session key above."
        )

config_paths = _config_candidates(settings.default_config_path)
selected_config_label = st.sidebar.selectbox(
    "Selected config file",
    [_relative_label(path) for path in config_paths],
    index=0,
)

benchmark_dirs = list_benchmark_report_dirs(settings.reports_dir)
if (settings.reports_dir / "demo_benchmark").exists() and (settings.reports_dir / "demo_benchmark" not in benchmark_dirs):
    benchmark_dirs.append(settings.reports_dir / "demo_benchmark")
if not benchmark_dirs and DEMO_BENCHMARK_DIR.exists():
    benchmark_dirs.append(DEMO_BENCHMARK_DIR)
benchmark_label_to_path = {_relative_label(path): path for path in benchmark_dirs}
selected_report_dir = (
    st.sidebar.selectbox(
        "Selected benchmark report",
        list(benchmark_label_to_path.keys()),
        index=0,
    )
    if benchmark_dirs
    else "(none)"
)

st.title("Search Ranking Dashboard")
st.caption(
    "Safe public demo with deterministic benchmark artifacts. Live external API calls are disabled by default."
)

if settings.demo_mode:
    st.warning(
        "Public demo mode: live model calls are disabled. This demo uses sample benchmark reports. "
        "Clone the repo and configure keys locally for live workflows."
    )

tabs = st.tabs(["Ranking Demo", "Benchmark & Qualitative Evaluation"])

with tabs[0]:
    pipeline = load_pipeline(str(PIPELINE_PATH))
    metrics = load_metrics(str(METRICS_PATH))

    if pipeline is None:
        st.error(
            "Missing trained model artifact. Run `python -m src.train_ranker` to enable ranking demo."
        )
    elif metrics is None:
        st.error(
            "Missing metrics artifact. Run `python -m src.evaluate_ranker` to enable metrics display."
        )
    else:
        queries, qrels = load_query_data()
        judgments = qrels_by_query(qrels)
        examples = [query.text for query in queries if query.split == "test"][:3]
        query_lookup = {query.text: query for query in queries}

        selected_example = st.selectbox("Example queries", examples, index=0)
        query_text = st.text_input("Query", value=selected_example)
        top_k = st.slider("Top-k results", min_value=3, max_value=10, value=5)

        known_query = query_lookup.get(query_text)
        candidate_doc_ids = None
        relevance = {}
        if known_query and known_query.query_id in judgments:
            candidate_doc_ids = list(judgments[known_query.query_id].keys())
            relevance = judgments[known_query.query_id]

        baseline_rows = pipeline.rank_baseline(query_text, candidate_doc_ids, top_k=top_k)
        learned_rows = pipeline.rank_learned(query_text, candidate_doc_ids, top_k=top_k)

        st.markdown("### Offline Test Metrics")
        rows = metric_rows(metrics)
        st.dataframe(rows, hide_index=True, use_container_width=True)

        chart_rows = []
        for row in rows:
            chart_rows.extend(
                [
                    {"model": row["model"], "metric": "NDCG@10", "value": row["NDCG@10"]},
                    {"model": row["model"], "metric": "MAP", "value": row["MAP"]},
                    {"model": row["model"], "metric": "MRR", "value": row["MRR"]},
                ]
            )
        st.bar_chart(chart_rows, x="metric", y="value", color="model")

        left, right = st.columns(2)
        with left:
            render_ranked_results("BM25 Baseline", baseline_rows, relevance)
        with right:
            render_ranked_results("Learned Ranker", learned_rows, relevance)

        st.caption(f"Ranker backend: {metrics['ranker_backend']}.")

with tabs[1]:
    st.markdown("### Benchmark & Qualitative Evaluation")
    if settings.demo_mode:
        st.info("Hosted demo uses deterministic local benchmark artifacts only.")
    if st.button("Regenerate demo benchmark (no API calls)"):
        try:
            generate_demo_benchmark(output_dir=settings.reports_dir / "demo_benchmark")
            st.success("Demo benchmark regenerated.")
            st.cache_data.clear()
        except Exception as exc:
            st.error(f"Benchmark generation failed: {exc}")

    if selected_report_dir == "(none)":
        st.warning("No benchmark report found. Generate one with the button above.")
    else:
        report_dir = benchmark_label_to_path[selected_report_dir]
        try:
            report = load_benchmark_report(report_dir)
        except Exception as exc:
            st.error(f"Could not load benchmark report `{selected_report_dir}`: {exc}")
        else:
            summary = report["summary"]
            results = report["results"]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total", summary["total_cases"])
            c2.metric("Pass", summary["pass_count"])
            c3.metric("Warning", summary["warning_count"])
            c4.metric("Fail", summary["fail_count"])
            c5.metric("Pass rate", f"{summary['pass_rate'] * 100:.1f}%")

            st.markdown("#### Category breakdown")
            breakdown_rows = [
                {"category": category, "count": count}
                for category, count in summary["category_breakdown"].items()
            ]
            st.dataframe(breakdown_rows, hide_index=True, use_container_width=True)

            st.markdown("#### Qualitative case gallery")
            for case in results:
                with st.expander(f"{case['case_id']} | {case['category']} | {case['status']}"):
                    st.write(f"**Input prompt:** {case['input_prompt']}")
                    st.write(f"**Expected behavior:** {case['expected_behavior']}")
                    st.write(f"**Assessment:** {case['qualitative_assessment']}")
                    st.json(case["observed_demo_output"])
                    if case["notes"]:
                        st.caption(case["notes"])

            st.markdown("#### Failure/warning examples")
            issue_rows = [case for case in results if case["status"] != "pass"]
            if not issue_rows:
                st.success("No warning/fail cases in this demo report.")
            else:
                st.dataframe(
                    [
                        {
                            "case_id": case["case_id"],
                            "status": case["status"],
                            "category": case["category"],
                            "assessment": case["qualitative_assessment"],
                        }
                        for case in issue_rows
                    ],
                    hide_index=True,
                    use_container_width=True,
                )

            json_path = report_dir / "demo_results.json"
            csv_path = report_dir / "demo_summary.csv"
            md_path = report_dir / "qualitative_case_gallery.md"
            if json_path.exists():
                st.download_button(
                    "Download JSON report",
                    data=json_path.read_text(encoding="utf-8"),
                    file_name=json_path.name,
                    mime="application/json",
                )
            if csv_path.exists():
                st.download_button(
                    "Download CSV summary",
                    data=csv_path.read_text(encoding="utf-8"),
                    file_name=csv_path.name,
                    mime="text/csv",
                )
            if md_path.exists():
                st.download_button(
                    "Download Markdown gallery",
                    data=md_path.read_text(encoding="utf-8"),
                    file_name=md_path.name,
                    mime="text/markdown",
                )
