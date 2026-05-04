from __future__ import annotations

import json

import streamlit as st

from src.config import METRICS_PATH, PIPELINE_PATH
from src.data_loader import load_dataset, qrels_by_query
from src.evaluate_ranker import evaluate
from src.model import load_pickle
from src.train_ranker import train


st.set_page_config(page_title="Search Ranking LTR Demo", layout="wide")


@st.cache_resource
def load_pipeline():
    if not PIPELINE_PATH.exists():
        train()
    return load_pickle(PIPELINE_PATH)


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        evaluate()
    with METRICS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_query_data():
    queries, _documents, qrels = load_dataset()
    return queries, qrels


def metric_table(metrics: dict[str, object]) -> list[dict[str, object]]:
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


def render_results(title: str, rows: list[dict[str, object]], relevance: dict[str, int]) -> None:
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


pipeline = load_pipeline()
metrics = load_metrics()
queries, qrels = load_query_data()
judgments = qrels_by_query(qrels)
examples = [query.text for query in queries if query.split == "test"][:3]
query_lookup = {query.text: query for query in queries}

st.title("Search Result Ranking Demo")
st.caption(
    "Portfolio/demo learning-to-rank project comparing BM25 retrieval against a supervised ranker."
)

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
metric_rows = metric_table(metrics)
st.dataframe(metric_rows, hide_index=True, use_container_width=True)

chart_rows = []
for row in metric_rows:
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
    render_results("BM25 Baseline", baseline_rows, relevance)
with right:
    render_results("Learned Ranker", learned_rows, relevance)

st.caption(
    f"Ranker backend: {metrics['ranker_backend']}. Dataset: synthetic/demo data, not a production benchmark."
)
