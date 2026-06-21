"""
Day 8 — Retrieval metrics dashboard.

Reads MLflow runs logged by scripts/run_eval.py and plots how each retrieval
metric trends across the days of the series. Falls back to the canonical
eval_results/day*/retrieval_eval_*.json files when the MLflow store is empty.

Run:
    streamlit run app/eval_dashboard.py
"""

import glob
import json
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag_hub.config.settings import RESULTS_DIR
from rag_hub.eval.tracking import TRACKING_URI, EXPERIMENT_NAME

METRIC_COLS = [
    "recall_at_5", "precision_at_5", "mrr", "map_at_5",
    "hit_at_5", "err_at_5", "ndcg_at_5",
]


@st.cache_data(ttl=30)
def load_from_mlflow() -> pd.DataFrame:
    """Return a tidy DataFrame [day, config, <metrics...>] from MLflow, or empty."""
    try:
        import mlflow
        mlflow.set_tracking_uri(TRACKING_URI)
        runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])
    except Exception:
        return pd.DataFrame()
    if runs is None or len(runs) == 0:
        return pd.DataFrame()

    rows = []
    for _, r in runs.iterrows():
        row = {"day": _to_int(r.get("tags.day")), "config": r.get("tags.config")}
        for m in METRIC_COLS:
            row[m] = r.get(f"metrics.{m}")
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.dropna(subset=["day"]).sort_values(["day", "config"])


@st.cache_data(ttl=30)
def load_from_files() -> pd.DataFrame:
    """Fallback: read canonical per-config JSON result files."""
    rows = []
    for path in glob.glob(str(RESULTS_DIR / "day*" / "retrieval_eval_*.json")):
        with open(path) as f:
            res = json.load(f)
        row = {"day": res.get("day"), "config": res.get("config")}
        for k, v in res.get("overall", {}).items():
            row[k.replace("@", "_at_")] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values(["day", "config"]) if len(df) else df


def _to_int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


st.set_page_config(page_title="RAG Retrieval Metrics", layout="wide")
st.title("📊 FinanceBench — Retrieval Metrics Across Days")

df = load_from_mlflow()
source = "MLflow"
if df.empty:
    df = load_from_files()
    source = "result files (MLflow store empty)"

if df.empty:
    st.warning("No eval runs found. Run `python scripts/run_eval.py` first.")
    st.stop()

st.caption(f"Source: {source} · {len(df)} runs")

available = [m for m in METRIC_COLS if m in df.columns and df[m].notna().any()]
metric = st.selectbox(
    "Metric", available,
    format_func=lambda m: m.replace("_at_", "@"),
)

# Trend across days: one line per config, x = day.
st.subheader(f"{metric.replace('_at_', '@')} across days")
chart_df = df.pivot_table(index="day", columns="config", values=metric, aggfunc="mean")
st.line_chart(chart_df)

# Per-config comparison table (all metrics).
st.subheader("All configurations")
show = df[["day", "config"] + available].copy()
show.columns = ["day", "config"] + [m.replace("_at_", "@") for m in available]
st.dataframe(
    show.style.format({c: "{:.4f}" for c in show.columns if c not in ("day", "config")}),
    use_container_width=True,
)
