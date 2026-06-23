"""
FinanceBench RAG — Retrieval Quality Dashboard

Shows how retrieval improved across the 14-day build, reading directly from
the eval_results/ JSON files produced by each day's evaluation script.

Run:
    streamlit run app/eval_dashboard.py
"""

import glob
import json
import os
import sys

import pandas as pd
import streamlit as st

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(ROOT, "eval_results")

st.set_page_config(page_title="RAG Retrieval Dashboard", layout="wide")
st.title("FinanceBench RAG — Retrieval Quality Across Days")

METRIC_LABELS = {
    "recall@5": "Recall@5",
    "recall@10": "Recall@10",
    "precision@5": "Precision@5",
    "mrr": "MRR",
    "map@5": "MAP@5",
    "map@10": "MAP@10",
    "hit@5": "Hit@5",
    "err@5": "ERR@5",
    "ndcg@5": "nDCG@5",
    "ndcg@10": "nDCG@10",
}


def _load(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Metric selector (sidebar)
# ---------------------------------------------------------------------------
st.sidebar.header("Settings")
all_metrics = list(METRIC_LABELS.keys())
metric = st.sidebar.selectbox(
    "Primary metric",
    all_metrics,
    format_func=lambda m: METRIC_LABELS[m],
    index=all_metrics.index("recall@5"),
)

# ---------------------------------------------------------------------------
# Summary trend — best config per day
# ---------------------------------------------------------------------------
st.header("Overall Trend — Best Config Per Day")

trend_rows = []

# Day 2: hybrid is the best config
try:
    d2 = _load(os.path.join(RESULTS, "day2/chunking_sweep.json"))
    s = d2["summary"]["hybrid"]
    trend_rows.append({"Day": 2, "Config": "Hybrid RRF", **s})
except Exception:
    pass

# Day 3: best chunker (recursive had highest nDCG for dense)
try:
    files = sorted(glob.glob(os.path.join(RESULTS, "day3/day3_chunker_sweep_*.json")))
    d3 = _load(files[-1])
    best_strat, best_val = "recursive", -1
    for strat, methods in d3["summary"].items():
        v = methods.get("dense", {}).get("ndcg@10", 0)
        if v > best_val:
            best_val, best_strat = v, strat
    s = d3["summary"][best_strat]["dense"]
    trend_rows.append({"Day": 3, "Config": f"Chunking:{best_strat}+dense", **s})
except Exception:
    pass

# Day 4: baseline (dense) was best
try:
    d4 = _load(os.path.join(RESULTS, "day4/baseline.json"))
    trend_rows.append({"Day": 4, "Config": "Dense baseline", **d4["summary"]})
except Exception:
    pass

# Day 5: LLM routing overall
try:
    d5 = _load(os.path.join(RESULTS, "day5/routing_eval.json"))
    s = d5["retrieval_metrics"]["overall"]
    trend_rows.append({"Day": 5, "Config": "LLM Routing", **s})
except Exception:
    pass

# Day 6: BGE+CRAG (best phase)
try:
    d6c = _load(os.path.join(RESULTS, "day6/comparison.json"))
    s = d6c["phases"]["+BGE+CRAG"]["overall"]
    trend_rows.append({"Day": 6, "Config": "BGE+CRAG", **s})
except Exception:
    pass

if trend_rows:
    trend_df = pd.DataFrame(trend_rows).set_index("Day")
    cols = [c for c in trend_df.columns if c != "Config"]
    available = [m for m in all_metrics if m in cols]
    if metric in available:
        chart_col = metric
    else:
        chart_col = available[0] if available else None

    if chart_col:
        st.line_chart(trend_df[[chart_col]].rename(columns={chart_col: METRIC_LABELS[chart_col]}))
    st.dataframe(
        trend_df[["Config"] + [m for m in all_metrics if m in cols]]
        .rename(columns=METRIC_LABELS)
        .style.format({METRIC_LABELS[m]: "{:.3f}" for m in all_metrics if m in cols}),
        use_container_width=True,
    )
else:
    st.warning("No trend data found in eval_results/.")

st.divider()

# ---------------------------------------------------------------------------
# Day 2 — Retrieval Method Comparison
# ---------------------------------------------------------------------------
st.header("Day 2 — Retrieval Method: Dense vs BM25 vs Hybrid RRF")
try:
    d2 = _load(os.path.join(RESULTS, "day2/chunking_sweep.json"))
    rows = []
    for method, metrics in d2["summary"].items():
        rows.append({"Method": method.upper(), **metrics})
    df2 = pd.DataFrame(rows).set_index("Method")
    cols2 = [m for m in all_metrics if m in df2.columns]

    col_a, col_b = st.columns([3, 2])
    with col_a:
        if metric in cols2:
            st.bar_chart(df2[[metric]].rename(columns={metric: METRIC_LABELS[metric]}))
    with col_b:
        st.dataframe(
            df2[cols2].rename(columns=METRIC_LABELS)
            .style.format({METRIC_LABELS[m]: "{:.3f}" for m in cols2}),
            use_container_width=True,
        )
    st.caption("50 questions · k=5 · recursive chunking")
except Exception as e:
    st.warning(f"Day 2 data unavailable: {e}")

st.divider()

# ---------------------------------------------------------------------------
# Day 3 — Chunking Strategy Comparison
# ---------------------------------------------------------------------------
st.header("Day 3 — Chunking Strategy: Recursive vs Section-aware vs Parent-child vs Semantic")
try:
    files = sorted(glob.glob(os.path.join(RESULTS, "day3/day3_chunker_sweep_*.json")))
    d3 = _load(files[-1])
    strat_names = d3["strategies"]

    retrieval_method = st.radio(
        "Retrieval method", ["dense", "bm25", "hybrid"], horizontal=True, key="d3_method"
    )

    rows3 = []
    for strat in strat_names:
        s = d3["summary"].get(strat, {}).get(retrieval_method, {})
        rows3.append({"Strategy": strat, **s})
    df3 = pd.DataFrame(rows3).set_index("Strategy")
    cols3 = [m for m in all_metrics if m in df3.columns]

    col_a, col_b = st.columns([3, 2])
    with col_a:
        if metric in cols3:
            st.bar_chart(df3[[metric]].rename(columns={metric: METRIC_LABELS[metric]}))
        elif cols3:
            st.bar_chart(df3[[cols3[0]]].rename(columns={cols3[0]: METRIC_LABELS[cols3[0]]}))
    with col_b:
        st.dataframe(
            df3[cols3].rename(columns=METRIC_LABELS)
            .style.format({METRIC_LABELS[m]: "{:.3f}" for m in cols3}),
            use_container_width=True,
        )
    st.caption(f"20 questions · k=10 · retrieval={retrieval_method}")
except Exception as e:
    st.warning(f"Day 3 data unavailable: {e}")

st.divider()

# ---------------------------------------------------------------------------
# Day 4 — Query Transformation
# ---------------------------------------------------------------------------
st.header("Day 4 — Query Transformation: Baseline vs HyDE vs Decomposition vs RAG Fusion")
try:
    techniques = ["baseline", "hyde", "decomposition", "rag_fusion"]
    labels = {"baseline": "Baseline (dense)", "hyde": "HyDE", "decomposition": "Decomposition", "rag_fusion": "RAG Fusion"}
    rows4 = []
    for t in techniques:
        path = os.path.join(RESULTS, f"day4/{t}.json")
        if os.path.exists(path):
            d = _load(path)
            rows4.append({"Technique": labels[t], **d["summary"]})
    df4 = pd.DataFrame(rows4).set_index("Technique")
    cols4 = [m for m in all_metrics if m in df4.columns]

    col_a, col_b = st.columns([3, 2])
    with col_a:
        if metric in cols4:
            st.bar_chart(df4[[metric]].rename(columns={metric: METRIC_LABELS[metric]}))
    with col_b:
        st.dataframe(
            df4[cols4].rename(columns=METRIC_LABELS)
            .style.format({METRIC_LABELS[m]: "{:.3f}" for m in cols4}),
            use_container_width=True,
        )
    st.caption("50 questions · k=5 · dense retrieval")
except Exception as e:
    st.warning(f"Day 4 data unavailable: {e}")

st.divider()

# ---------------------------------------------------------------------------
# Day 5 — LLM Routing
# ---------------------------------------------------------------------------
st.header("Day 5 — LLM Routing: Overall + Per-Route Breakdown")
try:
    d5 = _load(os.path.join(RESULTS, "day5/routing_eval.json"))
    rm = d5["retrieval_metrics"]
    overall5 = rm["overall"]
    per_route = rm.get("per_route", {})

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Overall")
        st.dataframe(
            pd.DataFrame([overall5]).rename(columns=METRIC_LABELS)
            .style.format("{:.3f}"),
            use_container_width=True,
        )

    with col_b:
        st.subheader("Per Route")
        route_rows = []
        for route, s in per_route.items():
            n = s.pop("n", "?")
            route_rows.append({"Route": f"{route} (n={n})", **s})
        df5r = pd.DataFrame(route_rows).set_index("Route")
        cols5 = [m for m in all_metrics if m in df5r.columns]
        st.dataframe(
            df5r[cols5].rename(columns=METRIC_LABELS)
            .style.format({METRIC_LABELS[m]: "{:.3f}" for m in cols5}),
            use_container_width=True,
        )

    if metric in [m for m in all_metrics if m in df5r.columns]:
        st.bar_chart(df5r[[metric]].rename(columns={metric: METRIC_LABELS[metric]}))

    routing_acc = d5.get("routing_accuracy")
    if routing_acc is not None:
        st.caption(f"Routing accuracy: {routing_acc:.1%} · 50 questions · k=5")
except Exception as e:
    st.warning(f"Day 5 data unavailable: {e}")

st.divider()

# ---------------------------------------------------------------------------
# Day 6 — Reranking + CRAG
# ---------------------------------------------------------------------------
st.header("Day 6 — Reranking + CRAG: Baseline → +BGE → +ColBERT → +BGE+CRAG")
try:
    d6 = _load(os.path.join(RESULTS, "day6/comparison.json"))
    rows6 = []
    phase_labels = {
        "baseline": "Baseline (no rerank)",
        "+BGE": "+BGE reranker",
        "+ColBERT": "+ColBERT reranker",
        "+BGE+CRAG": "+BGE + CRAG",
    }
    for phase, data in d6["phases"].items():
        label = phase_labels.get(phase, phase)
        rows6.append({"Phase": label, **data["overall"]})
    df6 = pd.DataFrame(rows6).set_index("Phase")
    cols6 = [m for m in all_metrics if m in df6.columns]

    col_a, col_b = st.columns([3, 2])
    with col_a:
        if metric in cols6:
            st.bar_chart(df6[[metric]].rename(columns={metric: METRIC_LABELS[metric]}))
    with col_b:
        st.dataframe(
            df6[cols6].rename(columns=METRIC_LABELS)
            .style.format({METRIC_LABELS[m]: "{:.3f}" for m in cols6}),
            use_container_width=True,
        )

    # CRAG fallback stats — read from retrieval_eval_bge_crag.json (most current)
    try:
        d6crag = _load(os.path.join(RESULTS, "day6/retrieval_eval_bge_crag.json"))
        pq = d6crag.get("per_question", [])
        total = len([r for r in pq if "metrics" in r])
        fallbacks = sum(1 for r in pq if r.get("used_fallback"))
        st.caption(f"CRAG web fallback triggered on {fallbacks}/{total} questions "
                   f"({fallbacks/total:.0%}) · k=5")
    except Exception:
        pass
except Exception as e:
    st.warning(f"Day 6 data unavailable: {e}")
