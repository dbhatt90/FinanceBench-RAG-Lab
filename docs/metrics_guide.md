# Retrieval Metrics Guide — Which Metric When

All metrics here are **binary-relevance** retrieval metrics: a retrieved chunk is
either relevant (its page is in the gold evidence set) or not. They live in
[`src/rag_hub/eval/retrieval_metrics.py`](../src/rag_hub/eval/retrieval_metrics.py)
and are computed by [`scripts/run_eval.py`](../scripts/run_eval.py).

Notation: `k` = cutoff (we report @5 by default), gold = the set of relevant chunk ids.

| Metric | Measures | Reach for it when… |
|--------|----------|--------------------|
| **Recall@k** | Fraction of all gold chunks found in the top-k | Coverage matters — you must not miss evidence (RAG: the answer needs every relevant page) |
| **Precision@k** | Fraction of the top-k that are relevant | Noise matters — a small context window where junk crowds out signal |
| **Hit@k** | Did *any* relevant chunk appear in top-k (0/1) | A coarse "did it work at all" sanity signal; easy to communicate |
| **MRR** | Reciprocal rank of the *first* relevant chunk | Single-answer lookups where one good hit near the top is enough |
| **MAP@k** | Mean precision across *all* relevant ranks | Multi-evidence questions where ranking *all* relevant chunks well matters |
| **nDCG@k** | Rank-discounted gain, normalised to the ideal ranking | The general-purpose ranking-quality metric; rewards putting relevant chunks higher |
| **ERR@k** | Expected reciprocal rank (cascade model: user stops at first relevant) | Models real user scanning; generalises cleanly to graded relevance later |

## Rules of thumb

- **Recall@k** — *"Did we fetch the evidence at all?"* The most important RAG retrieval metric: the generator cannot cite what was never retrieved.
- **Precision@k** — *"How much of what we fed the LLM was useful?"* Trades off against recall as `k` grows.
- **Hit@k** — *"Yes/no, did we surface something relevant?"* Use for quick dashboards and stakeholder summaries, not for fine ranking comparisons.
- **MRR** — *"How high is the first good hit?"* Best for single-fact questions (most of FinanceBench's "direct" route).
- **MAP@k** — *"Across everything relevant, how good is the ordering?"* Best for comparative / multi-hop questions where several pages must rank well.
- **nDCG@k** — *"Overall ranking quality, position-weighted."* The default when comparing retrievers/rerankers head-to-head.
- **ERR@k** — *"Where does a top-down reader actually stop?"* For binary relevance ERR ≈ MRR; its value is the clean upgrade path to graded relevance.

## In this project

- Days 2–6 optimise these retrieval metrics. The unified harness
  (`scripts/run_eval.py`) computes all seven in one pass per configuration and
  logs them to MLflow; the Streamlit dashboard (`app/eval_dashboard.py`) plots
  their trend across days.
- For reranker / pipeline comparisons, lead with **nDCG@k** and **recall@k**;
  for the routing layer's single-fact "direct" questions, **MRR** is the most
  diagnostic.
- Generation-quality metrics (Exact Match, Numeric Match, ROUGE, BERTScore,
  RAGAS) are separate — see [`scripts/day7_eval.py`](../scripts/day7_eval.py) —
  and will be folded into the unified harness later.
