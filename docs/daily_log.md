#### Day 0 definition of done

- [ ] `.env` has 2–4 valid API keys
- [ ] `uv sync` runs clean
- [ ] `http://localhost:6333/dashboard` loads (Qdrant is up)
- [ ] `data/raw/financebench/pdfs/` contains ~90 PDFs
- [ ] `datasets.load_dataset("PatronusAI/financebench")` returns ~150 rows in a Python REPL
- [ ] GitHub repo exists, 2 commits pushed, README shows "Day 0/14"
- [ ] You can explain in one sentence why FinanceBench is the right eval set (answer: because its ground truth lives in the same PDFs you just indexed)

**LinkedIn Day 0 post (optional, builds anticipation):** *"Starting a 14-day RAG build today. The goal: production-grade Retrieval-Augmented Generation over SEC 10-K filings, benchmarked on FinanceBench, deployed live by Day 14. Stack: LangGraph, Qdrant, Gemini, Groq. Follow along. #BuildInPublic"*

---

#### Day 4 — Query Translation Layer

**Theme:** "Rewrite the question before you retrieve."

**Techniques implemented:**
- `HyDETransform` — generates a hypothetical 10-K/Q passage, embeds it as `RETRIEVAL_DOCUMENT`
- `MultiQueryTransform` — expands query into N rephrased variants using domain terminology variation
- `RAGFusionRetriever` — multi-query + per-variant dense retrieval + RRF fusion
- `DecompositionTransform` — splits multi-hop questions into atomic sub-questions, RRF-fuses results
- `StepBackTransform` — stub (passthrough); deferred to a future day
- All implement a shared `QueryTransform` ABC (`transform(query) -> List[str]`)

**New metric added:** `ERR@k` (Expected Reciprocal Rank) in `retrieval_metrics.py`. For binary relevance it equals MRR; designed to generalise to graded relevance in later days.

**Eval setup:** `financebench_v1` Qdrant collection (recursive chunks), `smoke_50.jsonl`, K=5.
Each technique writes to `eval_results/day4/<technique>.json`; `compare.py` produces the comparison table.

**Results (hit@5 / recall@5):**

| technique     | hit@5  | recall@5 |
|---------------|--------|----------|
| baseline      | 0.8200 | 0.7800   |
| decomposition | 0.7000 | 0.6100   |
| rag_fusion    | 0.6200 | 0.5700   |
| hyde          | 0.5400 | 0.5100   |

**By question type (hit@5):**

| question_type     | baseline | hyde   | rag_fusion | decomposition | n  |
|-------------------|----------|--------|------------|---------------|----|
| domain-relevant   | 0.7059   | 0.4118 | 0.5294     | 0.5882        | 17 |
| metrics-generated | 1.0000   | 0.5294 | 0.7059     | 0.8235        | 17 |
| novel-generated   | 0.7500   | 0.6875 | 0.6250     | 0.6875        | 16 |

**Key finding — query translation hurt, not helped:**

Baseline outperformed all three techniques across every question type. The root cause is the embedding model choice.

`gemini-embedding-001` uses separate task types (`RETRIEVAL_QUERY` vs `RETRIEVAL_DOCUMENT`) and is trained as an **asymmetric retrieval model**: `RETRIEVAL_QUERY(question)` is already optimised to align with `RETRIEVAL_DOCUMENT(chunk)` across the semantic and vocabulary gap. Query translation techniques were designed to solve exactly this gap — but it doesn't exist here.

Specific failure modes:
- **HyDE** is the biggest loser (recall drops 0.78 → 0.51). It embeds the hypothetical passage as `RETRIEVAL_DOCUMENT`, bypassing the `RETRIEVAL_QUERY` path that is specifically optimised for question-to-document alignment. It adds LLM generation noise on top of an already well-calibrated model. HyDE would likely win on symmetric models like `text-embedding-ada-002` or `all-MiniLM`.
- **RAG-Fusion** hurts because FinanceBench questions are already maximally precise. Rephrased variants introduce semantic drift rather than vocabulary coverage. RRF then fuses 3 noisy signals instead of 1 clean one.
- **Decomposition** comes closest to baseline on `novel-generated` questions (gap: 0.75 → 0.69). The majority of questions are still single-hop in terms of evidence location even if they look complex, so decomposition spreads retrieval budget without concentrating on the gold page.

**Transferable lesson:** Query translation is not universally beneficial. Its value is model-dependent. For asymmetric embedding models with task-type support, invest retrieval budget elsewhere (better chunking, reranking, contextual compression). For symmetric models, HyDE and RAG-Fusion can meaningfully improve recall.