"""
run_generation_eval.py
----------------------
Day 3: End-to-end generation quality eval across 4 chunking strategies.

For each strategy:
    1. Retrieve top-K chunks (hybrid RRF — best retriever from sweep)
    2. For parent-child: swap child text → parent_text in payload
    3. Feed retrieved context to GeminiFlashGenerator
    4. Score prediction against gold answer: Exact Match + Token F1

Strategies evaluated: recursive, section_aware, semantic, parent_child
Eval set: smoke_20 (20 questions)

Usage
-----
python scripts/run_generation_eval.py

# Single strategy
python scripts/run_generation_eval.py --strategy recursive

# Use dense-only retrieval instead of hybrid
python scripts/run_generation_eval.py --retriever dense
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List

from rag_hub.eval.financebench import load_questions
from rag_hub.eval.metrics import exact_match, token_f1
from rag_hub.embeddings.gemini_001 import GeminiEmbeddingClient
from rag_hub.generation.gemini_LLM import GeminiFlashGenerator
from rag_hub.retrievers.bm25_retriever import BM25Retriever
from rag_hub.retrievers.hybrid_retriever import HybridRRFRetriever
from rag_hub.vectorstore.qdrant_store import QdrantStore


# ── Config ────────────────────────────────────────────────────────────────────
EVAL_SET   = "data/eval/smoke_20.jsonl"
OUTPUT_DIR = "eval_results"
K          = 5    # chunks fed to LLM — keep low to control token cost

STRATEGIES = ["recursive", "section_aware", "semantic", "parent_child"]


def collection_name(strategy: str) -> str:
    if strategy == "recursive":
        return "financebench_v1"
    return f"fb_day3_{strategy}"


# ── Parent-child context swap ─────────────────────────────────────────────────

def prepare_context_chunks(payloads: List[Dict], strategy: str) -> List[Dict]:
    """
    For parent_child: replace the small child text with the full parent_text
    so the LLM receives richer context. All other strategies pass through as-is.

    The generator's _format_context() reads chunk["text"], so we overwrite it.
    We keep the original child text under chunk["child_text"] for inspection.
    """
    if strategy != "parent_child":
        return payloads

    enriched = []
    for p in payloads:
        chunk = dict(p)  # don't mutate original
        if "parent_text" in chunk and chunk["parent_text"]:
            chunk["child_text"] = chunk["text"]   # preserve for logging
            chunk["text"]       = chunk["parent_text"]
        enriched.append(chunk)
    return enriched


# ── Per-strategy eval ─────────────────────────────────────────────────────────

def eval_strategy(
    strategy: str,
    questions: List[Dict],
    embedder: GeminiEmbeddingClient,
    generator: GeminiFlashGenerator,
    retriever_mode: str = "hybrid",
) -> Dict:
    coll  = collection_name(strategy)
    store = QdrantStore(collection=coll)

    # Build BM25 + hybrid on this collection's corpus
    print(f"  Building BM25 index from '{coll}' …")
    bm25   = BM25Retriever()
    bm25.build_from_qdrant(store)
    hybrid = HybridRRFRetriever(store=store, bm25=bm25, rrf_k=60)

    rows = []

    for i, q in enumerate(questions):
        question    = q["question"]
        gold_answer = q.get("answer", "")

        print(f"\n  [{i+1}/{len(questions)}] {question[:80]}")

        # ── Retrieve ──────────────────────────────────────────────────────────
        query_vec = embedder.embed_query(question)

        if retriever_mode == "dense":
            points   = store.search(query_vec, k=K)
            payloads = [p.payload for p in points]
        elif retriever_mode == "bm25":
            payloads = bm25.search(question, k=K)
        else:  # hybrid (default)
            payloads = hybrid.search(question, query_vec, top_k=K)

        # ── Context prep (parent-child swap) ──────────────────────────────────
        context_chunks = prepare_context_chunks(payloads, strategy)

        # ── Generate ──────────────────────────────────────────────────────────
        try:
            prediction = generator.generate(question=question, chunks=context_chunks)
        except Exception as e:
            print(f"  [WARN] Generation failed: {e}")
            prediction = ""

        # ── Score ─────────────────────────────────────────────────────────────
        em = int(exact_match(prediction, gold_answer))
        f1 = token_f1(prediction, gold_answer)

        print(f"    Gold  : {gold_answer}")
        print(f"    Pred  : {prediction}")
        print(f"    EM={em}  F1={f1:.3f}")

        rows.append({
            "question_id": q.get("financebench_id", i),
            "question":    question,
            "doc_name":    q.get("doc_name", ""),
            "gold_answer": gold_answer,
            "prediction":  prediction,
            "exact_match": em,
            "token_f1":    round(f1, 4),
            "n_chunks":    len(context_chunks),
        })

    # ── Aggregate ─────────────────────────────────────────────────────────────
    n = len(rows)
    summary = {
        "exact_match": round(sum(r["exact_match"] for r in rows) / n, 4),
        "token_f1":    round(sum(r["token_f1"]    for r in rows) / n, 4),
        "n_questions": n,
    }

    print(f"\n  [{strategy}]  EM={summary['exact_match']:.3f}  F1={summary['token_f1']:.3f}")

    return {"summary": summary, "rows": rows}


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_comparison_table(all_results: Dict):
    strategies  = list(all_results.keys())
    metric_keys = ["exact_match", "token_f1"]
    col_w = 16

    header = f"\n{'Metric':<16}" + "".join(f"{s:>{col_w}}" for s in strategies)
    print("\n" + "=" * len(header))
    print("GENERATION EVAL  —  EM + Token F1  (smoke_20)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for mk in metric_keys:
        row = f"{mk:<16}"
        for s in strategies:
            val = all_results[s]["summary"].get(mk, 0)
            row += f"{val:>{col_w}.4f}"
        print(row)
    print("=" * len(header))


def save_results(all_results: Dict, retriever_mode: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUTPUT_DIR, f"day3_generation_eval_{ts}.json")

    output = {
        "timestamp":     datetime.utcnow().isoformat(),
        "eval_set":      EVAL_SET,
        "k":             K,
        "retriever":     retriever_mode,
        "strategies":    list(all_results.keys()),
        "summary":       {s: v["summary"] for s, v in all_results.items()},
        "per_question":  {s: v["rows"]    for s, v in all_results.items()},
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n[INFO] Results saved → {path}")
    return path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy",  type=str, default=None,
                        help=f"Single strategy to run. Options: {STRATEGIES}")
    parser.add_argument("--retriever", type=str, default="hybrid",
                        choices=["dense", "bm25", "hybrid"],
                        help="Retriever to use for context (default: hybrid)")
    args = parser.parse_args()

    strategies = (
        {args.strategy: None for args in [args]} if args.strategy
        else {s: None for s in STRATEGIES}
    )
    if args.strategy:
        if args.strategy not in STRATEGIES:
            print(f"[ERROR] Unknown strategy '{args.strategy}'. Choose from: {STRATEGIES}")
            return
        strategies = [args.strategy]
    else:
        strategies = STRATEGIES

    questions = load_questions(EVAL_SET)
    print(f"[INFO] Eval set: {len(questions)} questions")
    print(f"[INFO] Retriever: {args.retriever}  |  K={K}")
    print(f"[INFO] Strategies: {strategies}")

    # Init shared clients (one instance, reused across strategies)
    embedder  = GeminiEmbeddingClient()
    generator = GeminiFlashGenerator()

    all_results = {}

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"  STRATEGY: {strategy}")
        print(f"{'='*60}")
        all_results[strategy] = eval_strategy(
            strategy=strategy,
            questions=questions,
            embedder=embedder,
            generator=generator,
            retriever_mode=args.retriever,
        )

    print_comparison_table(all_results)
    save_results(all_results, args.retriever)


if __name__ == "__main__":
    main()
