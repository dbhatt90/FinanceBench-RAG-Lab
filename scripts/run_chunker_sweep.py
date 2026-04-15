"""
run_chunker_sweep.py
--------------------
Day 3: Build one Qdrant collection per chunking strategy, run retrieval eval
on smoke_20, and compare NDCG@10 (+ Recall, MRR) across strategies.

Usage
-----
# Build indexes + eval all strategies (skips already-indexed collections)
python scripts/run_chunker_sweep.py

# Force rebuild everything from scratch
python scripts/run_chunker_sweep.py --force

# Skip indexing, just re-run eval (collections already exist)
python scripts/run_chunker_sweep.py --eval-only

# Run a single strategy
python scripts/run_chunker_sweep.py --strategy recursive
"""

import argparse
import json
import math
import os
from datetime import datetime
from typing import Callable, Dict, List, Set

from rag_hub.indexing.index_builder import IndexBuilder
from rag_hub.embeddings.gemini_001 import GeminiEmbeddingClient
from rag_hub.vectorstore.qdrant_store import QdrantStore
from rag_hub.retrievers.bm25_retriever import BM25Retriever
from rag_hub.retrievers.hybrid_retriever import HybridRRFRetriever
from rag_hub.eval.financebench import load_questions, gold_pages
from rag_hub.eval.retrieval_metrics import recall_at_k, precision_at_k, mrr, map_at_k


# ── Config ───────────────────────────────────────────────────────────────────
EVAL_SET   = "data/eval/smoke_20.jsonl"
OUTPUT_DIR = "eval_results"
K          = 10      # NDCG@10 as primary metric


# ── Strategy registry ────────────────────────────────────────────────────────
# Each entry: (collection_name, chunker_callable)
# Chunker callables must have signature:  pages -> List[chunk_dict]

def _get_strategies() -> Dict[str, Callable]:
    from rag_hub.chunking.recursive      import chunk_pages as recursive
    from rag_hub.chunking.section_aware  import chunk_pages as section_aware
    from rag_hub.chunking.parent_child   import chunk_pages as parent_child
    from rag_hub.chunking.semantic       import chunk_pages as semantic
    from rag_hub.chunking.raptor         import chunk_pages as raptor
    from rag_hub.chunking.dense_x        import chunk_pages as dense_x

    return {
        "recursive":     recursive,
        "section_aware": section_aware,
        "parent_child":  parent_child,
        "semantic":      semantic,
        # "raptor":        raptor,
        # "dense_x":       dense_x,
    }


def collection_name(strategy: str) -> str:
    if strategy == "recursive":
        return "financebench_v1"

    return f"fb_day3_{strategy}"


# ── Metrics helpers ──────────────────────────────────────────────────────────

def make_chunk_id(doc_name: str, page: int) -> str:
    return f"{doc_name}_p{page}"


def dedupe_ranked(ids: list) -> list:
    seen = set()
    return [x for x in ids if not (x in seen or seen.add(x))]


def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Binary NDCG@k.
    DCG  = sum  rel_i / log2(i+1)   for i in 1..k
    IDCG = DCG of perfect ranking (all relevant items at top)
    """
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(i + 1)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


def compute_metrics(retrieved_ids: List[str], relevant_ids: Set[str], k: int = K) -> Dict:
    return {
        f"ndcg@{k}":      round(ndcg_at_k(retrieved_ids, relevant_ids, k), 4),
        f"recall@{k}":    round(recall_at_k(retrieved_ids, relevant_ids, k), 4),
        f"precision@{k}": round(precision_at_k(retrieved_ids, relevant_ids, k), 4),
        "mrr":            round(mrr(retrieved_ids, relevant_ids), 4),
        f"map@{k}":       round(map_at_k(retrieved_ids, relevant_ids, k), 4),
    }


# ── Indexing phase ───────────────────────────────────────────────────────────

def build_indexes(strategies: Dict[str, Callable], force: bool = False):
    for name, chunker in strategies.items():
        coll = collection_name(name)
        print(f"\n{'='*60}")
        print(f"  INDEXING: {name}  →  {coll}")
        print(f"{'='*60}")

        builder = IndexBuilder(
            collection_name=coll,
            eval_set_path=EVAL_SET,
        )
        builder.run(chunker=chunker, force=force)


# ── Eval phase ───────────────────────────────────────────────────────────────

def eval_strategy(
    name: str,
    questions: List[Dict],
    embedder: GeminiEmbeddingClient,
) -> Dict:
    """Run retrieval eval for one strategy. Returns per-question rows + summary."""
    coll  = collection_name(name)
    store = QdrantStore(collection=coll)

    # Build BM25 + hybrid on top of this collection's corpus
    print(f"  Building BM25 index from '{coll}' …")
    bm25   = BM25Retriever()
    bm25.build_from_qdrant(store)
    hybrid = HybridRRFRetriever(store=store, bm25=bm25, rrf_k=60)

    rows = []

    for i, q in enumerate(questions):
        question       = q["question"]
        doc_name       = q["doc_name"]
        gold_pages_set = gold_pages(q)
        doc_name_pdf   = doc_name + ".pdf"
        relevant_ids   = {make_chunk_id(doc_name_pdf, p) for p in gold_pages_set}

        query_vec = embedder.embed_query(question)

        # ── Dense ──
        dense_points   = store.search(query_vec, k=K)
        dense_ids      = dedupe_ranked([
            make_chunk_id(p.payload["doc_name"], p.payload["page"]) for p in dense_points
        ])

        # ── BM25 ──
        bm25_payloads  = bm25.search(question, k=K)
        bm25_ids       = dedupe_ranked([
            make_chunk_id(p["doc_name"], p["page"]) for p in bm25_payloads
        ])

        # ── Hybrid RRF ──
        hybrid_payloads = hybrid.search(question, query_vec, top_k=K)
        hybrid_ids      = dedupe_ranked([
            make_chunk_id(p["doc_name"], p["page"]) for p in hybrid_payloads
        ])

        row = {
            "question_id": q.get("financebench_id", i),
            "question":    question,
            "doc_name":    doc_name,
            "gold_pages":  list(gold_pages_set),
            "dense":       compute_metrics(dense_ids,  relevant_ids, k=K),
            "bm25":        compute_metrics(bm25_ids,   relevant_ids, k=K),
            "hybrid":      compute_metrics(hybrid_ids, relevant_ids, k=K),
        }
        rows.append(row)

        if (i + 1) % 5 == 0:
            print(f"    [{i+1}/{len(questions)}] done")

    # Aggregate all 3 retrievers
    retriever_keys = ["dense", "bm25", "hybrid"]
    metric_keys    = list(rows[0]["dense"].keys())

    summary = {
        ret: {
            mk: round(sum(r[ret][mk] for r in rows) / len(rows), 4)
            for mk in metric_keys
        }
        for ret in retriever_keys
    }

    return {"summary": summary, "rows": rows}


def run_eval(strategies: Dict[str, Callable], questions: List[Dict]) -> Dict:
    embedder = GeminiEmbeddingClient()
    all_results = {}

    for name in strategies:
        print(f"\n{'='*60}")
        print(f"  EVAL: {name}")
        print(f"{'='*60}")
        all_results[name] = eval_strategy(name, questions, embedder)

    return all_results


# ── Reporting ────────────────────────────────────────────────────────────────

def print_comparison_table(all_results: Dict):
    """
    Print one table per retriever (dense / bm25 / hybrid),
    with strategies as columns and metrics as rows.
    """
    strategies    = list(all_results.keys())
    retriever_keys = ["dense", "bm25", "hybrid"]
    metric_keys    = list(all_results[strategies[0]]["summary"]["dense"].keys())
    col_w = 14

    for ret in retriever_keys:
        header = f"\n{'Metric':<16}" + "".join(f"{s:>{col_w}}" for s in strategies)
        print("\n" + "=" * len(header))
        print(f"CHUNKER SWEEP  [{ret.upper()}]  —  strategies as columns")
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        for mk in metric_keys:
            row_str = f"{mk:<16}"
            for s in strategies:
                val = all_results[s]["summary"][ret].get(mk, 0)
                row_str += f"{val:>{col_w}.4f}"
            print(row_str)
        print("=" * len(header))


def save_results(all_results: Dict):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUTPUT_DIR, f"day3_chunker_sweep_{ts}.json")

    output = {
        "timestamp":    datetime.utcnow().isoformat(),
        "eval_set":     EVAL_SET,
        "k":            K,
        "strategies":   list(all_results.keys()),
        "summary":      {s: v["summary"] for s, v in all_results.items()},
        "per_question": {s: v["rows"]    for s, v in all_results.items()},
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n[INFO] Results saved → {path}")
    return path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force",      action="store_true", help="Wipe + rebuild all indexes")
    parser.add_argument("--eval-only",  action="store_true", help="Skip indexing, run eval only")
    parser.add_argument("--strategy",   type=str, default=None,
                        help="Run only one strategy (e.g. --strategy recursive)")
    args = parser.parse_args()

    strategies = _get_strategies()

    if args.strategy:
        if args.strategy not in strategies:
            print(f"[ERROR] Unknown strategy '{args.strategy}'. "
                  f"Choose from: {list(strategies.keys())}")
            return
        strategies = {args.strategy: strategies[args.strategy]}

    questions = load_questions(EVAL_SET)
    print(f"[INFO] Eval set: {len(questions)} questions, "
          f"{len(set(q['doc_name'] for q in questions))} unique docs")

    if not args.eval_only:
        build_indexes(strategies, force=args.force)

    all_results = run_eval(strategies, questions)

    print_comparison_table(all_results)
    save_results(all_results)


if __name__ == "__main__":
    main()
