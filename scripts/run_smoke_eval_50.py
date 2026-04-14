import os
import json
from datetime import datetime

from rag_hub.eval.financebench import load_questions, gold_pages
from rag_hub.embeddings.gemini_001 import GeminiEmbeddingClient
from rag_hub.eval.retrieval_metrics import (
    recall_at_k,
    precision_at_k,
    mrr,
    map_at_k,
    hit_rate_at_k,
)
from rag_hub.vectorstore.qdrant_store import QdrantStore
from rag_hub.retrievers.bm25_retriever import BM25Retriever
from rag_hub.retrievers.hybrid_retriever import HybridRRFRetriever


SMOKE_PATH = "data/eval/smoke_50.jsonl"
OUTPUT_PATH = "eval_results/day2_smoke_50.json"
K = 5


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_chunk_id(doc_name: str, page: int) -> str:
    return f"{doc_name}_p{page}"


def dedupe_ranked(ids: list) -> list:
    """
    Remove duplicate page-level IDs while preserving rank order.
    Needed because multiple chunks from the same page produce the same ID,
    which would inflate Recall and MAP by counting the same page multiple times.
    """
    seen = set()
    return [x for x in ids if not (x in seen or seen.add(x))]


def compute_metrics(retrieved_ids, relevant_ids, k=K):
    return {
        f"recall@{k}":    recall_at_k(retrieved_ids, relevant_ids, k=k),
        f"precision@{k}": precision_at_k(retrieved_ids, relevant_ids, k=k),
        "mrr":            mrr(retrieved_ids, relevant_ids),
        f"map@{k}":       map_at_k(retrieved_ids, relevant_ids, k=k),
        f"hit@{k}":       int(hit_rate_at_k(retrieved_ids, relevant_ids, k=k)),
    }


def mean_metrics(rows, key):
    return sum(r[key] for r in rows) / len(rows)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    questions = load_questions(SMOKE_PATH)
    print(f"[INFO] Loaded {len(questions)} questions")

    # -----------------------------
    # Init retrievers
    # -----------------------------
    embedder = GeminiEmbeddingClient()
    store    = QdrantStore(collection="financebench_v1")

    print("[INFO] Building BM25 index from Qdrant corpus …")
    bm25 = BM25Retriever()
    bm25.build_from_qdrant(store)

    hybrid = HybridRRFRetriever(store=store, bm25=bm25, rrf_k=60)

    results = []

    # -----------------------------
    # Per-question eval loop
    # -----------------------------
    for i, q in enumerate(questions):
        question        = q["question"]
        doc_name        = q["doc_name"]
        gold_pages_set  = gold_pages(q)
        doc_name_pdf = doc_name + ".pdf"
        relevant_ids    = {make_chunk_id(doc_name_pdf, p) for p in gold_pages_set}

        print("\n" + "=" * 80)
        print(f"[{i+1}/{len(questions)}] {question}")
        print(f"Target doc : {doc_name_pdf}")
        print(f"Gold pages : {sorted(gold_pages_set)}")

        # ---- Dense ----
        query_vec = embedder.embed_query(question)
        dense_points = store.search(query_vec, k=K)
        dense_payloads = [p.payload for p in dense_points]
        dense_ids = dedupe_ranked([make_chunk_id(p["doc_name"], p["page"]) for p in dense_payloads])

        # ---- BM25 ----
        bm25_payloads = bm25.search(question, k=K)
        bm25_ids = dedupe_ranked([make_chunk_id(p["doc_name"], p["page"]) for p in bm25_payloads])

        # ---- Hybrid ----
        hybrid_payloads = hybrid.search(question, query_vec, top_k=K)
        hybrid_ids = dedupe_ranked([make_chunk_id(p["doc_name"], p["page"]) for p in hybrid_payloads])

        # ---- Metrics ----
        dense_m  = compute_metrics(dense_ids,  relevant_ids)
        bm25_m   = compute_metrics(bm25_ids,   relevant_ids)
        hybrid_m = compute_metrics(hybrid_ids, relevant_ids)

        print(f"\n{'Metric':<14} {'Dense':>8} {'BM25':>8} {'Hybrid':>8}")
        print("-" * 42)
        for metric in dense_m:
            print(f"{metric:<14} {dense_m[metric]:>8.3f} {bm25_m[metric]:>8.3f} {hybrid_m[metric]:>8.3f}")

        row = {
            "question_id": q.get("id", i),
            "question":    question,
            "doc_name":    doc_name,
            "gold_pages":  list(gold_pages_set),
            "dense":       dense_m,
            "bm25":        bm25_m,
            "hybrid":      hybrid_m,
        }
        results.append(row)

    # -----------------------------
    # Aggregate across all questions
    # -----------------------------
    metric_keys = list(results[0]["dense"].keys())

    def agg(retriever_key):
        return {mk: round(mean_metrics(
            [r[retriever_key] for r in results], mk
        ), 4) for mk in metric_keys}

    summary = {
        "dense":  agg("dense"),
        "bm25":   agg("bm25"),
        "hybrid": agg("hybrid"),
    }

    print("\n" + "=" * 80)
    print("RETRIEVAL EVAL SUMMARY  (50 questions)")
    print("=" * 80)
    print(f"\n{'Metric':<14} {'Dense':>8} {'BM25':>8} {'Hybrid':>8}")
    print("-" * 42)
    for mk in metric_keys:
        print(
            f"{mk:<14} "
            f"{summary['dense'][mk]:>8.3f} "
            f"{summary['bm25'][mk]:>8.3f} "
            f"{summary['hybrid'][mk]:>8.3f}"
        )

    # -----------------------------
    # Save
    # -----------------------------
    os.makedirs("eval_results", exist_ok=True)

    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "n_questions": len(questions),
        "k": K,
        "summary": summary,
        "results": results,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n[INFO] Saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
