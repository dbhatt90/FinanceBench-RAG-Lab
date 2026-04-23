"""
Day 4 RAG-Fusion eval: multi-query expansion + per-query dense retrieval + RRF.
Results saved to eval_results/day4/rag_fusion.json.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from _eval_helpers import (
    SMOKE_PATH, K,
    make_chunk_id, dedupe_ranked, compute_metrics, aggregate,
    save_results, print_summary,
)
from rag_hub.eval.financebench import load_questions, gold_pages
from rag_hub.embeddings.gemini_001 import GeminiEmbeddingClient
from rag_hub.vectorstore.qdrant_store import QdrantStore
from rag_hub.query.rag_fusion import RAGFusionRetriever


def main():
    questions = load_questions(SMOKE_PATH)
    print(f"[INFO] Loaded {len(questions)} questions")

    embedder   = GeminiEmbeddingClient()
    store      = QdrantStore(collection="financebench_v1")
    rag_fusion = RAGFusionRetriever(store=store, embedder=embedder, n_queries=3)

    results = []

    for i, q in enumerate(questions):
        question       = q["question"]
        doc_name       = q["doc_name"]
        gold_pages_set = gold_pages(q)
        doc_name_pdf   = doc_name + ".pdf"
        relevant_ids   = {make_chunk_id(doc_name_pdf, p) for p in gold_pages_set}

        print(f"\n[{i+1}/{len(questions)}] {question[:80]}")

        fused_payloads, expanded_queries = rag_fusion.search(question, top_k=K)

        print(f"  Expanded queries ({len(expanded_queries)}):")
        for eq in expanded_queries:
            print(f"    • {eq}")

        retrieved_ids = dedupe_ranked([make_chunk_id(p["doc_name"], p["page"]) for p in fused_payloads])

        metrics = compute_metrics(retrieved_ids, relevant_ids)
        print("  " + "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

        results.append({
            "question_id":      q.get("financebench_id", i),
            "question":         question,
            "doc_name":         doc_name,
            "gold_pages":       list(gold_pages_set),
            "expanded_queries": expanded_queries,
            "rag_fusion":       metrics,
        })

    summary = aggregate(results, "rag_fusion")
    print_summary("rag_fusion", summary)
    save_results("rag_fusion", results, summary)


if __name__ == "__main__":
    main()
