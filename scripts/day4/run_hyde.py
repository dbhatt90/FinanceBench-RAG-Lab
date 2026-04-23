"""
Day 4 HyDE eval: embed the hypothetical passage (RETRIEVAL_DOCUMENT task type)
instead of the raw question.
Results saved to eval_results/day4/hyde.json.
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
from rag_hub.query.hyde import HyDETransform


def main():
    questions = load_questions(SMOKE_PATH)
    print(f"[INFO] Loaded {len(questions)} questions")

    embedder = GeminiEmbeddingClient()
    store    = QdrantStore(collection="financebench_v1")
    hyde     = HyDETransform(verbose=True)

    results = []

    for i, q in enumerate(questions):
        question       = q["question"]
        doc_name       = q["doc_name"]
        gold_pages_set = gold_pages(q)
        doc_name_pdf   = doc_name + ".pdf"
        relevant_ids   = {make_chunk_id(doc_name_pdf, p) for p in gold_pages_set}

        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(questions)}] {question[:80]}")

        # HyDE: generate hypothetical passage, then embed as RETRIEVAL_DOCUMENT
        [hypothetical_passage] = hyde.transform(question)

        # embed_documents uses RETRIEVAL_DOCUMENT task type — intentional
        [hyde_vec] = embedder.embed_documents([hypothetical_passage])

        points        = store.search(hyde_vec, k=K)
        payloads      = [p.payload for p in points]
        retrieved_ids = dedupe_ranked([make_chunk_id(p["doc_name"], p["page"]) for p in payloads])

        metrics = compute_metrics(retrieved_ids, relevant_ids)
        print("  " + "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

        results.append({
            "question_id":        q.get("financebench_id", i),
            "question":           question,
            "doc_name":           doc_name,
            "gold_pages":         list(gold_pages_set),
            "hypothetical_passage": hypothetical_passage,
            "hyde":               metrics,
        })

    summary = aggregate(results, "hyde")
    print_summary("hyde", summary)
    save_results("hyde", results, summary)


if __name__ == "__main__":
    main()
