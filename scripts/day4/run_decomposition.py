"""
Day 4 decomposition eval: split multi-hop questions into sub-questions,
retrieve for each, dedupe-merge by RRF, evaluate.
Results saved to eval_results/day4/decomposition.json.
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
from rag_hub.query.decomposition import DecompositionTransform
from rag_hub.eval.retrieval_metrics import rrf_score


def main():
    questions = load_questions(SMOKE_PATH)
    print(f"[INFO] Loaded {len(questions)} questions")

    embedder     = GeminiEmbeddingClient()
    store        = QdrantStore(collection="financebench_v1")
    decomposer   = DecompositionTransform(verbose=True)

    results = []

    for i, q in enumerate(questions):
        question       = q["question"]
        doc_name       = q["doc_name"]
        gold_pages_set = gold_pages(q)
        doc_name_pdf   = doc_name + ".pdf"
        relevant_ids   = {make_chunk_id(doc_name_pdf, p) for p in gold_pages_set}

        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(questions)}] {question[:80]}")

        sub_questions = decomposer.transform(question)

        # ── Retrieve for each sub-question, then RRF-fuse ────────────────
        candidate_k = K * 3
        all_ranked_ids = []
        payload_map = {}

        for sq in sub_questions:
            vec    = embedder.embed_query(sq)
            points = store.search(vec, k=candidate_k)
            payloads = [p.payload for p in points]

            ranked_ids = [make_chunk_id(p["doc_name"], p["page"]) for p in payloads]
            all_ranked_ids.append(ranked_ids)

            for payload in payloads:
                cid = make_chunk_id(payload["doc_name"], payload["page"])
                payload_map[cid] = payload

        fused_scores = rrf_score(all_ranked_ids, k=60)
        sorted_ids   = sorted(fused_scores, key=fused_scores.get, reverse=True)
        retrieved_ids = dedupe_ranked(sorted_ids)[:K]

        metrics = compute_metrics(retrieved_ids, relevant_ids)
        print("  " + "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

        results.append({
            "question_id":   q.get("financebench_id", i),
            "question":      question,
            "doc_name":      doc_name,
            "gold_pages":    list(gold_pages_set),
            "sub_questions": sub_questions,
            "decomposition": metrics,
        })

    summary = aggregate(results, "decomposition")
    print_summary("decomposition", summary)
    save_results("decomposition", results, summary)


if __name__ == "__main__":
    main()
