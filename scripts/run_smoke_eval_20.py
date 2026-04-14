import os
import json
from datetime import datetime

from rag_hub.eval.financebench import load_questions, gold_pages
from rag_hub.embeddings.gemini_001 import GeminiEmbeddingClient
from rag_hub.vectorstore.qdrant_store import QdrantStore
from rag_hub.generation.gemini_LLM import GeminiFlashGenerator
from rag_hub.eval.metrics import hit_at_k, exact_match, token_f1


SMOKE_PATH = "data/eval/smoke_20.jsonl"
OUTPUT_PATH = "eval_results/day1_smoke.json"


# def load_jsonl(path):
#     with open(path, "r", encoding="utf-8") as f:
#         return [json.loads(line) for line in f]


def main():
    # -----------------------------
    # Load smoke questions
    # -----------------------------
    questions = load_questions(SMOKE_PATH)

    print(f"[INFO] Loaded {len(questions)} questions")

    # -----------------------------
    # Init components
    # -----------------------------
    embedder = GeminiEmbeddingClient()
    store = QdrantStore(collection="financebench_v1")
    generator = GeminiFlashGenerator()

    results = []

    # -----------------------------
    # Loop over questions
    # -----------------------------
    for i, q in enumerate(questions):
        question = q["question"]
        gold_answer = q["answer"]
        doc_name = q["doc_name"]
        gold = gold_pages(q)

        print(f"\n[{i+1}/{len(questions)}] {question}")

        # -----------------------------
        # Retrieval
        # -----------------------------
        query_vec = embedder.embed_query(question)

        retrieved = store.search(query_vec, k=5)

        # convert Qdrant objects → payload dicts
        retrieved_chunks = [r.payload for r in retrieved]

        # -----------------------------
        # Debug: retrieved vs gold
        # -----------------------------
        print(f"\n  GOLD  doc: '{doc_name}'  pages: {gold}")
        print(f"  RETRIEVED top-5:")
        for rc in retrieved_chunks[:5]:
            match_doc  = rc.get("doc_name") == doc_name
            match_page = rc.get("page") in gold
            print(
                f"    doc='{rc.get('doc_name')}'  page={rc.get('page')}"
                f"  | doc_match={match_doc}  page_match={match_page}"
            )

        # -----------------------------
        # Metrics: Hit@K
        # -----------------------------
        hit = hit_at_k(
            retrieved_chunks,
            gold_pages=gold,
            doc_name=doc_name,
            k=5
        )

        # -----------------------------
        # Generation + answer debug
        # -----------------------------
        pred = generator.generate(question, retrieved_chunks)

        # -----------------------------
        # Metrics: EM + F1
        # -----------------------------
        em = exact_match(pred, gold_answer)
        f1 = token_f1(pred, gold_answer)

        print(f"\n  PRED : {pred}")
        print(f"  GOLD : {gold_answer}")
        print(f"  EM={int(em)}  F1={f1:.2f}  Hit@5={int(hit)}")

        row = {
            "question_id": q.get("id", i),
            "question": question,
            "doc_name": doc_name,
            "gold_pages": list(gold),
            "hit@5": hit,
            "em": em,
            "f1": round(f1, 4),
            "pred": pred,
            "gold": gold_answer,
        }

        results.append(row)

    # -----------------------------
    # Aggregate metrics
    # -----------------------------
    hit_mean = sum(r["hit@5"] for r in results) / len(results)
    em_mean = sum(r["em"] for r in results) / len(results)
    f1_mean = sum(r["f1"] for r in results) / len(results)

    # -----------------------------
    # Print summary table
    # -----------------------------
    print("\n==============================")
    print("SMOKE EVAL RESULTS")
    print("==============================")

    for r in results:
        print(
            f"{r['question_id']:>3} | "
            f"Hit@5: {int(r['hit@5'])} | "
            f"EM: {int(r['em'])} | "
            f"F1: {r['f1']:.2f}"
        )

    print("\n------------------------------")
    print(f"Hit@5 Mean: {hit_mean:.3f}")
    print(f"EM Mean   : {em_mean:.3f}")
    print(f"F1 Mean   : {f1_mean:.3f}")
    print("------------------------------")

    # -----------------------------
    # Save results with timestamp
    # -----------------------------
    os.makedirs("eval_results", exist_ok=True)

    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": {
            "hit@5": hit_mean,
            "em": em_mean,
            "f1": f1_mean,
        },
        "results": results,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n[INFO] Saved results → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()