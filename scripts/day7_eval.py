"""
Day 7 evaluation: generation quality on FinanceBench.

Runs Day7Pipeline on --limit questions and computes:
  hallucination_rate, Self-RAG confidence, ROUGE-L, BERTScore (optional), RAGAS (optional)

Writes: eval_results/day_07_results.md + eval_results/day_07_raw.json
"""
import sys, os, json, argparse
from datetime import datetime
from typing import List, Dict
from statistics import mean, stdev

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag_hub.eval.financebench import load_questions
from rag_hub.eval.generation_metrics import GenerationMetrics
from rag_hub.embeddings.gemini_001 import GeminiEmbeddingClient
from rag_hub.vectorstore.qdrant_store import QdrantStore
from rag_hub.retrievers.bm25_retriever import BM25Retriever
from rag_hub.generation.day7_pipeline import Day7Pipeline

SMOKE_PATH = "data/eval/smoke_50.jsonl"
RESULTS_DIR = "eval_results"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = "financebench_v1"
BM25_PATH = "data/processed/bm25_index.pkl"
RAGAS_LIMIT = 10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--no-ragas", action="store_true")
    p.add_argument("--no-bert", action="store_true")
    return p.parse_args()


def run_eval(questions, pipeline, metrics, args):
    rows = []
    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] {q['question'][:70]}...")
        result = pipeline.run(q["question"])
        answer = result["answer"]
        prediction = answer.text
        reference = q.get("answer", "")
        rouge = metrics.rouge_scores(prediction, reference)

        row = {
            "question": q["question"],
            "prediction": prediction,
            "reference": reference,
            "hallucination_rate": answer.hallucination_rate,
            "confidence": answer.confidence,
            "generation_iterations": answer.generation_iterations,
            "num_citations": len(answer.citations),
            **rouge,
            "bert_score": None,
            "ragas_faithfulness": None,
            "ragas_answer_relevancy": None,
        }

        if not args.no_bert and reference and prediction.lower() != "i don't know":
            try:
                row["bert_score"] = metrics.bert_score(prediction, reference)
            except Exception as e:
                print(f"  [WARN] BERTScore failed: {e}")

        if not args.no_ragas and i < RAGAS_LIMIT and reference:
            contexts = [d.get("text", "") for d in result["docs"][:5]]
            try:
                ragas = metrics.ragas_metrics(q["question"], prediction, contexts, reference)
                row["ragas_faithfulness"] = ragas["faithfulness"]
                row["ragas_answer_relevancy"] = ragas["answer_relevancy"]
            except Exception as e:
                print(f"  [WARN] RAGAS failed: {e}")

        rows.append(row)
    return rows


def write_results(rows):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    def avg(vals):
        v = [x for x in vals if x is not None]
        return round(mean(v), 4) if v else "—"

    corrective_fired = sum(1 for r in rows if r["generation_iterations"] > 1)

    md = [
        f"# Day 7 Eval Results — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"\n**Questions evaluated:** {len(rows)}",
        f"**Corrective loop fired:** {corrective_fired}/{len(rows)} ({100*corrective_fired/len(rows):.1f}%)",
        "\n## Aggregate Metrics\n",
        "| Metric | Mean |",
        "|--------|------|",
        f"| Hallucination Rate | {avg([r['hallucination_rate'] for r in rows])} |",
        f"| Self-RAG Confidence | {avg([r['confidence'] for r in rows])} |",
        f"| ROUGE-1 | {avg([r['rouge1'] for r in rows])} |",
        f"| ROUGE-L | {avg([r['rougeL'] for r in rows])} |",
        f"| BERTScore F1 | {avg([r['bert_score'] for r in rows])} |",
        f"| RAGAS Faithfulness (n≤{RAGAS_LIMIT}) | {avg([r['ragas_faithfulness'] for r in rows])} |",
        f"| RAGAS Answer Relevancy (n≤{RAGAS_LIMIT}) | {avg([r['ragas_answer_relevancy'] for r in rows])} |",
        f"\n**Citation coverage:** {sum(1 for r in rows if r['num_citations'] > 0)}/{len(rows)} answers cited",
    ]

    if rows:
        ex = rows[0]
        md += [
            "\n## Example Answer\n",
            f"**Q:** {ex['question']}",
            f"\n**A:** {ex['prediction']}",
            f"\n**Ref:** {ex['reference']}",
            f"\n**Metrics:** hallucination={ex['hallucination_rate']}, confidence={ex['confidence']}, rougeL={ex['rougeL']}",
        ]

    path = os.path.join(RESULTS_DIR, "day_07_results.md")
    with open(path, "w") as f:
        f.write("\n".join(md))

    json_path = os.path.join(RESULTS_DIR, "day_07_raw.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"\nResults → {path}")


def main():
    args = parse_args()
    questions = load_questions(SMOKE_PATH)[:args.limit]
    print(f"Loaded {len(questions)} questions")

    embedder = GeminiEmbeddingClient()
    store = QdrantStore(url=QDRANT_URL, collection_name=COLLECTION)
    bm25 = BM25Retriever.load(BM25_PATH)
    pipeline = Day7Pipeline(store=store, bm25=bm25, embedder=embedder)
    metrics = GenerationMetrics()

    rows = run_eval(questions, pipeline, metrics, args)
    write_results(rows)


if __name__ == "__main__":
    main()
