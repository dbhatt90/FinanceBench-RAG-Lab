"""
Day 5 — LangGraph Router: routing accuracy + retrieval metrics eval.

Two evaluation phases:

  Phase 1 — Routing accuracy
    Run the LLM classifier on all 50 questions, compare predicted route to
    hand-labeled expected route. Report overall accuracy and per-route breakdown.

  Phase 2 — Retrieval metrics (full pipeline)
    Run the complete LangGraph DAG (classify → retrieve → generate) on all 50
    questions. Compare retrieved chunks to FinanceBench gold pages and report:
      recall@5, precision@5, MRR, MAP@5, hit@5, ERR@5
    Also break down retrieval quality per route so we can see which strategy
    retrieves best for its question type.

Hand-labeling rationale (HAND_LABELS dict below):
  direct    — single fact, one number/name/date, no formula needed
  decompose — multi-year, comparison, ratio/formula, conditional
  stepback  — "why", "what drove", risk/legal/strategic context
  hyde      — qualitative/explanatory prose, "nature of", "describe"
"""

import sys
import os
import json
from datetime import datetime
from typing import Dict, List, Set

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag_hub.eval.financebench import load_questions, gold_pages
from rag_hub.eval.retrieval_metrics import (
    recall_at_k, precision_at_k, mrr, map_at_k, hit_rate_at_k, err_at_k,
)
from rag_hub.routing.router import QuestionRouter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SMOKE_PATH = "data/eval/smoke_50.jsonl"
RESULTS_DIR = "eval_results/day5"
K = 5

# Set True to run full pipeline (classify → retrieve → generate) on all 50 Qs.
# Each question makes 1-3 LLM calls depending on route. ~50-150 total API calls.
FULL_PIPELINE = True

# ---------------------------------------------------------------------------
# Hand-labeled ground truth
# ---------------------------------------------------------------------------

HAND_LABELS: Dict[str, str] = {
    # Multi-year averages / YoY comparisons → decompose
    "financebench_id_02981": "decompose",
    "financebench_id_07966": "decompose",
    "financebench_id_06741": "decompose",
    "financebench_id_07507": "decompose",
    "financebench_id_00288": "decompose",
    "financebench_id_00302": "decompose",
    "financebench_id_00460": "decompose",
    "financebench_id_00552": "decompose",
    "financebench_id_00566": "decompose",
    "financebench_id_01077": "decompose",

    # Conditional / multi-part → decompose
    "financebench_id_00222": "decompose",
    "financebench_id_00070": "decompose",
    "financebench_id_01107": "decompose",

    # Complex analyst-framed / formula-based → decompose
    "financebench_id_04735": "decompose",
    "financebench_id_03069": "decompose",
    "financebench_id_03620": "decompose",
    "financebench_id_08286": "decompose",
    "financebench_id_06272": "decompose",   # dividend payout ratio
    "financebench_id_10499": "decompose",   # inventory turnover ratio
    "financebench_id_03473": "decompose",   # ROA = net income / total assets
    "financebench_id_10136": "decompose",   # "calculate a financial metric"
    "financebench_id_04458": "decompose",   # "calculate a financial metric"

    # "What drove" / "why" → stepback
    "financebench_id_00720": "stepback",
    "financebench_id_00603": "stepback",
    "financebench_id_00601": "stepback",
    "financebench_id_01474": "stepback",

    # Risk / legal / qualitative strategic → stepback
    "financebench_id_01091": "stepback",
    "financebench_id_00651": "stepback",
    "financebench_id_01981": "stepback",
    "financebench_id_00790": "stepback",
    "financebench_id_00684": "stepback",
    "financebench_id_00438": "stepback",

    # Nature / purpose / description → hyde
    "financebench_id_01936": "hyde",
    "financebench_id_00995": "hyde",
    "financebench_id_01935": "hyde",
    "financebench_id_01928": "hyde",

    # Single-fact lookups → direct
    "financebench_id_00299": "direct",
    "financebench_id_01491": "direct",
    "financebench_id_07661": "direct",
    "financebench_id_03882": "direct",
    "financebench_id_00746": "direct",
    "financebench_id_00941": "direct",
    "financebench_id_04980": "direct",
    "financebench_id_01328": "direct",
    "financebench_id_01488": "direct",
    "financebench_id_01912": "direct",
    "financebench_id_01319": "direct",
    "financebench_id_00517": "direct",
    "financebench_id_02416": "direct",
    "financebench_id_04209": "direct",
}


# ---------------------------------------------------------------------------
# Helpers shared with day4
# ---------------------------------------------------------------------------

def make_chunk_id(doc_name: str, page: int) -> str:
    return f"{doc_name}_p{page}"


def dedupe_ranked(ids: List[str]) -> List[str]:
    seen: Set[str] = set()
    return [x for x in ids if not (x in seen or seen.add(x))]


def compute_metrics(retrieved_ids: List[str], relevant_ids: Set[str]) -> Dict:
    return {
        f"recall@{K}":    recall_at_k(retrieved_ids, relevant_ids, k=K),
        f"precision@{K}": precision_at_k(retrieved_ids, relevant_ids, k=K),
        "mrr":            mrr(retrieved_ids, relevant_ids),
        f"map@{K}":       map_at_k(retrieved_ids, relevant_ids, k=K),
        f"hit@{K}":       int(hit_rate_at_k(retrieved_ids, relevant_ids, k=K)),
        f"err@{K}":       err_at_k(retrieved_ids, relevant_ids, k=K),
    }


# ---------------------------------------------------------------------------
# Phase 1 — Routing accuracy
# ---------------------------------------------------------------------------

def eval_routing_accuracy(questions: List[Dict], router: QuestionRouter) -> Dict:
    results = []
    correct = 0

    for i, q in enumerate(questions):
        qid = q.get("financebench_id", "")
        question = q["question"]
        predicted = router.route(question)
        expected = HAND_LABELS.get(qid, "UNLABELED")

        is_correct = (predicted == expected) if expected != "UNLABELED" else None
        if is_correct:
            correct += 1

        print(f"  [{i+1:02d}] {predicted:<10} (expected {expected}) — {question[:60]}")
        results.append({
            "question_id": qid,
            "question": question,
            "predicted_route": predicted,
            "expected_route": expected,
            "correct": is_correct,
        })

    labeled = [r for r in results if r["expected_route"] != "UNLABELED"]
    accuracy = correct / len(labeled) if labeled else 0.0

    routes = ["direct", "hyde", "decompose", "stepback"]
    per_route = {}
    for route in routes:
        route_labeled = [r for r in labeled if r["expected_route"] == route]
        route_correct = [r for r in route_labeled if r["correct"]]
        per_route[route] = {
            "total": len(route_labeled),
            "correct": len(route_correct),
            "accuracy": round(len(route_correct) / len(route_labeled), 4) if route_labeled else 0.0,
        }

    return {
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "labeled": len(labeled),
        "per_route": per_route,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Phase 2 — Full pipeline with retrieval metrics
# ---------------------------------------------------------------------------

def run_pipeline_eval(questions: List[Dict]) -> Dict:
    """
    Runs the LangGraph DAG on all 50 questions and computes retrieval metrics.

    For each question:
      1. Invoke the graph → get state["route"] and state["docs"]
      2. Convert docs to chunk IDs: "{doc_name}_p{page}"
      3. Compare against FinanceBench gold pages
      4. Compute recall@5, precision@5, MRR, MAP@5, hit@5, ERR@5

    Also aggregates metrics per route so we can compare:
      Does decompose retrieve better than direct on multi-hop questions?
      Does stepback improve recall on "why" questions?
    """
    from rag_hub.routing.graph import rag_app

    per_question = []

    for i, q in enumerate(questions):
        qid = q.get("financebench_id", "")
        question = q["question"]
        doc_name = q["doc_name"]
        gold = gold_pages(q)
        relevant_ids = {make_chunk_id(doc_name + ".pdf", p) for p in gold}

        print(f"\n[{i+1:02d}/{len(questions)}] {question[:75]}")

        try:
            final_state = rag_app.invoke({"question": question})
            route = final_state.get("route", "unknown")
            docs = final_state.get("docs", [])

            retrieved_ids = dedupe_ranked([
                make_chunk_id(d["doc_name"], d["page"]) for d in docs
            ])

            metrics = compute_metrics(retrieved_ids, relevant_ids)
            answer = final_state.get("answer", "")

            print(f"  route={route}  docs={len(docs)}  hit@{K}={metrics[f'hit@{K}']}  "
                  f"recall@{K}={metrics[f'recall@{K}']:.3f}  mrr={metrics['mrr']:.3f}")
            print(f"  answer: {answer[:100]}")

            per_question.append({
                "question_id": qid,
                "question": question,
                "doc_name": doc_name,
                "gold_pages": list(gold),
                "route": route,
                "expected_route": HAND_LABELS.get(qid, "UNLABELED"),
                "n_docs_retrieved": len(docs),
                "metrics": metrics,
                "answer": answer,
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            per_question.append({
                "question_id": qid,
                "question": question,
                "error": str(e),
            })

    # Aggregate overall metrics
    valid = [r for r in per_question if "metrics" in r]
    metric_keys = list(valid[0]["metrics"].keys()) if valid else []

    overall = {
        mk: round(sum(r["metrics"][mk] for r in valid) / len(valid), 4)
        for mk in metric_keys
    } if valid else {}

    # Per-route aggregation
    routes = ["direct", "hyde", "decompose", "stepback"]
    per_route = {}
    for route in routes:
        route_qs = [r for r in valid if r.get("route") == route]
        per_route[route] = {
            "n": len(route_qs),
            **({
                mk: round(sum(r["metrics"][mk] for r in route_qs) / len(route_qs), 4)
                for mk in metric_keys
            } if route_qs else {}),
        }

    return {
        "overall": overall,
        "per_route": per_route,
        "per_question": per_question,
    }


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_routing_summary(eval_result: Dict):
    print(f"\n{'=' * 60}")
    print("ROUTING ACCURACY")
    print(f"{'=' * 60}")
    print(f"  Overall: {eval_result['accuracy']:.1%}  "
          f"({eval_result['correct']}/{eval_result['labeled']} labeled)")
    print(f"\n  {'Route':<12} {'Total':>5} {'Correct':>7} {'Accuracy':>10}")
    print(f"  {'-'*36}")
    for route, s in eval_result["per_route"].items():
        print(f"  {route:<12} {s['total']:>5} {s['correct']:>7} {s['accuracy']:>9.1%}")

    misses = [r for r in eval_result["results"] if r["correct"] is False]
    if misses:
        print(f"\n  Misclassified ({len(misses)}):")
        for r in misses:
            print(f"    expected={r['expected_route']} got={r['predicted_route']}: "
                  f"{r['question'][:70]}")


def print_retrieval_summary(pipeline: Dict):
    print(f"\n{'=' * 60}")
    print("RETRIEVAL METRICS  (full LangGraph pipeline)")
    print(f"{'=' * 60}")
    ov = pipeline["overall"]
    print(f"  Overall ({len([r for r in pipeline['per_question'] if 'metrics' in r])} questions):")
    for k, v in ov.items():
        print(f"    {k:<16} {v:.4f}")

    print(f"\n  Per-route breakdown:")
    print(f"  {'Route':<12} {'N':>3}  " + "  ".join(f"{k:<10}" for k in ov.keys()))
    print(f"  {'-'*70}")
    for route, stats in pipeline["per_route"].items():
        n = stats.get("n", 0)
        vals = "  ".join(f"{stats.get(k, 0):<10.4f}" for k in ov.keys()) if n > 0 else "—"
        print(f"  {route:<12} {n:>3}  {vals}")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_results(routing_eval: Dict, pipeline: Dict = None):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "routing_accuracy": {
            "accuracy": routing_eval["accuracy"],
            "correct": routing_eval["correct"],
            "labeled": routing_eval["labeled"],
            "per_route": routing_eval["per_route"],
        },
        "per_question_routing": routing_eval["results"],
    }
    if pipeline:
        output["retrieval_metrics"] = {
            "overall": pipeline["overall"],
            "per_route": pipeline["per_route"],
        }
        output["per_question_pipeline"] = pipeline["per_question"]

    path = os.path.join(RESULTS_DIR, "routing_eval.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n[INFO] Saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    questions = load_questions(SMOKE_PATH)
    print(f"[INFO] Loaded {len(questions)} questions")

    router = QuestionRouter(verbose=False)

    print("\n[PHASE 1] Routing accuracy vs hand labels...")
    routing_eval = eval_routing_accuracy(questions, router)
    print_routing_summary(routing_eval)

    pipeline = None
    if FULL_PIPELINE:
        print("\n[PHASE 2] Full pipeline retrieval metrics on all 50 questions...")
        pipeline = run_pipeline_eval(questions)
        print_retrieval_summary(pipeline)

    save_results(routing_eval, pipeline)


if __name__ == "__main__":
    main()
