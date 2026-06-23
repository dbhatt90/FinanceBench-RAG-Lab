"""
Day 9 — ReAct agent evaluation on FinanceBench smoke set.

Runs FinanceAgent on each question, scores generation quality, records
per-step latency, and prints a latency breakdown table at the end.

Usage:
    python scripts/day9_eval.py                 # all 50 questions
    python scripts/day9_eval.py --limit 5       # quick smoke test
    python scripts/day9_eval.py --no-bert       # skip BERTScore (slow)
    python scripts/day9_eval.py --no-mlflow     # skip MLflow logging
"""

import sys
import os
import json
import argparse
from datetime import datetime
from statistics import mean

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from rag_hub.config.settings import SMOKE_50_PATH, RESULTS_DIR
from rag_hub.eval.financebench import load_questions
from rag_hub.eval.generation_metrics import GenerationMetrics
from rag_hub.eval.tracking import mlflow_run, log_metrics
from rag_hub.agents.react_agent import FinanceAgent
from rag_hub.generation.hallucination_detector import HallucinationDetector
from rag_hub.generation.schemas import Answer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--no-bert", action="store_true", help="skip BERTScore")
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--max-iterations", type=int, default=5)
    return p.parse_args()


def _latency_summary(per_question: list) -> dict:
    """Aggregate timing across all questions."""
    tool_totals: dict = {}
    tool_counts: dict = {}
    grand_total = 0.0

    for r in per_question:
        t = r.get("timing", {})
        grand_total += r.get("total_time", 0)
        for tool, durations in t.items():
            tool_totals[tool] = tool_totals.get(tool, 0) + sum(durations)
            tool_counts[tool] = tool_counts.get(tool, 0) + len(durations)

    tool_time = sum(tool_totals.values())
    llm_time = max(grand_total - tool_time, 0)

    return {
        "per_tool": {
            tool: {
                "calls": tool_counts[tool],
                "total_s": round(tool_totals[tool], 2),
                "avg_s": round(tool_totals[tool] / tool_counts[tool], 3),
                "pct": round(tool_totals[tool] / grand_total * 100, 1) if grand_total else 0,
            }
            for tool in tool_totals
        },
        "llm_time_s": round(llm_time, 2),
        "llm_pct": round(llm_time / grand_total * 100, 1) if grand_total else 0,
        "total_s": round(grand_total, 2),
        "avg_per_question_s": round(grand_total / len(per_question), 2) if per_question else 0,
    }


def _print_latency_table(summary: dict):
    print(f"\n{'=' * 65}")
    print("LATENCY BREAKDOWN")
    print(f"{'=' * 65}")
    print(f"  {'Tool':<18} {'Calls':>6} {'Avg(s)':>8} {'Total(s)':>10} {'% runtime':>10}")
    print(f"  {'-' * 56}")
    for tool, s in summary["per_tool"].items():
        print(f"  {tool:<18} {s['calls']:>6} {s['avg_s']:>8.3f} {s['total_s']:>10.2f} {s['pct']:>9.1f}%")
    print(f"  {'LLM (inference)':<18} {'—':>6} {'—':>8} {summary['llm_time_s']:>10.2f} {summary['llm_pct']:>9.1f}%")
    print(f"  {'-' * 56}")
    print(f"  {'TOTAL':<18} {'':>6} {'':>8} {summary['total_s']:>10.2f} {'100.0':>9}%")
    print(f"  Avg per question: {summary['avg_per_question_s']:.2f}s")


def main():
    args = parse_args()
    questions = load_questions(str(SMOKE_50_PATH))[: args.limit]
    print(f"[day9] {len(questions)} questions · max_iterations={args.max_iterations}")

    agent = FinanceAgent(max_iterations=args.max_iterations)
    scorer = GenerationMetrics()
    detector = HallucinationDetector()

    per_question = []

    for i, q in enumerate(questions):
        question = q["question"]
        gold = q.get("answer", "")
        print(f"\n[{i+1:02d}/{len(questions)}] {question[:75]}")

        try:
            result = agent.run(question)
            answer = result["answer"]

            # Generation metrics
            em = scorer.exact_match(answer, gold)
            num = scorer.numeric_match(answer, gold)
            rouge = scorer.rouge_scores(answer, gold)

            bert_f1 = None
            if not args.no_bert:
                try:
                    bert_f1 = scorer.bert_score(answer, gold)
                except Exception as e:
                    print(f"  [BERTScore error] {e}")

            # Hallucination (post-generation, logged only — does not re-run)
            pseudo = Answer(text=answer, citations=[], question=question)
            halluc_rate, _ = detector.check(pseudo)

            row = {
                "question_id": q.get("financebench_id", ""),
                "question": question,
                "answer": answer,
                "gold": gold,
                "em": em,
                "numeric_match": num,
                "rouge1": round(rouge.get("rouge1", 0), 4),
                "rougeL": round(rouge.get("rougeL", 0), 4),
                "bert_score": round(bert_f1, 4) if bert_f1 is not None else None,
                "hallucination_rate": round(halluc_rate, 4),
                "timing": result["timing"],
                "total_time": result["total_time"],
                "iterations": result["iterations"],
            }
            per_question.append(row)

            print(
                f"  em={int(em)} num={num} rouge1={row['rouge1']:.3f} "
                f"halluc={halluc_rate:.2f} "
                f"iters={result['iterations']} t={result['total_time']:.1f}s"
            )

        except Exception as e:
            print(f"  ERROR: {e}")
            per_question.append({"question": question, "gold": gold, "error": str(e)})

    # Aggregate metrics
    valid = [r for r in per_question if "em" in r]
    overall = {}
    if valid:
        overall = {
            "em": round(mean(int(r["em"]) for r in valid), 4),
            "numeric_match_rate": round(
                sum(1 for r in valid if r["numeric_match"] is True) / len(valid), 4
            ),
            "rouge1": round(mean(r["rouge1"] for r in valid), 4),
            "rougeL": round(mean(r["rougeL"] for r in valid), 4),
            "hallucination_rate": round(mean(r["hallucination_rate"] for r in valid), 4),
            "avg_iterations": round(mean(r["iterations"] for r in valid), 2),
            "avg_total_time_s": round(mean(r["total_time"] for r in valid), 2),
        }
        if not args.no_bert:
            bert_vals = [r["bert_score"] for r in valid if r.get("bert_score") is not None]
            if bert_vals:
                overall["bert_score"] = round(mean(bert_vals), 4)

    # Latency breakdown
    latency = _latency_summary(valid)
    _print_latency_table(latency)

    # Save results
    out_dir = RESULTS_DIR / "day9"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_questions": len(valid),
        "overall": overall,
        "latency_summary": latency,
        "per_question": per_question,
    }
    out_path = out_dir / "agent_eval.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {out_path}")
    print(f"[overall] {overall}")

    # MLflow
    if not args.no_mlflow and overall:
        params = {"max_iterations": args.max_iterations, "n_questions": len(valid)}
        with mlflow_run(day=9, config_name="react_agent", params=params):
            log_metrics(overall)
        print("[MLflow] run logged")


if __name__ == "__main__":
    main()
