"""
Day 8 — Unified retrieval evaluation harness.

Runs a registry of representative pipeline configurations over a fixed eval set
(smoke_50), computes every retrieval metric in one pass, writes a canonical
per-config result file, and logs params + metrics to MLflow. The Streamlit
dashboard (app/eval_dashboard.py) reads these runs to plot metric trends.

Scope: retrieval metrics only (recall / precision / MRR / MAP / Hit / ERR / nDCG).
Generation metrics are evaluated separately (see scripts/day7_eval.py) and will be
folded into this harness later.

Usage:
    python scripts/run_eval.py                      # all configs, smoke_50, k=5
    python scripts/run_eval.py --configs bge_crag   # one config
    python scripts/run_eval.py --k 10 --limit 20    # smaller, k=10
"""

import sys
import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime
from typing import Dict, List, Set

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag_hub.config.settings import SMOKE_50_PATH, RESULTS_DIR
from rag_hub.eval.financebench import load_questions, gold_pages, make_chunk_id, dedupe_ranked
from rag_hub.eval.retrieval_metrics import (
    recall_at_k, precision_at_k, mrr, map_at_k, hit_rate_at_k, err_at_k, ndcg_at_k,
)
from rag_hub.eval.tracking import mlflow_run, log_metrics
from rag_hub.pipeline import RetrievalPipeline

# ---------------------------------------------------------------------------
# Config registry — each entry is one comparable point on the retrieval-quality
# trend. `day` tags which day of the series the configuration represents; the
# pipeline kwargs are applied to RetrievalPipeline (same collection throughout).
# ---------------------------------------------------------------------------
CONFIGS: Dict[str, Dict] = {
    "hybrid_rrf": {
        "day": 2,
        "desc": "Hybrid dense+BM25 RRF, no rerank (Day 2 baseline)",
        "kwargs": {"reranker": "none", "crag_enabled": False},
    },
    "bge_rerank": {
        "day": 6,
        "desc": "Hybrid + BGE cross-encoder rerank (Day 6)",
        "kwargs": {"reranker": "bge", "crag_enabled": False},
    },
    "bge_crag": {
        "day": 6,
        "desc": "Hybrid + BGE rerank + CRAG web fallback (Day 6 full)",
        "kwargs": {"reranker": "bge", "crag_enabled": True, "crag_threshold": 0.5},
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default="all",
                   help="comma-separated config names, or 'all'")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    p.add_argument("--resume", action="store_true",
                   help="skip questions already completed in an existing result file")
    p.add_argument("--timeout", type=int, default=120,
                   help="seconds before a single question is skipped (default: 120)")
    return p.parse_args()


def compute_metrics(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> Dict:
    return {
        f"recall@{k}":    recall_at_k(retrieved_ids, relevant_ids, k=k),
        f"precision@{k}": precision_at_k(retrieved_ids, relevant_ids, k=k),
        "mrr":            mrr(retrieved_ids, relevant_ids),
        f"map@{k}":       map_at_k(retrieved_ids, relevant_ids, k=k),
        f"hit@{k}":       int(hit_rate_at_k(retrieved_ids, relevant_ids, k=k)),
        f"err@{k}":       err_at_k(retrieved_ids, relevant_ids, k=k),
        f"ndcg@{k}":      ndcg_at_k(retrieved_ids, relevant_ids, k=k),
    }


def load_checkpoint(name: str, cfg: Dict) -> List[Dict]:
    day_dir = RESULTS_DIR / f"day{cfg['day']}"
    path = day_dir / f"retrieval_eval_{name}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f).get("per_question", [])
    return []


def eval_config(name: str, cfg: Dict, questions: List[Dict], k: int,
                resume: bool = False, timeout: int = 120) -> Dict:
    print(f"\n{'=' * 70}\nCONFIG: {name} (day {cfg['day']}) — {cfg['desc']}\n{'=' * 70}")

    # Load prior results when resuming
    done_questions: Dict[str, Dict] = {}
    if resume:
        for r in load_checkpoint(name, cfg):
            if "metrics" in r:
                done_questions[r["question"]] = r
        if done_questions:
            print(f"  [resume] {len(done_questions)} questions already done, skipping them")

    pipeline = None  # lazy init — skip entirely if everything is cached
    per_question = list(done_questions.values())  # seed with completed results

    # Partial result template for incremental writes
    partial = {
        "day": cfg["day"], "config": name, "description": cfg["desc"],
        "k": k, "params": cfg["kwargs"],
    }

    for i, q in enumerate(questions):
        if q["question"] in done_questions:
            print(f"[{i+1:02d}/{len(questions)}] SKIP (cached): {q['question'][:60]}")
            continue
        if pipeline is None:
            pipeline = RetrievalPipeline(**cfg["kwargs"])
        question = q["question"]
        doc_name = q["doc_name"]
        gold = gold_pages(q)
        relevant_ids = {make_chunk_id(doc_name + ".pdf", p) for p in gold}

        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(pipeline.run_retrieval_only, question)
                state = future.result(timeout=timeout)
            docs = state.get("reranked_docs") or state.get("docs", [])
            corpus_docs = [d for d in docs if d.get("source") != "web"]
            retrieved_ids = dedupe_ranked(
                [make_chunk_id(d["doc_name"], d["page"]) for d in corpus_docs]
            )
            metrics = compute_metrics(retrieved_ids, relevant_ids, k)
            per_question.append({
                "question_id": q.get("financebench_id", ""),
                "question": question,
                "route": state.get("route"),
                "used_fallback": state.get("used_fallback", False),
                "metrics": metrics,
            })
            print(f"[{i+1:02d}/{len(questions)}] {question[:60]:<60} "
                  f"hit@{k}={metrics[f'hit@{k}']} recall@{k}={metrics[f'recall@{k}']:.2f}")
        except FuturesTimeoutError:
            print(f"[{i+1:02d}/{len(questions)}] TIMEOUT ({timeout}s): {question[:60]}")
            per_question.append({"question": question, "error": f"timeout>{timeout}s"})
        except Exception as e:
            print(f"[{i+1:02d}/{len(questions)}] ERROR: {e}")
            per_question.append({"question": question, "error": str(e)})

        # Incremental checkpoint after every question
        _write_partial(partial, per_question)

    valid = [r for r in per_question if "metrics" in r]
    metric_keys = list(valid[0]["metrics"].keys()) if valid else []
    overall = {
        mk: round(sum(r["metrics"][mk] for r in valid) / len(valid), 4)
        for mk in metric_keys
    } if valid else {}

    return {
        "day": cfg["day"],
        "config": name,
        "description": cfg["desc"],
        "k": k,
        "n_questions": len(valid),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "params": cfg["kwargs"],
        "overall": overall,
        "per_question": per_question,
    }


def _write_partial(template: Dict, per_question: List[Dict]) -> None:
    """Write an in-progress result file so --resume can recover after a crash."""
    valid = [r for r in per_question if "metrics" in r]
    metric_keys = list(valid[0]["metrics"].keys()) if valid else []
    overall = {
        mk: round(sum(r["metrics"][mk] for r in valid) / len(valid), 4)
        for mk in metric_keys
    } if valid else {}
    result = {
        **template,
        "n_questions": len(valid),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "overall": overall,
        "per_question": per_question,
    }
    write_result(result)


def write_result(result: Dict) -> str:
    day_dir = RESULTS_DIR / f"day{result['day']}"
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / f"retrieval_eval_{result['config']}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return str(path)


def main():
    args = parse_args()
    names = list(CONFIGS) if args.configs == "all" else args.configs.split(",")
    questions = load_questions(str(SMOKE_50_PATH))[:args.limit]
    print(f"Loaded {len(questions)} questions; configs: {names}; k={args.k}")

    summary = []
    for name in names:
        cfg = CONFIGS[name]
        result = eval_config(name, cfg, questions, args.k, resume=args.resume, timeout=args.timeout)
        path = write_result(result)
        print(f"  → {path}")
        print(f"  overall: {result['overall']}")
        summary.append((name, result["day"], result["overall"]))

        if not args.no_mlflow:
            params = {**cfg["kwargs"], "k": args.k, "n_questions": result["n_questions"]}
            with mlflow_run(day=cfg["day"], config_name=name, params=params):
                log_metrics(result["overall"])

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for name, day, overall in summary:
        print(f"  day{day:<2} {name:<14} {overall}")
    if not args.no_mlflow:
        print(f"\nLogged to MLflow → run `mlflow ui` then open http://localhost:5000")


if __name__ == "__main__":
    main()
