"""
Shared utilities for Day 4 per-technique eval scripts.
Not a public module — prefixed with _ to signal internal use.
"""
import json
import os
from datetime import datetime
from typing import List, Dict, Set

from rag_hub.eval.financebench import load_questions, gold_pages
from rag_hub.eval.retrieval_metrics import (
    recall_at_k,
    precision_at_k,
    mrr,
    map_at_k,
    hit_rate_at_k,
    err_at_k,
)

SMOKE_PATH = "data/eval/smoke_50.jsonl"
RESULTS_DIR = "eval_results/day4"
K = 5


def make_chunk_id(doc_name: str, page: int) -> str:
    return f"{doc_name}_p{page}"


def dedupe_ranked(ids: List[str]) -> List[str]:
    seen: Set[str] = set()
    return [x for x in ids if not (x in seen or seen.add(x))]


def compute_metrics(retrieved_ids: List[str], relevant_ids: Set[str], k: int = K) -> Dict:
    return {
        f"recall@{k}":    recall_at_k(retrieved_ids, relevant_ids, k=k),
        f"precision@{k}": precision_at_k(retrieved_ids, relevant_ids, k=k),
        "mrr":            mrr(retrieved_ids, relevant_ids),
        f"map@{k}":       map_at_k(retrieved_ids, relevant_ids, k=k),
        f"hit@{k}":       int(hit_rate_at_k(retrieved_ids, relevant_ids, k=k)),
        f"err@{k}":       err_at_k(retrieved_ids, relevant_ids, k=k),
    }


def mean_metrics(rows: List[Dict], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows)


def aggregate(results: List[Dict], retriever_key: str) -> Dict:
    metric_keys = list(results[0][retriever_key].keys())
    return {
        mk: round(mean_metrics([r[retriever_key] for r in results], mk), 4)
        for mk in metric_keys
    }


def save_results(technique: str, results: List[Dict], summary: Dict) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, f"{technique}.json")
    payload = {
        "technique":   technique,
        "timestamp":   datetime.utcnow().isoformat(),
        "n_questions": len(results),
        "k":           K,
        "summary":     summary,
        "results":     results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[INFO] Saved → {output_path}")
    return output_path


def print_summary(technique: str, summary: Dict):
    metric_keys = list(summary.keys())
    print(f"\n{'=' * 60}")
    print(f"SUMMARY  —  {technique.upper()}")
    print(f"{'=' * 60}")
    for mk in metric_keys:
        print(f"  {mk:<16} {summary[mk]:.4f}")
