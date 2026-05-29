"""
Day 6 — Reranking + CRAG evaluation.

Three phases, each building on the last:

  Phase 1 — Baseline (no reranker, no CRAG)
    Reproduces Day 5 retrieval numbers for a fair before/after comparison.
    Uses RetrievalPipeline(reranker="none", crag_enabled=False).

  Phase 2 — Reranking only (BGE, then ColBERT)
    Isolates the contribution of cross-encoder reranking.
    Uses RetrievalPipeline(reranker="bge"|"colbert", crag_enabled=False).

  Phase 3 — Full pipeline (BGE + CRAG + web fallback)
    Measures CRAG fallback rate and net effect on final retrieval metrics.
    Uses RetrievalPipeline(reranker="bge", crag_enabled=True).

Outputs:
  eval_results/day6/comparison.json    — per-phase metric table
  eval_results/day6/crag_analysis.json — per-question CRAG decisions + fallback rate
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
from rag_hub.pipeline import RetrievalPipeline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SMOKE_PATH = "data/eval/smoke_50.jsonl"
RESULTS_DIR = "eval_results/day6"
K = 5

# Set False to skip ColBERT phase (slow on CPU; ~10 min for 50 questions)
RUN_COLBERT = True

# ---------------------------------------------------------------------------
# Helpers
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
# Phase runner
# ---------------------------------------------------------------------------

def run_phase(
    pipeline: RetrievalPipeline,
    questions: List[Dict],
    phase_name: str,
) -> Dict:
    """
    Run pipeline.run_retrieval_only() on all questions and compute metrics.
    Returns per-question results and aggregated metrics.
    """
    print(f"\n{'=' * 60}")
    print(f"PHASE: {phase_name}")
    print(f"{'=' * 60}")

    per_question = []

    for i, q in enumerate(questions):
        qid = q.get("financebench_id", "")
        question = q["question"]
        doc_name = q["doc_name"]
        gold = gold_pages(q)
        relevant_ids = {make_chunk_id(doc_name + ".pdf", p) for p in gold}

        print(f"\n[{i+1:02d}/{len(questions)}] {question[:75]}")

        try:
            state = pipeline.run_retrieval_only(question)
            route = state.get("route", "unknown")

            # Use reranked_docs if present, else fall back to docs
            docs = state.get("reranked_docs") or state.get("docs", [])

            # Filter out web results when computing corpus retrieval metrics
            corpus_docs = [d for d in docs if not d.get("source") == "web"]

            retrieved_ids = dedupe_ranked([
                make_chunk_id(d["doc_name"], d["page"]) for d in corpus_docs
            ])

            metrics = compute_metrics(retrieved_ids, relevant_ids)
            crag_conf = state.get("crag_confidence", None)
            used_fallback = state.get("used_fallback", False)

            conf_str = f"  crag={crag_conf:.3f}" if crag_conf is not None else ""
            fallback_str = "  [WEB FALLBACK]" if used_fallback else ""
            print(
                f"  route={route}  docs={len(corpus_docs)}"
                f"  hit@{K}={metrics[f'hit@{K}']}"
                f"  recall@{K}={metrics[f'recall@{K}']:.3f}"
                f"  mrr={metrics['mrr']:.3f}"
                f"{conf_str}{fallback_str}"
            )

            per_question.append({
                "question_id": qid,
                "question": question,
                "doc_name": doc_name,
                "gold_pages": list(gold),
                "route": route,
                "n_docs_retrieved": len(corpus_docs),
                "metrics": metrics,
                "crag_confidence": crag_conf,
                "used_fallback": used_fallback,
                "reranker_type": state.get("reranker_type"),
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            per_question.append({
                "question_id": qid,
                "question": question,
                "error": str(e),
            })

    # Aggregate
    valid = [r for r in per_question if "metrics" in r]
    if not valid:
        return {"overall": {}, "per_route": {}, "per_question": per_question}

    metric_keys = list(valid[0]["metrics"].keys())
    overall = {
        mk: round(sum(r["metrics"][mk] for r in valid) / len(valid), 4)
        for mk in metric_keys
    }

    routes = ["direct", "hyde", "decompose", "stepback"]
    per_route = {}
    for route in routes:
        rqs = [r for r in valid if r.get("route") == route]
        per_route[route] = {
            "n": len(rqs),
            **(
                {mk: round(sum(r["metrics"][mk] for r in rqs) / len(rqs), 4)
                 for mk in metric_keys}
                if rqs else {}
            ),
        }

    # CRAG stats
    fallback_count = sum(1 for r in valid if r.get("used_fallback"))
    per_route_fallback = {}
    for route in routes:
        rqs = [r for r in valid if r.get("route") == route]
        fb = sum(1 for r in rqs if r.get("used_fallback"))
        per_route_fallback[route] = {"n": len(rqs), "fallbacks": fb}

    return {
        "phase": phase_name,
        "overall": overall,
        "per_route": per_route,
        "crag_fallback_total": fallback_count,
        "crag_fallback_per_route": per_route_fallback,
        "per_question": per_question,
    }


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_comparison(results: Dict[str, Dict]):
    phases = list(results.keys())
    if not phases:
        return

    metric_keys = list(results[phases[0]].get("overall", {}).keys())

    print(f"\n{'=' * 80}")
    print("COMPARISON: Before/After Reranking + CRAG")
    print(f"{'=' * 80}")
    header = f"  {'Metric':<18}" + "".join(f"  {p[:18]:<18}" for p in phases)
    print(header)
    print(f"  {'-' * (18 + 20 * len(phases))}")

    for mk in metric_keys:
        row = f"  {mk:<18}"
        for phase in phases:
            val = results[phase].get("overall", {}).get(mk, 0)
            row += f"  {val:<18.4f}"
        print(row)

    print(f"\n  Per-route MRR breakdown:")
    print(f"  {'Route':<12}" + "".join(f"  {p[:14]:<14}" for p in phases))
    print(f"  {'-' * (12 + 16 * len(phases))}")
    for route in ["direct", "hyde", "decompose", "stepback"]:
        row = f"  {route:<12}"
        for phase in phases:
            val = results[phase].get("per_route", {}).get(route, {}).get("mrr", 0)
            n = results[phase].get("per_route", {}).get(route, {}).get("n", 0)
            row += f"  {val:.4f} (n={n})"
        print(row)


def print_crag_analysis(full_result: Dict):
    print(f"\n{'=' * 60}")
    print("CRAG ANALYSIS (Phase 3 — BGE + CRAG)")
    print(f"{'=' * 60}")

    total = len([r for r in full_result["per_question"] if "metrics" in r])
    fallbacks = full_result.get("crag_fallback_total", 0)
    print(f"  Total questions: {total}")
    print(f"  Web fallback triggered: {fallbacks} ({fallbacks/total:.1%})")

    print(f"\n  Per-route fallback rate:")
    for route, stats in full_result.get("crag_fallback_per_route", {}).items():
        n = stats.get("n", 0)
        fb = stats.get("fallbacks", 0)
        rate = fb / n if n else 0.0
        print(f"    {route:<12} {fb}/{n} ({rate:.1%})")

    # Show questions where CRAG fired
    fired = [r for r in full_result["per_question"]
             if r.get("used_fallback") and "metrics" in r]
    if fired:
        print(f"\n  Questions with web fallback:")
        for r in fired:
            print(f"    [{r.get('route','?')}] conf={r.get('crag_confidence',0):.3f}  "
                  f"{r['question'][:70]}")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_results(results: Dict[str, Dict], full_result: Dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    comparison = {
        "timestamp": datetime.utcnow().isoformat(),
        "phases": {
            phase: {
                "overall": data["overall"],
                "per_route": data["per_route"],
            }
            for phase, data in results.items()
        },
    }
    comp_path = os.path.join(RESULTS_DIR, "comparison.json")
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n[INFO] Saved comparison → {comp_path}")

    if full_result:
        crag_path = os.path.join(RESULTS_DIR, "crag_analysis.json")
        crag_output = {
            "timestamp": datetime.utcnow().isoformat(),
            "crag_fallback_total": full_result.get("crag_fallback_total", 0),
            "crag_fallback_per_route": full_result.get("crag_fallback_per_route", {}),
            "per_question": [
                {k: v for k, v in r.items() if k != "gold_pages"}
                for r in full_result.get("per_question", [])
                if "metrics" in r
            ],
        }
        with open(crag_path, "w") as f:
            json.dump(crag_output, f, indent=2)
        print(f"[INFO] Saved CRAG analysis → {crag_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    questions = load_questions(SMOKE_PATH)
    print(f"[INFO] Loaded {len(questions)} questions from {SMOKE_PATH}")

    results: Dict[str, Dict] = {}
    full_result = None

    # -----------------------------------------------------------------------
    # Phase 1 — Baseline (no reranker, no CRAG)
    # -----------------------------------------------------------------------
    print("\n[PHASE 1] Baseline — no reranker, no CRAG...")
    p1 = RetrievalPipeline(reranker="none", crag_enabled=False)
    results["baseline"] = run_phase(p1, questions, "baseline (no rerank, no CRAG)")

    # -----------------------------------------------------------------------
    # Phase 2a — BGE reranker only
    # -----------------------------------------------------------------------
    print("\n[PHASE 2a] BGE cross-encoder reranker, no CRAG...")
    p2_bge = RetrievalPipeline(reranker="bge", crag_enabled=False)
    results["+BGE"] = run_phase(p2_bge, questions, "+BGE reranker")

    # -----------------------------------------------------------------------
    # Phase 2b — ColBERT reranker only
    # -----------------------------------------------------------------------
    if RUN_COLBERT:
        print("\n[PHASE 2b] ColBERT late-interaction reranker, no CRAG...")
        p2_colbert = RetrievalPipeline(reranker="colbert", crag_enabled=False)
        results["+ColBERT"] = run_phase(p2_colbert, questions, "+ColBERT reranker")

    # -----------------------------------------------------------------------
    # Phase 3 — Full pipeline (BGE + CRAG + web fallback)
    # -----------------------------------------------------------------------
    print("\n[PHASE 3] Full pipeline — BGE + CRAG + web fallback...")
    p3 = RetrievalPipeline(
        reranker="bge",
        crag_enabled=True,
        crag_threshold=0.5,
        web_fallback_enabled=True,
    )
    full_result = run_phase(p3, questions, "+BGE+CRAG+WebFallback")
    results["+BGE+CRAG"] = full_result

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print_comparison(results)
    if full_result:
        print_crag_analysis(full_result)

    save_results(results, full_result)


if __name__ == "__main__":
    main()
