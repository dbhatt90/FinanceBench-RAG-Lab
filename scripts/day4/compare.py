"""
Day 4 comparison: load all per-technique JSON result files and print:
  1. Overall side-by-side metric table
  2. Per-question-type breakdown (hit@5 and recall@5)
  3. Per-question winner list

Usage:
    cd <repo_root>
    python scripts/day4/compare.py
    python scripts/day4/compare.py --dir eval_results/day4 --smoke data/eval/smoke_50.jsonl
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

TECHNIQUE_ORDER = ["baseline", "hyde", "rag_fusion", "decomposition"]
RESULTS_DIR     = "eval_results/day4"
SMOKE_PATH      = "data/eval/smoke_50.jsonl"


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_results(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_question_types(smoke_path: Path) -> dict:
    """Returns {financebench_id: question_type}."""
    mapping = {}
    with open(smoke_path, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            mapping[q["financebench_id"]] = q.get("question_type", "unknown")
    return mapping


def print_table(title: str, techniques: list, metrics: list, data: dict):
    """Generic table printer. data[technique][metric] = float."""
    col_w = 13
    header = f"{'Metric':<16}" + "".join(f"{t:>{col_w}}" for t in techniques)
    sep = "=" * len(header)
    print(f"\n{sep}")
    print(title)
    print(sep)
    print(header)
    print("-" * len(header))
    for mk in metrics:
        row = f"{mk:<16}"
        vals = [data[t].get(mk, float("nan")) for t in techniques]
        best = max(v for v in vals if v == v)   # nan-safe max
        for val in vals:
            cell = f"{val:.4f}" + ("*" if abs(val - best) < 1e-6 else " ")
            row += f"{cell:>{col_w}}"
        print(row)
    print(sep)
    print("* = best for that metric")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",   default=RESULTS_DIR)
    parser.add_argument("--smoke", default=SMOKE_PATH)
    args = parser.parse_args()

    results_dir = Path(args.dir)
    qtype_map   = load_question_types(Path(args.smoke))

    # ── Load result files ─────────────────────────────────────────────
    all_data: dict[str, dict] = {}
    for technique in TECHNIQUE_ORDER:
        path = results_dir / f"{technique}.json"
        if path.exists():
            all_data[technique] = load_results(path)
        else:
            print(f"[WARN] No results for '{technique}' — skipping ({path})")

    if not all_data:
        print("[ERROR] No result files found. Run the eval scripts first.")
        return

    techniques  = list(all_data.keys())
    metrics     = list(next(iter(all_data.values()))["summary"].keys())

    # ── 1. Overall table ──────────────────────────────────────────────
    overall = {t: all_data[t]["summary"] for t in techniques}
    print_table("DAY 4  —  QUERY TRANSLATION  (overall)", techniques, metrics, overall)

    # ── 2. Build per-question index ───────────────────────────────────
    # per_q[qid] = {question, doc_name, question_type, baseline: {...}, hyde: {...}, ...}
    per_q: dict = {}

    for technique, data in all_data.items():
        for row in data["results"]:
            qid = row.get("question_id", row["question"])
            if qid not in per_q:
                per_q[qid] = {
                    "question":      row["question"],
                    "doc_name":      row["doc_name"],
                    "question_type": qtype_map.get(qid, "unknown"),
                }
            per_q[qid][technique] = row.get(technique, {})

    # ── 3. Per-question-type breakdown ────────────────────────────────
    qtypes = sorted({v["question_type"] for v in per_q.values()})

    for focus_metric in ("hit@5", "recall@5"):
        print(f"\n{'─'*60}")
        print(f"By question_type  —  {focus_metric}")
        print(f"{'─'*60}")

        col_w = 13
        header = f"{'question_type':<22}" + "".join(f"{t:>{col_w}}" for t in techniques) + f"{'n':>5}"
        print(header)
        print("-" * len(header))

        for qt in qtypes:
            rows_for_type = [v for v in per_q.values() if v["question_type"] == qt]
            n = len(rows_for_type)
            vals = {}
            for t in techniques:
                scores = [r[t].get(focus_metric, 0) for r in rows_for_type if t in r]
                vals[t] = sum(scores) / len(scores) if scores else float("nan")

            best = max(v for v in vals.values() if v == v)
            row_str = f"{qt:<22}"
            for t in techniques:
                v = vals[t]
                cell = f"{v:.4f}" + ("*" if abs(v - best) < 1e-6 else " ")
                row_str += f"{cell:>{col_w}}"
            row_str += f"{n:>5}"
            print(row_str)

    # ── 4. Per-question winner list ───────────────────────────────────
    hit_key = "hit@5"
    print(f"\n{'─'*70}")
    print(f"Per-question winner  ({hit_key})")
    print(f"{'─'*70}")
    wins: dict[str, int] = defaultdict(int)

    for qid, data in per_q.items():
        scores = {t: data[t].get(hit_key, 0) for t in techniques if t in data}
        best_score = max(scores.values())
        winners = [t for t, v in scores.items() if v == best_score]
        for w in winners:
            wins[w] += 1
        winner_str = " & ".join(winners) if best_score > 0 else "none"
        qt = data["question_type"]
        print(f"  [{winner_str:<22}] [{qt:<20}] {data['question'][:55]}")

    print(f"\nWin counts  ({hit_key}):")
    for t in techniques:
        print(f"  {t:<18} {wins[t]}")


if __name__ == "__main__":
    main()
