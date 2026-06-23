"""
Day 2/3 — Chunking strategy sweep.

Builds one Qdrant collection per strategy (recursive, section_aware, parent_child,
semantic), runs hybrid retrieval eval on smoke_20 questions, and compares
NDCG@10, Recall@10, MRR, and MAP across all strategies.

Results written to eval_results/day2/chunking_sweep_<timestamp>.json.

Usage:
    python scripts/day2_chunking_sweep.py              # build + eval all
    python scripts/day2_chunking_sweep.py --force      # rebuild from scratch
    python scripts/day2_chunking_sweep.py --eval-only  # skip indexing
    python scripts/day2_chunking_sweep.py --strategy recursive
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.run_chunker_sweep import main

if __name__ == "__main__":
    main()
