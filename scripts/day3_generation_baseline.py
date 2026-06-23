"""
Day 3 — Baseline generation quality eval.

Runs hybrid retrieval (top-5) + plain Gemini generation (no citations, no
hallucination detection) across 4 chunking strategies on smoke_20 questions.
Scores predictions with Exact Match and Token F1 against gold answers.

Results written to eval_results/day3/generation_baseline_<timestamp>.json.

Use this as a baseline to compare against Day 7's citation-aware + corrective
generation pipeline.

Usage:
    python scripts/day3_generation_baseline.py
    python scripts/day3_generation_baseline.py --strategy recursive
    python scripts/day3_generation_baseline.py --retriever dense
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.run_generation_eval import main

if __name__ == "__main__":
    main()
