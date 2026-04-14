import pytest

from rag_hub.eval.retrieval_metrics import (
    recall_at_k,
    precision_at_k,
    mrr,
    map_at_k,
    hit_rate_at_k,
    rrf_score,
)


# -----------------------------
# Test data (simple + controlled)
# -----------------------------
retrieved = ["A", "B", "C", "D", "E"]
relevant = {"B", "D"}


# -----------------------------
# Recall@k
# -----------------------------
def test_recall_at_k():
    # Top-3 = ["A","B","C"] → only "B" is relevant → 1/2
    assert recall_at_k(retrieved, relevant, k=3) == 0.5

    # Top-5 → both B and D found → 2/2
    assert recall_at_k(retrieved, relevant, k=5) == 1.0


# -----------------------------
# Precision@k
# -----------------------------
def test_precision_at_k():
    # Top-3 → 1 relevant out of 3
    assert precision_at_k(retrieved, relevant, k=3) == 1 / 3

    # Top-5 → 2 relevant out of 5
    assert precision_at_k(retrieved, relevant, k=5) == 2 / 5


# -----------------------------
# Hit Rate@k
# -----------------------------
def test_hit_rate_at_k():
    assert hit_rate_at_k(retrieved, relevant, k=1) == 0  # A not relevant
    assert hit_rate_at_k(retrieved, relevant, k=2) == 1  # B found


# -----------------------------
# MRR
# -----------------------------
def test_mrr():
    # First relevant = B at rank 2 → 1/2
    assert mrr(retrieved, relevant) == 0.5


# -----------------------------
# MAP@k
# -----------------------------
def test_map_at_k():
    """
    Relevant at:
    rank 2 → precision = 1/2
    rank 4 → precision = 2/4

    MAP = (1/2 + 2/4) / 2 = (0.5 + 0.5) / 2 = 0.5
    """
    assert map_at_k(retrieved, relevant, k=5) == 0.5


# -----------------------------
# RRF Score
# -----------------------------
def test_rrf_score():
    ranked_lists = [
        ["A", "B", "C"],   # system 1
        ["B", "D", "A"],   # system 2
    ]

    scores = rrf_score(ranked_lists, k=60)

    # B appears rank 2 and rank 1 → should be highest
    assert scores["B"] > scores["A"]
    assert "D" in scores


# -----------------------------
# Edge cases
# -----------------------------
def test_empty_relevant():
    assert recall_at_k(retrieved, set(), k=5) == 0.0
    assert precision_at_k(retrieved, set(), k=5) == 0.0
    assert mrr(retrieved, set()) == 0.0
    assert map_at_k(retrieved, set(), k=5) == 0.0


def test_no_overlap():
    retrieved = ["X", "Y", "Z"]
    relevant = {"A", "B"}

    assert recall_at_k(retrieved, relevant, k=3) == 0.0
    assert precision_at_k(retrieved, relevant, k=3) == 0.0
    assert mrr(retrieved, relevant) == 0.0
    assert map_at_k(retrieved, relevant, k=3) == 0.0