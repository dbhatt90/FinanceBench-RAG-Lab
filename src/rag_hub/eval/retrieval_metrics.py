# eval/retrieval_metrics.py

from typing import List, Set


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    retrieved_k = retrieved[:k]
    hits = sum(1 for r in set(retrieved_k) if r in relevant)
    return hits / len(relevant) if relevant else 0.0


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    retrieved_k = retrieved[:k]
    hits = sum(1 for r in retrieved_k if r in relevant)
    return hits / k if k > 0 else 0.0


def hit_rate_at_k(retrieved: List[str], relevant: Set[str], k: int) -> bool:
    return any(r in relevant for r in retrieved[:k])


def mrr(retrieved: List[str], relevant: Set[str]) -> float:
    for idx, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / idx
    return 0.0


def map_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    retrieved_k = retrieved[:k]

    hits = 0
    precision_sum = 0.0

    for i, r in enumerate(retrieved_k, start=1):
        if r in relevant:
            hits += 1
            precision_sum += hits / i  # precision@i

    return precision_sum / len(relevant) if relevant else 0.0


def rrf_score(rank_lists: List[List[str]], k: int = 60) -> dict:
    """
    rank_lists: list of ranked lists (e.g. [dense_results, bm25_results])
    returns: dict {doc_id: score}
    """
    scores = {}

    for rank_list in rank_lists:
        for rank, doc_id in enumerate(rank_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)

    return scores