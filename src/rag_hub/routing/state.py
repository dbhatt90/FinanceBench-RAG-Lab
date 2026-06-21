from typing import List, Dict, Optional
from typing_extensions import TypedDict


class RouterState(TypedDict, total=False):
    """
    Shared state that flows through every node in the LangGraph DAG.

    Day 5 fields:
      question — original user question (immutable)
      route    — set by classify_node; read by conditional edge
      docs     — set by whichever retrieval node runs
      answer   — set by generate_node; final output

    Day 6 additions:
      reranked_docs    — set by rerank_node; generate_node prefers this over docs
      crag_confidence  — set by crag_node; float in [0, 1]
      crag_labels      — per-doc relevance labels from CRAG evaluator
      used_fallback    — True if web_fallback_node fired
      reranker_type    — class name of reranker used (for eval logging)

    Pipeline config (seeded by RetrievalPipeline into the initial state so the
    conditional edges use the configured values, not hardcoded constants):
      crag_threshold       — confidence below which web fallback fires
      web_fallback_enabled — if False, CRAG evaluates but never triggers fallback
    """
    question: str
    route: str
    docs: List[Dict]
    answer: str
    reranked_docs: List[Dict]
    crag_confidence: float
    crag_labels: List[str]
    used_fallback: bool
    reranker_type: str
    crag_threshold: float
    web_fallback_enabled: bool
