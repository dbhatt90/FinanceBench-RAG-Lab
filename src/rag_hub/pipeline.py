"""
Unified RetrievalPipeline — the single entry point for the full RAG pipeline.

Wraps the LangGraph DAG (routing → retrieval → rerank → CRAG → generate) and
exposes a clean, configurable API used by all downstream days.

Reranker injection: the chosen reranker is written into the nodes._singletons dict
before graph invocation. This is the same lazy-singleton pattern used throughout
nodes.py — keeping graph.py and nodes.py unaware of which reranker was selected.

Usage:
    pipeline = RetrievalPipeline(reranker="bge", crag_enabled=True)
    result = pipeline.run("What was Apple's revenue in FY2022?")
    # result keys: question, answer, route, docs, reranked_docs,
    #              crag_confidence, crag_labels, used_fallback, reranker_type
"""

from typing import Dict, Literal, Optional

from rag_hub.routing import nodes as _nodes
from rag_hub.routing.graph import build_graph

RankerType = Literal["bge", "colbert", "none"]


class RetrievalPipeline:
    def __init__(
        self,
        reranker: RankerType = "bge",
        top_k: int = 5,
        crag_enabled: bool = False,
        crag_threshold: float = 0.5,
        web_fallback_enabled: bool = True,
    ):
        """
        Args:
            reranker:             "bge" | "colbert" | "none"
            top_k:                number of docs to return after reranking
            crag_enabled:         whether to run the CRAG evaluator node
            crag_threshold:       confidence below which web fallback fires
            web_fallback_enabled: if False, CRAG evaluates but never triggers fallback
        """
        self.reranker_type = reranker
        self.top_k = top_k
        self.crag_enabled = crag_enabled
        self.crag_threshold = crag_threshold
        self.web_fallback_enabled = web_fallback_enabled

        # Register chosen reranker into the nodes singleton dict
        self._setup_reranker(reranker)

        # Patch CRAG evaluator threshold if it was already initialised
        if "crag_evaluator" in _nodes._singletons:
            _nodes._singletons["crag_evaluator"].threshold = crag_threshold

        # Build fresh graph (picks up patched singletons at runtime)
        self._graph = build_graph()

    def _setup_reranker(self, reranker: RankerType) -> None:
        if reranker == "none":
            _nodes._singletons.pop("reranker", None)
            return

        if reranker == "bge":
            from rag_hub.rerankers.bge_reranker import BGEReranker
            _nodes._singletons["reranker"] = BGEReranker()
        elif reranker == "colbert":
            from rag_hub.rerankers.colbert_reranker import ColBERTReranker
            _nodes._singletons["reranker"] = ColBERTReranker()

    def run(self, question: str) -> Dict:
        """
        Full pipeline: route → retrieve → rerank → CRAG eval → [web fallback] → generate.

        Returns the final LangGraph state dict with diagnostic fields:
          question, answer, route, docs, reranked_docs,
          crag_confidence, crag_labels, used_fallback, reranker_type
        """
        initial = {"question": question}

        # When CRAG is disabled, pre-set confidence high so _crag_selector
        # always routes to generate without calling the evaluator.
        # We do this by patching crag_node's behaviour via the evaluator threshold.
        if not self.crag_enabled:
            _nodes._singletons["crag_evaluator"] = _DisabledCRAGEvaluator()
        elif "crag_evaluator" not in _nodes._singletons or isinstance(
            _nodes._singletons.get("crag_evaluator"), _DisabledCRAGEvaluator
        ):
            from rag_hub.crag.evaluator import CRAGEvaluator
            _nodes._singletons["crag_evaluator"] = CRAGEvaluator(
                threshold=self.crag_threshold
            )

        if not self.web_fallback_enabled:
            _nodes._singletons["_web_fallback_disabled"] = True
        else:
            _nodes._singletons.pop("_web_fallback_disabled", None)

        return self._graph.invoke(initial)

    def run_retrieval_only(self, question: str) -> Dict:
        """
        Runs the pipeline up to and including CRAG evaluation, then stops.
        Does not call the LLM generator — cheaper for retrieval-only evals.

        Returns: {question, route, docs, reranked_docs, crag_confidence,
                  crag_labels, used_fallback, reranker_type}
        """
        # Build a retrieval-only graph (stops before generate)
        from langgraph.graph import StateGraph, START, END
        from rag_hub.routing.state import RouterState
        from rag_hub.routing.nodes import (
            classify_node, direct_node, hyde_node,
            decompose_node, stepback_node,
            rerank_node, crag_node, web_fallback_node,
        )
        from rag_hub.routing.graph import _route_selector, _crag_selector, CRAG_THRESHOLD

        if not self.crag_enabled:
            _nodes._singletons["crag_evaluator"] = _DisabledCRAGEvaluator()
        elif "crag_evaluator" not in _nodes._singletons or isinstance(
            _nodes._singletons.get("crag_evaluator"), _DisabledCRAGEvaluator
        ):
            from rag_hub.crag.evaluator import CRAGEvaluator
            _nodes._singletons["crag_evaluator"] = CRAGEvaluator(
                threshold=self.crag_threshold
            )

        g = StateGraph(RouterState)
        g.add_node("classify", classify_node)
        g.add_node("direct", direct_node)
        g.add_node("hyde", hyde_node)
        g.add_node("decompose", decompose_node)
        g.add_node("stepback", stepback_node)
        g.add_node("rerank", rerank_node)
        g.add_node("crag", crag_node)
        g.add_node("web_fallback", web_fallback_node)

        g.add_edge(START, "classify")
        g.add_conditional_edges("classify", _route_selector, {
            "direct": "direct", "hyde": "hyde",
            "decompose": "decompose", "stepback": "stepback",
        })
        for branch in ("direct", "hyde", "decompose", "stepback"):
            g.add_edge(branch, "rerank")
        g.add_edge("rerank", "crag")

        if self.web_fallback_enabled:
            g.add_conditional_edges("crag", _crag_selector, {
                "web_fallback": "web_fallback",
                "generate": END,
            })
            g.add_edge("web_fallback", END)
        else:
            g.add_edge("crag", END)

        app = g.compile()
        return app.invoke({"question": question})


class _DisabledCRAGEvaluator:
    """Stub that always returns high confidence so the fallback never fires."""
    threshold = 1.0

    def evaluate(self, question, docs):
        return 1.0, ["relevant"] * min(3, len(docs))
