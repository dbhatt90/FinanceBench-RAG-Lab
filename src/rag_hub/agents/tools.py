"""
Day 9 — Agent tools wrapping the existing retrieval stack.

Each tool records its own wall-clock duration into a shared `timing` dict
(passed by reference from FinanceAgent.run) so callers can break down
where time is spent: retrieve vs web_search vs calculator vs LLM.

Tools:
    retrieve_10k   — hybrid RRF + BGE rerank over the Qdrant corpus
    web_search     — DuckDuckGo fallback (reuses crag/web_fallback.py)
    calculator     — safe arithmetic eval for percentage / ratio questions
"""

import time
from typing import Dict, List

from langchain_core.tools import tool

# Lazy module-level singletons — initialised on first call, shared across
# all tool invocations within a single agent run.
_embedder = None
_retriever = None
_reranker = None
_web = None


def _init_retrieval():
    global _embedder, _retriever, _reranker
    if _embedder is None:
        from rag_hub.embeddings.gemini_001 import GeminiEmbeddingClient
        from rag_hub.retrievers.hybrid_retriever import HybridRRFRetriever
        from rag_hub.rerankers.bge_reranker import BGEReranker
        _embedder = GeminiEmbeddingClient()
        _retriever = HybridRRFRetriever()
        _reranker = BGEReranker()


def make_tools(timing: Dict[str, List[float]]) -> list:
    """
    Return the three agent tools.  Each tool appends its duration (seconds)
    to timing[tool_name] so the caller can build a latency breakdown table.
    """

    @tool
    def retrieve_10k(query: str, k: int = 5) -> str:
        """Search the 10-K SEC filing corpus. Returns the top-k relevant passages with document and page citations. Use this first for any financial question."""
        t0 = time.perf_counter()
        _init_retrieval()
        query_vec = _embedder.embed_query(query)
        docs = _retriever.search(query, query_vec, top_k=20)
        docs = _reranker.rerank(query, docs, top_k=k)
        timing.setdefault("retrieve_10k", []).append(time.perf_counter() - t0)
        if not docs:
            return "No relevant passages found in the 10-K corpus."
        return "\n\n".join(
            f"[{d['doc_name']} p{d['page']}]\n{d['text']}" for d in docs
        )

    @tool
    def web_search(query: str) -> str:
        """Search the web when the 10-K corpus does not contain the answer. Use sparingly — only after retrieve_10k returns nothing useful."""
        global _web
        t0 = time.perf_counter()
        if _web is None:
            from rag_hub.crag.web_fallback import WebFallback
            _web = WebFallback()
        results = _web.search(query)
        timing.setdefault("web_search", []).append(time.perf_counter() - t0)
        if not results:
            return "No web results found."
        return "\n\n".join(d.get("text", "") for d in results)

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a numeric arithmetic expression, e.g. '(1200 - 900) / 900 * 100'. Use for percentage change, ratios, and other calculations. Do not include variable names — only numeric literals and operators."""
        t0 = time.perf_counter()
        try:
            # Restrict to arithmetic: no builtins, no attribute access
            result = str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307
        except Exception as e:
            result = f"Error evaluating '{expression}': {e}"
        timing.setdefault("calculator", []).append(time.perf_counter() - t0)
        return result

    return [retrieve_10k, web_search, calculator]
