"""
LangGraph DAG for the Day 6 routing + reranking + CRAG pipeline.

Graph structure:

    START
      │
      ▼
  classify_node          ← reads question, writes route
      │
      │  conditional_edge: state["route"] → branch
      │
    ┌─┴──────┬────────────┬──────────────┐
    ▼        ▼            ▼              ▼
  direct   hyde      decompose      stepback
    │        │            │              │
    └────────┴────────────┴──────────────┘
                          │
                          ▼
                    rerank_node      ← Day 6: cross-encoder or ColBERT rerank
                          │
                          ▼
                    crag_node        ← Day 6: LLM relevance evaluator
                          │
              ┌───────────┴──────────────┐
              ▼ confidence < threshold   ▼ confidence >= threshold
        web_fallback_node          generate_node
              │                         │
              └─────────────────────────┘
                          │
                          ▼
                         END

Day 5 vs Day 6 change:
  All retrieval branches now converge at rerank_node (not generate_node).
  rerank → crag → conditional branch → [web_fallback →] generate.
"""

from langgraph.graph import StateGraph, START, END

from rag_hub.routing.state import RouterState
from rag_hub.routing.nodes import (
    classify_node,
    direct_node,
    hyde_node,
    decompose_node,
    stepback_node,
    generate_node,
    rerank_node,
    crag_node,
    web_fallback_node,
)

CRAG_THRESHOLD = 0.5


def _route_selector(state: RouterState) -> str:
    return state["route"]  # one of: "direct", "hyde", "decompose", "stepback"


def _crag_selector(state: RouterState) -> str:
    """
    Conditional edge after crag_node.
    Routes to web_fallback if confidence is below the configured threshold, else
    generate. Threshold and the enable flag are read from state (seeded by the
    RetrievalPipeline) so a per-run config actually takes effect — falling back
    to the module default only when unset.
    """
    confidence = state.get("crag_confidence", 1.0)
    threshold = state.get("crag_threshold", CRAG_THRESHOLD)
    enabled = state.get("web_fallback_enabled", True)
    # used_fallback being True means we already ran fallback (shouldn't loop)
    if enabled and not state.get("used_fallback") and confidence < threshold:
        return "web_fallback"
    return "generate"


def build_graph() -> StateGraph:
    graph = StateGraph(RouterState)

    # Existing Day 5 nodes
    graph.add_node("classify", classify_node)
    graph.add_node("direct", direct_node)
    graph.add_node("hyde", hyde_node)
    graph.add_node("decompose", decompose_node)
    graph.add_node("stepback", stepback_node)
    graph.add_node("generate", generate_node)

    # Day 6 new nodes
    graph.add_node("rerank", rerank_node)
    graph.add_node("crag", crag_node)
    graph.add_node("web_fallback", web_fallback_node)

    # Entry
    graph.add_edge(START, "classify")

    # Routing branch (unchanged from Day 5)
    graph.add_conditional_edges(
        "classify",
        _route_selector,
        {
            "direct": "direct",
            "hyde": "hyde",
            "decompose": "decompose",
            "stepback": "stepback",
        },
    )

    # All retrieval branches converge at rerank (Day 6 change: was → generate)
    graph.add_edge("direct", "rerank")
    graph.add_edge("hyde", "rerank")
    graph.add_edge("decompose", "rerank")
    graph.add_edge("stepback", "rerank")

    # rerank → crag
    graph.add_edge("rerank", "crag")

    # crag → web_fallback or generate
    graph.add_conditional_edges(
        "crag",
        _crag_selector,
        {
            "web_fallback": "web_fallback",
            "generate": "generate",
        },
    )

    # web_fallback → generate
    graph.add_edge("web_fallback", "generate")

    # generate → END
    graph.add_edge("generate", END)

    return graph.compile()


# Module-level compiled app
rag_app = build_graph()
