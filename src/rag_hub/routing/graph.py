"""
LangGraph DAG for the Day 5 routing pipeline.

How LangGraph works:
  - A StateGraph is a directed graph where nodes are Python functions and
    edges define execution order.
  - Every node receives the full state dict, returns a partial update,
    and LangGraph merges the update back into state.
  - Conditional edges read a field from state (here: "route") and branch
    to the appropriate next node — this is what makes routing possible.
  - START and END are special LangGraph sentinels for the entry and exit points.

Graph structure (ASCII):

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
                    generate_node   ← reads question + docs, writes answer
                          │
                          ▼
                         END

Why this topology?
  - All four retrieval branches converge to a single generate_node.
    Generation logic is identical regardless of which branch ran —
    keeping it DRY and making it trivial to swap the generator later.
  - classify_node → conditional_edge → retrieval node is the key LangGraph
    pattern: one node sets a routing key, the edge reads it, LangGraph
    dispatches to the right branch. No if/else in the graph definition.
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
)


def _route_selector(state: RouterState) -> str:
    """
    Conditional edge function.

    LangGraph calls this after classify_node completes. It reads the "route"
    field that classify_node wrote and returns a node name. LangGraph then
    jumps execution to that node.

    This is a pure routing function — no LLM calls, no side effects.
    """
    return state["route"]  # one of: "direct", "hyde", "decompose", "stepback"


def build_graph() -> StateGraph:
    """
    Constructs and compiles the routing DAG.

    Returns a compiled LangGraph app. Call app.invoke({"question": "..."})
    to run the full pipeline and get back the final state including "answer".

    Why compile()?
      compile() validates the graph (no orphan nodes, all edges reachable,
      START/END connected) and returns a Runnable — the same interface as
      any LangChain chain, so it works with .invoke(), .stream(), .batch().
    """
    graph = StateGraph(RouterState)

    # -----------------------------------------------------------------------
    # Register nodes
    # Each call gives the node a string name used in edge definitions.
    # -----------------------------------------------------------------------
    graph.add_node("classify", classify_node)
    graph.add_node("direct", direct_node)
    graph.add_node("hyde", hyde_node)
    graph.add_node("decompose", decompose_node)
    graph.add_node("stepback", stepback_node)
    graph.add_node("generate", generate_node)

    # -----------------------------------------------------------------------
    # Entry edge: START → classify
    # Every invocation starts here.
    # -----------------------------------------------------------------------
    graph.add_edge(START, "classify")

    # -----------------------------------------------------------------------
    # Conditional edge: classify → one of {direct, hyde, decompose, stepback}
    #
    # add_conditional_edges(source, path_fn, path_map):
    #   source   — the node whose output triggers the branch
    #   path_fn  — function that reads state and returns a string key
    #   path_map — dict mapping that key to the next node name
    #
    # LangGraph calls path_fn(state) after "classify" finishes, looks up the
    # returned key in path_map, and jumps to the mapped node.
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Convergence edges: all retrieval branches → generate
    # After whichever branch runs, execution always continues to generate.
    # -----------------------------------------------------------------------
    graph.add_edge("direct", "generate")
    graph.add_edge("hyde", "generate")
    graph.add_edge("decompose", "generate")
    graph.add_edge("stepback", "generate")

    # -----------------------------------------------------------------------
    # Exit edge: generate → END
    # -----------------------------------------------------------------------
    graph.add_edge("generate", END)

    return graph.compile()


# Module-level compiled app — import this for one-liner usage:
#   from rag_hub.routing.graph import rag_app
#   result = rag_app.invoke({"question": "What was Apple's revenue?"})
rag_app = build_graph()
