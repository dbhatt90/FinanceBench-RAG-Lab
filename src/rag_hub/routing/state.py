from typing import List, Dict, Annotated
from typing_extensions import TypedDict


class RouterState(TypedDict):
    """
    Shared state that flows through every node in the LangGraph DAG.

    LangGraph passes the full state dict into each node function and merges
    the returned partial dict back. Nodes only need to read what they depend
    on and write what they produce.

    Fields:
      question — the original user question (set at graph invocation, never mutated)
      route    — set by classify_node; read by the conditional edge to branch
      docs     — set by whichever retrieval node runs; read by generate_node
      answer   — set by generate_node; the final output
    """
    question: str
    route: str
    docs: List[Dict]
    answer: str
