"""
Day 9 — ReAct agent for FinanceBench 10-K Q&A.

Uses LangGraph's prebuilt create_react_agent with three tools:
  retrieve_10k, web_search, calculator

The LLM (Gemini Flash) decides which tools to call and how many times,
iterating until it can produce a final answer.  This replaces the fixed
classify → retrieve → rerank → CRAG → generate DAG from days 1–8.

Latency breakdown is captured per tool call and returned alongside the
answer so callers can analyse where time is spent.
"""

import time
from typing import Dict, List, TypedDict

from langgraph.prebuilt import create_react_agent

from rag_hub.config.vertex_ai import make_chat_llm
from rag_hub.agents.tools import make_tools

SYSTEM_PROMPT = (
    "You are a financial analyst answering questions about SEC 10-K annual filings. "
    "Always start by calling retrieve_10k with the user's question. "
    "If the retrieved passages are insufficient, call retrieve_10k again with a more "
    "specific or rephrased query. "
    "Use calculator for any arithmetic (percentage change, ratios, etc.). "
    "Use web_search only as a last resort if the corpus has nothing relevant. "
    "Always cite the document name and page number from the passages in your final answer."
)


class AgentResult(TypedDict):
    question: str
    answer: str
    timing: Dict[str, List[float]]   # tool_name → list of per-call durations (s)
    total_time: float                 # wall-clock seconds for the whole run
    iterations: int                   # number of tool calls made


class FinanceAgent:
    def __init__(self, max_iterations: int = 5):
        """
        Args:
            max_iterations: maximum tool-call rounds before the agent must
                            produce a final answer (default 5).
        """
        self.max_iterations = max_iterations
        # Reuse the shared Gemini Flash helper — picks up project/location
        # from vertexai.init() globals, max_retries=2.
        self._llm = make_chat_llm()

    def run(self, question: str) -> AgentResult:
        timing: Dict[str, List[float]] = {}
        tools = make_tools(timing)

        agent = create_react_agent(
            self._llm,
            tools,
            prompt=SYSTEM_PROMPT,
        )

        t0 = time.perf_counter()
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"recursion_limit": self.max_iterations * 2 + 1},
        )
        total = time.perf_counter() - t0

        messages = result.get("messages", [])
        answer = messages[-1].content if messages else ""
        # Each tool response is a ToolMessage; count them as iterations
        iterations = sum(
            1 for m in messages if getattr(m, "type", "") == "tool"
        )

        return AgentResult(
            question=question,
            answer=answer,
            timing=timing,
            total_time=round(total, 3),
            iterations=iterations,
        )
