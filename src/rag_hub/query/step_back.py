from typing import List

from rag_hub.query.base import QueryTransform


class StepBackTransform(QueryTransform):
    """
    Step-back prompting (stub — not evaluated in Day 4).

    The technique asks an LLM to reformulate a specific question as a
    more general one. For example:
      "What was Apple's iPhone revenue in Q3 2022?"
      → "What are Apple's iPhone revenue trends?"

    This is most useful for questions that require broad background context
    before narrowing to specifics. For FinanceBench's direct lookup questions
    the benefit is marginal — implemented fully in a future day.
    """

    def transform(self, query: str) -> List[str]:
        # stub: returns the original query unchanged
        return [query]
