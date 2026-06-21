import re
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from rag_hub.config.settings import GEMINI_LLM_MODEL
from rag_hub.config.vertex_ai import make_chat_llm
from rag_hub.query.base import QueryTransform

load_dotenv()

MULTI_QUERY_PROMPT_TEMPLATE = """\
You are helping improve document retrieval over SEC 10-K and 10Q filings.

Generate {n} different phrasings of the question below. Each rephrasing should:
- Use different financial terminology (e.g., "revenue" vs "net sales", "profit" vs "income")
- Vary sentence structure
- Stay semantically identical to the original

Return ONLY the rephrased questions, one per line, no numbering or bullets.

Original question: {question}

Rephrased questions:"""


class MultiQueryTransform(QueryTransform):
    """
    Multi-query expansion.

    Generates N rephrased variants of the original question. The original
    is prepended so retrieval always includes the unmodified query.

    Used as a building block inside RAGFusionRetriever.
    """

    def __init__(self, n: int = 3, model: str = GEMINI_LLM_MODEL):
        self.n = n
        self.llm = make_chat_llm(model, temperature=0.4)
        self.prompt = ChatPromptTemplate.from_template(MULTI_QUERY_PROMPT_TEMPLATE)
        self.chain = self.prompt | self.llm

    def transform(self, query: str) -> List[str]:
        """
        Returns [original_query, variant_1, ..., variant_n].
        Falls back gracefully if the LLM returns fewer variants than requested.
        """
        response = self.chain.invoke({"question": query, "n": self.n})
        raw = response.content.strip()

        variants = [line.strip() for line in raw.splitlines() if line.strip()]
        # deduplicate while preserving order
        seen = set()
        unique_variants = []
        for v in variants:
            if v not in seen:
                seen.add(v)
                unique_variants.append(v)

        return [query] + unique_variants[: self.n]
