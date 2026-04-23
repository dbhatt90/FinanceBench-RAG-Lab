import os
import re
from typing import List

import vertexai
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from google.oauth2 import service_account
from dotenv import load_dotenv

from rag_hub.query.base import QueryTransform

load_dotenv()

_credentials = service_account.Credentials.from_service_account_file(
    os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)
vertexai.init(
    project=os.getenv("GCP_PROJECT_ID"),
    location=os.getenv("GCP_LOCATION", "us-central1"),
    credentials=_credentials,
)

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

    def __init__(self, n: int = 3, model: str = "gemini-2.5-flash"):
        self.n = n
        self.llm = ChatVertexAI(
            model_name=model,
            temperature=0.4,
            project=os.getenv("GCP_PROJECT_ID"),
            location=os.getenv("GCP_LOCATION", "us-central1"),
            credentials=_credentials,
        )
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
