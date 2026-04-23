import os
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

DECOMPOSITION_PROMPT_TEMPLATE = """\
You are analyzing a question about a company's SEC 10-K & 10Q filings.

Decompose the question below into the minimal set of atomic sub-questions that must \
each be answered independently from the filing to fully answer the original question.

Rules:
- Each sub-question should be answerable from a single section of the filing.
- If the original question is already atomic (single lookup), return it unchanged.
- For conditional questions ("if X then Y, else Z"), produce one sub-question per branch.
- Return ONLY the sub-questions, one per line, no numbering or bullets.

Question: {question}

Sub-questions:"""


class DecompositionTransform(QueryTransform):
    """
    Query decomposition for multi-hop financial questions.

    Splits a complex question (e.g. with comparisons across years, segments,
    or conditional reasoning) into atomic sub-questions that can each be
    retrieved independently.

    Example:
      "Does Adobe have an improving operating margin profile as of FY2022?
       If operating margin is not a useful metric for a company like this,
       then state that and explain why."

      Becomes:
        - "What was Adobe's operating margin in FY2022?"
        - "What was Adobe's operating margin trend over the past 3 years?"
        - "Is operating margin a relevant profitability metric for software/SaaS companies?"

    Each sub-question is retrieved separately; the caller aggregates the chunks.
    """

    def __init__(self, model: str = "gemini-2.5-flash", verbose: bool = True):
        self.verbose = verbose
        self.llm = ChatVertexAI(
            model_name=model,
            temperature=0.0,   # deterministic decomposition
            project=os.getenv("GCP_PROJECT_ID"),
            location=os.getenv("GCP_LOCATION", "us-central1"),
            credentials=_credentials,
        )
        self.prompt = ChatPromptTemplate.from_template(DECOMPOSITION_PROMPT_TEMPLATE)
        self.chain = self.prompt | self.llm

    def transform(self, query: str) -> List[str]:
        """
        Returns a list of atomic sub-questions.
        Falls back to [query] if the LLM returns an empty response.
        """
        response = self.chain.invoke({"question": query})
        raw = response.content.strip()

        sub_questions = [line.strip() for line in raw.splitlines() if line.strip()]

        if self.verbose:
            print(f"\n[Decomposition] Original: {query}")
            print(f"[Decomposition] Sub-questions ({len(sub_questions)}):")
            for i, sq in enumerate(sub_questions, 1):
                print(f"  {i}. {sq}")

        return sub_questions if sub_questions else [query]
