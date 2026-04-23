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

# This is the prompt the LLM receives. Logged at transform() time so callers
# can inspect what the model is asked to produce.
HYDE_PROMPT_TEMPLATE = """\
You are a financial analyst reading a 10-K and 10Q SEC filings.

Write a passage that would appear in a 10-K or 10Q filing and directly answers the question below.
Use the style of a real financial filing: precise numbers, formal language, standard headings.
If you don't know the exact numbers, invent plausible ones consistent with the question.
The passage will be used purely for document retrieval — accuracy matters less than style match.

Question: {question}

Passage (2-4 sentences, filing style):"""


class HyDETransform(QueryTransform):
    """
    Hypothetical Document Embeddings (HyDE).

    Instead of embedding the raw question, we ask an LLM to generate a
    hypothetical 10-K passage that would answer it, then embed that passage
    using RETRIEVAL_DOCUMENT task type.

    Why this helps:
      - Short questions embed in a different vector subspace from long
        document chunks. The hypothetical passage lives in the same dense
        prose space as the actual 10-K text, closing the query-document gap.
      - Particularly effective here because Gemini uses separate task types
        (RETRIEVAL_QUERY vs RETRIEVAL_DOCUMENT) — HyDE lets us embed the
        "query" with RETRIEVAL_DOCUMENT, matching the index task type.

    transform() returns a single-item list containing the hypothetical passage.
    The caller should embed it with RETRIEVAL_DOCUMENT task type (not RETRIEVAL_QUERY).
    """

    def __init__(self, model: str = "gemini-2.5-flash", verbose: bool = True):
        """
        Args:
            model:   Vertex AI model name for generation.
            verbose: If True, print the prompt and generated passage to stdout.
        """
        self.verbose = verbose
        self.llm = ChatVertexAI(
            model_name=model,
            temperature=0.3,   # slight creativity — we want plausible prose
            project=os.getenv("GCP_PROJECT_ID"),
            location=os.getenv("GCP_LOCATION", "us-central1"),
            credentials=_credentials,
        )
        self.prompt = ChatPromptTemplate.from_template(HYDE_PROMPT_TEMPLATE)
        self.chain = self.prompt | self.llm

    def transform(self, query: str) -> List[str]:
        """
        Generate a hypothetical 10-K/Q passage for the given question.

        Returns:
            [hypothetical_passage] — embed with RETRIEVAL_DOCUMENT task type.
        """
        if self.verbose:
            rendered = HYDE_PROMPT_TEMPLATE.format(question=query)
            print("\n" + "─" * 60)
            print("[HyDE] PROMPT SENT TO LLM:")
            print("─" * 60)
            print(rendered)
            print("─" * 60)

        response = self.chain.invoke({"question": query})
        passage = response.content.strip()

        if self.verbose:
            print("[HyDE] GENERATED PASSAGE:")
            print(passage)
            print("─" * 60 + "\n")

        return [passage]
