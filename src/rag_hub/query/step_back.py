from typing import List

from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from rag_hub.config.settings import GEMINI_LLM_MODEL
from rag_hub.config.vertex_ai import make_chat_llm
from rag_hub.query.base import QueryTransform

load_dotenv()

STEP_BACK_PROMPT_TEMPLATE = """\
You are analyzing a question about a company's SEC 10-K or 10-Q filing.

Reformulate the specific question below into a broader, more general question that:
- Removes specific numbers, dates, or named metrics
- Asks about the general concept, trend, or principle behind the original question
- Can be answered by a broader section of the filing (e.g. MD&A, Risk Factors, Business Overview)

The step-back question should retrieve context that helps answer the original question.

Rules:
- Return ONLY the step-back question, nothing else.
- If the original question is already broad (e.g. "What are the main risks?"), return it unchanged.

Original question: {question}

Step-back question:"""


class StepBackTransform(QueryTransform):
    """
    Step-back prompting for financial 10-K/Q questions.

    Reformulates a specific question (e.g. "What was Apple's iPhone revenue
    in Q3 2022?") into a broader one ("What are Apple's iPhone revenue trends
    and segment reporting?") before retrieval.

    Why this helps:
      - Specific questions often land in narrow chunks (a single table row).
        The step-back question retrieves surrounding context — trends, MD&A
        commentary, segment breakdowns — that gives the LLM richer grounding
        to answer the original specific question.
      - Particularly useful for "why" and "how" questions where the answer
        lives in explanatory prose rather than a single data point.

    transform() returns [step_back_question, original_question] so the caller
    can retrieve for both and union the results.
    """

    def __init__(self, model: str = GEMINI_LLM_MODEL, verbose: bool = True):
        self.verbose = verbose
        self.llm = make_chat_llm(model, temperature=0.0)
        self.prompt = ChatPromptTemplate.from_template(STEP_BACK_PROMPT_TEMPLATE)
        self.chain = self.prompt | self.llm

    def transform(self, query: str) -> List[str]:
        """
        Returns [step_back_question, original_question].

        Both are retrieved; union gives broad context + specific lookup.
        Falls back to [query] if the LLM returns empty.
        """
        response = self.chain.invoke({"question": query})
        step_back = response.content.strip()

        if self.verbose:
            print(f"\n[StepBack] Original:  {query}")
            print(f"[StepBack] Step-back: {step_back}")

        if not step_back or step_back.lower() == query.lower():
            return [query]

        # Return step-back first — broader context; original is focused follow-up.
        return [step_back, query]
