"""
Question Router — classifies a financial question into one of four retrieval strategies.

Why pure LLM (no keyword rules)?
  FinanceBench questions are long, analyst-framed, and structurally misleading
  to keyword patterns. "What is the FY2022 dividend payout ratio?" starts with
  "what is" (looks like a prose question) but is actually a formula calculation
  requiring two numbers. Rules cannot distinguish surface form from semantic intent.
  An LLM trained on financial language handles this naturally.

Routes:
  direct    — Single-fact lookup. One number, name, or date retrievable from
               one chunk. No arithmetic needed.
               Example: "What was Apple's total revenue in FY2022?"

  hyde      — Conceptual / explanatory question. Answer lives in dense prose:
               MD&A, business overview, footnotes, or qualitative sections.
               Example: "How does Apple generate revenue from services?"
               Strategy: generate a hypothetical 10-K passage → embed → search.

  decompose — Multi-hop, comparative, formula-based, or conditional. Requires
               retrieving from multiple sections or computing across two+ values.
               Example: "Did Apple's gross margin improve from FY2021 to FY2022?"
               Example: "What is Apple's FY2022 ROA?" (needs net income + assets)
               Strategy: split into sub-questions, retrieve each, merge.

  stepback  — "Why" / risk / strategic reasoning. Answer needs broad background
               context before a specific data point can be interpreted.
               Example: "Why is Apple exposed to supply chain risk?"
               Strategy: abstract to a general question, retrieve broad context
               + specific question.
"""

from typing import Literal

from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from rag_hub.config.settings import GEMINI_LLM_MODEL
from rag_hub.config.vertex_ai import init_vertex_ai

load_dotenv()
init_vertex_ai()

Route = Literal["direct", "hyde", "decompose", "stepback"]

# ---------------------------------------------------------------------------
# LLM classifier prompt
#
# Few-shot examples chosen to cover the hardest disambiguation cases:
#   - "What is X?" where X is a ratio (decompose) vs. prose (hyde) vs. fact (direct)
#   - Analyst preambles ("As an investment banker…") that mask question intent
#   - Multi-year averages and formula-defined metrics → decompose
#   - "What drove / why" → stepback
# ---------------------------------------------------------------------------

_CLASSIFY_PROMPT = """\
You are a routing classifier for a financial RAG system over SEC 10-K and 10-Q filings.

Your job: given a question, output EXACTLY ONE WORD — the retrieval strategy that will
best find the answer in the filing corpus.

Strategies:
  direct    — The answer is a single fact (one number, name, date, or list) that can
              be found in one place in the filing. No arithmetic or formula needed.

  hyde      — The answer is qualitative / explanatory prose. The question asks HOW,
              WHAT IS THE NATURE OF, DESCRIBE, or seeks context from MD&A / Risk
              Factors / Business Overview sections.

  decompose — The answer requires EITHER:
              (a) comparing values across multiple years, segments, or companies, OR
              (b) computing a ratio or formula (e.g. ROA = net income / total assets),
                  even if the question only asks for one number, OR
              (c) a multi-part or conditional question ("if X then Y, else Z").
              Multi-year averages (e.g. "FY2017-FY2019 3-year average") always decompose.

  stepback  — The question asks WHY something happened, asks about RISK / EXPOSURE /
              STRATEGY, or needs broad background context before a specific answer
              makes sense. Includes "what drove", "is X able to", "was X able to".

Key disambiguation rules:
  - "What is X?" where X is a defined ratio/formula → ALWAYS decompose (not hyde/direct)
  - "What is X?" where X is a descriptive/qualitative concept → hyde
  - "What is X?" where X is a single named fact (revenue, headcount) → direct
  - Analyst preamble ("As an investment banker…", "Using only the balance sheet…")
    does NOT change the underlying question type — classify the actual question.
  - "Which segment / region had the highest/lowest X?" → direct (single comparison)
  - "Has X reported any legal battles?" → stepback (qualitative, needs context)
  - "What are the major acquisitions in FY2021, FY2022, FY2023?" → decompose (multi-year)

Examples:
  Q: What was Apple's total revenue in FY2022?
  A: direct

  Q: What is Coca-Cola's FY2022 dividend payout ratio (using dividends paid and net income)?
  A: decompose

  Q: What is the FY2017-FY2019 3-year average capex as a % of revenue for Activision Blizzard?
  A: decompose

  Q: What is the FY2017 return on assets for Coca-Cola? ROA is defined as net income / total assets.
  A: decompose

  Q: What is the nature and purpose of AMCOR's restructuring liability?
  A: hyde

  Q: What are the major products and services that AMD sells as of FY2022?
  A: hyde

  Q: What drove the increase in Ulta Beauty's merchandise inventories at end of FY2023?
  A: stepback

  Q: Why did Pepsico raise full year guidance for FY2023?
  A: stepback

  Q: Has Boeing reported any materially important ongoing legal battles from FY2022?
  A: stepback

  Q: Which of JPMorgan's business segments had the lowest net revenue in 2021 Q1?
  A: direct

  Q: Did Pfizer grow its PP&E between FY2020 and FY2021?
  A: decompose

  Q: Does Adobe have an improving operating margin profile as of FY2022? If operating margin
     is not a useful metric for a company like this, state that and explain why.
  A: stepback

  Q: What are three main companies acquired by Pfizer mentioned in this 10K report?
  A: direct

  Q: Using only the information within the balance sheet, how much total assets did Costco have
     as of the end of FY2023?
  A: direct

  Q: We want to calculate a financial metric. Please help us compute it by basing your answers
     on the information provided in the income statement and balance sheet only.
  A: decompose

Now classify:
  Q: {question}
  A:"""


class QuestionRouter:
    """
    Pure-LLM question classifier.

    Uses a single Gemini call with few-shot examples to route every question.
    No keyword rules — the LLM handles structural and semantic disambiguation
    that patterns cannot (e.g. "What is X?" where X is a ratio vs. a fact).

    Usage:
        router = QuestionRouter()
        route = router.route("What was Apple's revenue in 2022?")
        # → "direct"
    """

    def __init__(self, model: str = GEMINI_LLM_MODEL, verbose: bool = True):
        self.verbose = verbose
        self.llm = ChatVertexAI(
            model_name=model,
            temperature=0.0,
        )
        self.chain = (
            ChatPromptTemplate.from_template(_CLASSIFY_PROMPT)
            | self.llm
            | StrOutputParser()
        )

    def route(self, question: str) -> Route:
        """
        Returns one of: "direct", "hyde", "decompose", "stepback".

        Falls back to "direct" if the LLM returns an unexpected value.
        """
        raw = self.chain.invoke({"question": question}).strip().lower()
        # Extract just the first word in case the LLM adds explanation
        first_word = raw.split()[0] if raw else ""
        route = first_word if first_word in ("direct", "hyde", "decompose", "stepback") else "direct"

        if self.verbose:
            print(f"[Router] '{question[:80]}' → {route}")

        return route
