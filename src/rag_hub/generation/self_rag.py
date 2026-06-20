"""
Self-RAG scorer: LLM-as-judge that evaluates whether an answer is grounded
in the retrieved context. Returns a confidence score [0, 1].
"""
from typing import List, Dict

from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from rag_hub.config.settings import GEMINI_LLM_MODEL
from rag_hub.config.vertex_ai import init_vertex_ai
from rag_hub.generation.schemas import Answer

load_dotenv()
init_vertex_ai()


class _ScoreResult(BaseModel):
    is_relevant: bool = Field(..., description="Does the answer address the question?")
    is_supported: bool = Field(..., description="Are all claims supported by the context?")
    is_useful: bool = Field(..., description="Is the answer specific and non-vague?")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall support score")
    reasoning: str = Field(..., description="One-sentence explanation")


_PROMPT = ChatPromptTemplate.from_template(
    """You are evaluating a RAG system's answer against the retrieved context.

Question: {question}

Answer: {answer}

Context (retrieved passages):
{context}

Evaluate the answer on three criteria:
1. is_relevant: Does it directly address the question?
2. is_supported: Are ALL factual claims backed by the context passages?
3. is_useful: Is it specific and actionable (not "I don't know" or overly vague)?
4. confidence: Float 0.0-1.0. 1.0 = every claim is explicitly in the context. 0.0 = no claim is supported.
5. reasoning: One sentence explaining the score."""
)


class SelfRAGScorer:
    """LLM-as-judge: scores answer faithfulness to retrieved context. Returns confidence [0, 1]."""

    def __init__(self, model: str = GEMINI_LLM_MODEL):
        llm = ChatVertexAI(
            model_name=model,
            temperature=0,
        )
        self._chain = _PROMPT | llm.with_structured_output(_ScoreResult)

    def score(self, question: str, answer: Answer, chunks: List[Dict]) -> float:
        """Returns 0.5 on any error (neutral fallback)."""
        context = "\n\n".join(
            f"[{c['doc_name']} p.{c['page']}] {c['text']}" for c in chunks[:5]
        )
        try:
            result: _ScoreResult = self._chain.invoke(
                {"question": question, "answer": answer.text, "context": context}
            )
            return result.confidence
        except Exception as e:
            print(f"[SelfRAG] scoring failed ({e}), defaulting to 0.5")
            return 0.5
