from typing import List, Dict

from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from rag_hub.config.settings import GEMINI_LLM_MODEL
from rag_hub.config.vertex_ai import init_vertex_ai
from rag_hub.generation.schemas import Answer, CitedAnswer

load_dotenv()
init_vertex_ai()

_PROMPT = ChatPromptTemplate.from_template(
    """You are a financial analyst answering questions from SEC filings.

Use ONLY the numbered context passages below. For each claim in your answer,
cite the passage by including its doc_id, page number, and a verbatim quote
(max 200 characters) from that passage.

If the answer is not present in the context, set text to "I don't know" and
return an empty citations list.

Context:
{context}

Question: {question}"""
)


class CitationAwareGenerator:
    """
    Generates structured answers with citation attribution using Gemini's
    structured output (function-calling) to enforce the CitedAnswer schema.

    Falls back to plain text generation if structured output fails.
    """

    def __init__(self, model: str = GEMINI_LLM_MODEL):
        base_llm = ChatVertexAI(
            model_name=model,
            temperature=0,
        )
        self._chain = _PROMPT | base_llm.with_structured_output(CitedAnswer)
        self._fallback_chain = _PROMPT | base_llm

    def _format_context(self, chunks: List[Dict]) -> str:
        lines = []
        for i, c in enumerate(chunks, 1):
            lines.append(
                f"[{i}] doc_id={c['doc_name']}  page={c['page']}\n{c['text']}"
            )
        return "\n\n".join(lines)

    def generate(self, question: str, chunks: List[Dict]) -> Answer:
        """Returns an Answer with text + citations. Falls back to no citations on LLM error."""
        context = self._format_context(chunks)
        try:
            cited: CitedAnswer = self._chain.invoke(
                {"context": context, "question": question}
            )
            return Answer(text=cited.text, citations=cited.citations)
        except Exception as e:
            print(f"[CitationGen] structured output failed ({e}), using fallback")
            response = self._fallback_chain.invoke(
                {"context": context, "question": question}
            )
            text = getattr(response, "content", str(response)).strip()
            return Answer(text=text, citations=[])
