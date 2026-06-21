from typing import List, Dict

from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from rag_hub.config.settings import GEMINI_LLM_MODEL
from rag_hub.config.vertex_ai import make_chat_llm

load_dotenv()


class GeminiFlashGenerator:
    """
    RAG generator using Gemini 2.5 Flash via Vertex AI.

    Why Vertex AI instead of Gemini API?
      Gemini API free tier quota exhausted during indexing.
      Vertex AI routes through GCP billing (education credits).

    Why LangChain's ChatVertexAI instead of raw genai.Client?
      LangChain's invoke() integrates automatically with LangSmith tracing
      via LANGCHAIN_TRACING_V2=true. A raw genai.Client call bypasses this
      entirely and would be invisible in LangSmith traces.

    Designed for FinanceBench: short factual answers — numbers, phrases, 1 sentence.
    """

    def __init__(self, model: str = GEMINI_LLM_MODEL):
        self.llm = make_chat_llm(model, temperature=0)

        self.prompt = ChatPromptTemplate.from_template(
            """You are a financial analyst answering questions from SEC filings.

Use ONLY the context below. If the answer is not present, say "I don't know".
Return the answer as concisely as possible (a number, a short phrase, or one sentence).

Context:
{context}

Question: {question}
Answer:"""
        )

        # LangChain chain — each invoke() is a single traced run in LangSmith
        self.chain = self.prompt | self.llm

    # -----------------------------
    # Format retrieved chunks
    # -----------------------------
    def _format_context(self, chunks: List[Dict]) -> str:
        """Converts retrieved chunks into prompt-ready context string."""
        return "\n\n".join(
            f"[{c['doc_name']} p.{c['page']}] {c['text']}"
            for c in chunks
        )

    # -----------------------------
    # Main generation function
    # -----------------------------
    def generate(self, question: str, chunks: List[Dict]) -> str:
        """
        Generates answer using retrieved chunks.
        Traced automatically in LangSmith via LANGCHAIN_TRACING_V2=true.
        """
        context = self._format_context(chunks)
        response = self.chain.invoke({"context": context, "question": question})
        return response.content.strip()
