"""
semantic.py
-----------
Semantic chunker using LangChain's SemanticChunker.

SemanticChunker embeds every sentence, then splits at positions where the
cosine similarity between adjacent sentences drops below a threshold — so
chunk boundaries align with topic shifts rather than character counts.

Requires: langchain-experimental
    uv add langchain-experimental
"""

import uuid
from typing import List, Dict

from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker


# ── LangChain Embeddings adapter for GeminiEmbeddingClient ──────────────────

class GeminiLangChainEmbeddings(Embeddings):
    """
    Thin adapter so GeminiEmbeddingClient works as a LangChain Embeddings object.
    SemanticChunker only needs embed_documents; embed_query is included for completeness.
    """

    def __init__(self, gemini_client=None):
        """
        Parameters
        ----------
        gemini_client : GeminiEmbeddingClient instance, or None to auto-create.
        """
        if gemini_client is None:
            from rag_hub.embeddings.gemini_001 import GeminiEmbeddingClient
            gemini_client = GeminiEmbeddingClient()
        self._client = gemini_client

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._client.embed_documents(texts, batch_size=100)

    def embed_query(self, text: str) -> List[float]:
        return self._client.embed_documents([text], batch_size=1)[0]


# ── Main chunker ─────────────────────────────────────────────────────────────

def chunk_pages(
    pages: List[Dict],
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: float = 95.0,
    gemini_client=None,
) -> List[Dict]:
    """
    Semantic chunker.

    Operates page-by-page (same boundary rule as recursive chunker — chunks
    never cross pages) so page metadata stays accurate for FinanceBench eval.

    Parameters
    ----------
    pages : list of page dicts from pdf_loader
    breakpoint_threshold_type   : "percentile" | "standard_deviation" | "interquartile"
    breakpoint_threshold_amount : 95 → only split at the top-5 % similarity drops
    gemini_client : optional pre-built GeminiEmbeddingClient (avoids re-init)

    Returns
    -------
    List of chunk dicts: {id, doc_name, page, text, chunk_idx}
    """
    if not pages:
        return []

    lc_embeddings = GeminiLangChainEmbeddings(gemini_client)

    splitter = SemanticChunker(
        embeddings=lc_embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
    )

    doc_name = pages[0]["doc_name"]
    chunks: List[Dict] = []
    chunk_idx = 0

    for page in pages:
        text     = page.get("text", "")
        page_num = page["page"]

        if not text.strip():
            continue

        # SemanticChunker.split_text returns a list of strings
        try:
            sub_texts = splitter.split_text(text)
        except Exception:
            # Fallback: if page has too few sentences, keep whole page as one chunk
            sub_texts = [text]

        for sub in sub_texts:
            if not sub.strip():
                continue
            chunks.append({
                "id":        str(uuid.uuid4()),
                "doc_name":  doc_name,
                "page":      page_num,
                "text":      sub,
                "chunk_idx": chunk_idx,
            })
            chunk_idx += 1

    return chunks
