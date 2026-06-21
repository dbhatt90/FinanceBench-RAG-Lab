"""
Day 7 RAG pipeline: direct retrieval (no LangGraph router, no CRAG).

Flow: embed query → HybridRRFRetriever → BGEReranker → CorrectiveGenerator → Answer

CRAG is omitted: FinanceBench questions are all answered by the indexed corpus so
CRAG always returns high confidence, adding one LLM call per question for no benefit.
"""
import time
from typing import Dict

from rag_hub.embeddings.gemini_001 import GeminiEmbeddingClient
from rag_hub.vectorstore.qdrant_store import QdrantStore
from rag_hub.retrievers.bm25_retriever import BM25Retriever
from rag_hub.retrievers.hybrid_retriever import HybridRRFRetriever
from rag_hub.rerankers.bge_reranker import BGEReranker
from rag_hub.generation.schemas import Answer
from rag_hub.generation.citation_generator import CitationAwareGenerator
from rag_hub.generation.self_rag import SelfRAGScorer
from rag_hub.generation.hallucination_detector import HallucinationDetector
from rag_hub.generation.corrective_generator import CorrectiveGenerator


class Day7Pipeline:
    """
    Self-contained Day 7 pipeline. Bypasses the LangGraph router.

    Args:
        store: Qdrant vector store (already populated).
        bm25: BM25Retriever (already built).
        embedder: GeminiEmbeddingClient for query embedding.
        retrieval_top_k: Candidate pool for reranker (default 10).
        rerank_top_k: Docs passed to generator after reranking (default 5).
        hallucination_threshold: Triggers re-retrieval + regeneration.
    """

    def __init__(
        self,
        store: QdrantStore,
        bm25: BM25Retriever,
        embedder: GeminiEmbeddingClient,
        retrieval_top_k: int = 10,
        rerank_top_k: int = 5,
        hallucination_threshold: float = 0.25,
        enable_correction: bool = False,
        timing: bool = True,
    ):
        self.embedder = embedder
        self.timing = timing
        self._retriever = HybridRRFRetriever(store=store, bm25=bm25)
        self._reranker = BGEReranker()
        # The corrective re-retrieve/regenerate pass roughly doubles per-question
        # cost and, on FinanceBench, fires on most questions because NLI rarely
        # entails numeric reasoning. Off by default; enable explicitly to study it.
        max_iterations = 2 if enable_correction else 1
        self._corrective = CorrectiveGenerator(
            citation_gen=CitationAwareGenerator(),
            self_rag=SelfRAGScorer(),
            detector=HallucinationDetector(threshold=hallucination_threshold),
            retriever=self._retriever,
            embed_fn=self.embedder.embed_query,
            max_iterations=max_iterations,
            hallucination_threshold=hallucination_threshold,
            initial_top_k=rerank_top_k,
            timing=timing,
        )
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_k = rerank_top_k

    def run(self, question: str) -> Dict:
        """Returns dict: question, answer (Answer), docs."""
        t0 = time.perf_counter()
        query_vec = self.embedder.embed_query(question)
        t_embed = time.perf_counter()
        docs = self._retriever.search(question, query_vec, top_k=self.retrieval_top_k)
        t_retrieve = time.perf_counter()
        reranked = self._reranker.rerank(question, docs, top_k=self.rerank_top_k)
        t_rerank = time.perf_counter()
        answer: Answer = self._corrective.generate(question, reranked)
        t_gen = time.perf_counter()

        if self.timing:
            print(
                f"  [timing] embed={t_embed-t0:.2f}s  retrieve={t_retrieve-t_embed:.2f}s  "
                f"rerank={t_rerank-t_retrieve:.2f}s  generate={t_gen-t_rerank:.2f}s  "
                f"total={t_gen-t0:.2f}s"
            )
        return {"question": question, "answer": answer, "docs": reranked}
