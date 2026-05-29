"""
Day 7 RAG pipeline: direct retrieval (no LangGraph router).

Flow: embed query → HybridRRFRetriever → BGEReranker → CRAGEvaluator
      → CorrectiveGenerator → Answer
"""
from typing import Dict

from rag_hub.embeddings.gemini_001 import GeminiEmbeddingClient
from rag_hub.vectorstore.qdrant_store import QdrantStore
from rag_hub.retrievers.bm25_retriever import BM25Retriever
from rag_hub.retrievers.hybrid_retriever import HybridRRFRetriever
from rag_hub.rerankers.bge_reranker import BGEReranker
from rag_hub.crag.evaluator import CRAGEvaluator
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
        crag_threshold: float = 0.5,
        hallucination_threshold: float = 0.25,
    ):
        self.embedder = embedder
        self._retriever = HybridRRFRetriever(store=store, bm25=bm25)
        self._reranker = BGEReranker()
        self._crag = CRAGEvaluator(threshold=crag_threshold)
        self._corrective = CorrectiveGenerator(
            citation_gen=CitationAwareGenerator(),
            self_rag=SelfRAGScorer(),
            detector=HallucinationDetector(threshold=hallucination_threshold),
            retriever=self._retriever,
            embed_fn=self.embedder.embed_query,
            hallucination_threshold=hallucination_threshold,
            initial_top_k=rerank_top_k,
        )
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_k = rerank_top_k

    def run(self, question: str) -> Dict:
        """Returns dict: question, answer (Answer), docs, crag_confidence, crag_labels."""
        query_vec = self.embedder.embed_query(question)
        docs = self._retriever.search(question, query_vec, top_k=self.retrieval_top_k)
        reranked = self._reranker.rerank(question, docs, top_k=self.rerank_top_k)
        crag_confidence, crag_labels = self._crag.evaluate(question, reranked)
        answer: Answer = self._corrective.generate(question, reranked)
        return {
            "question": question,
            "answer": answer,
            "docs": reranked,
            "crag_confidence": crag_confidence,
            "crag_labels": crag_labels,
        }
