"""
Corrective generation loop: generates a cited answer, scores it, checks for
hallucinations, and re-retrieves with expanded context if needed.
"""
import time
from typing import List, Dict, Optional, Callable

from rag_hub.generation.schemas import Answer
from rag_hub.generation.citation_generator import CitationAwareGenerator
from rag_hub.generation.self_rag import SelfRAGScorer
from rag_hub.generation.hallucination_detector import HallucinationDetector


class CorrectiveGenerator:
    """
    Corrective generation loop (max 2 iterations by default):
      1. Generate cited answer
      2. Check hallucination rate via NLI
      3. If rate > threshold and iterations remain: expand retrieval, regenerate
      4. Score the chosen answer with Self-RAG (LLM judge) — once, after the loop
      5. Return answer with lowest hallucination rate seen

    Self-RAG scoring runs once on the final answer (not every iteration): it only
    populates the reported confidence and does not drive the corrective decision,
    so scoring discarded intermediate answers wastes an LLM call per iteration.
    """

    def __init__(
        self,
        citation_gen: CitationAwareGenerator,
        self_rag: SelfRAGScorer,
        detector: HallucinationDetector,
        retriever=None,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        max_iterations: int = 2,
        hallucination_threshold: float = 0.25,
        initial_top_k: int = 5,
        timing: bool = False,
    ):
        self.citation_gen = citation_gen
        self.self_rag = self_rag
        self.detector = detector
        self.retriever = retriever
        self.embed_fn = embed_fn
        self.max_iterations = max_iterations
        self.hallucination_threshold = hallucination_threshold
        self.initial_top_k = initial_top_k
        self.timing = timing

    def generate(self, question: str, docs: List[Dict]) -> Answer:
        best: Optional[Answer] = None
        best_docs = docs
        current_docs = docs

        for iteration in range(1, self.max_iterations + 1):
            t0 = time.perf_counter()
            answer = self.citation_gen.generate(question, current_docs)
            t_cite = time.perf_counter()
            rate, _labels = self.detector.check(answer)
            t_nli = time.perf_counter()
            answer.hallucination_rate = rate
            answer.generation_iterations = iteration

            if self.timing:
                print(
                    f"    [gen iter {iteration}] cite={t_cite-t0:.2f}s  "
                    f"nli={t_nli-t_cite:.2f}s  h_rate={rate:.2f}"
                )

            if best is None or rate < best.hallucination_rate:
                best = answer
                best_docs = current_docs

            if rate <= self.hallucination_threshold:
                break

            if iteration < self.max_iterations and self.retriever and self.embed_fn:
                expanded_k = self.initial_top_k * 2
                query_vec = self.embed_fn(question)
                current_docs = self.retriever.search(question, query_vec, top_k=expanded_k)

        # Self-RAG scoring runs once, on the chosen answer (see class docstring).
        t_sr = time.perf_counter()
        best.confidence = self.self_rag.score(question, best, best_docs)
        if self.timing:
            print(f"    [gen final] self_rag={time.perf_counter()-t_sr:.2f}s")

        return best
