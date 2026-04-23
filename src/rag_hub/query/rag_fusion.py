from typing import List, Dict, Tuple

from rag_hub.eval.retrieval_metrics import rrf_score
from rag_hub.query.multi_query import MultiQueryTransform
from rag_hub.vectorstore.qdrant_store import QdrantStore
from rag_hub.embeddings.gemini_001 import GeminiEmbeddingClient


class RAGFusionRetriever:
    """
    RAG-Fusion = multi-query expansion + per-query dense retrieval + RRF fusion.

    Steps:
      1. Expand the original query into N variants (MultiQueryTransform).
      2. Embed each variant and retrieve a candidate pool from Qdrant.
      3. Fuse all ranked lists with RRF, return top-k.

    Why RRF instead of score averaging?
      RRF is rank-based, so it's robust to score scale differences across
      queries. A chunk that ranks #2 for three different query variants gets
      a high fused score regardless of whether the cosine similarities were
      0.91 or 0.73.
    """

    def __init__(
        self,
        store: QdrantStore,
        embedder: GeminiEmbeddingClient,
        n_queries: int = 3,
        rrf_k: int = 60,
    ):
        self.store = store
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.expander = MultiQueryTransform(n=n_queries)

    @staticmethod
    def _chunk_id(payload: Dict) -> str:
        return f"{payload['doc_name']}_p{payload['page']}"

    def search(
        self, query: str, top_k: int = 5
    ) -> Tuple[List[Dict], List[str]]:
        """
        Args:
            query:  Original user question.
            top_k:  Number of final results to return.

        Returns:
            (fused_payloads, expanded_queries)
            - fused_payloads: list of chunk dicts sorted by RRF score,
              each with an added 'rrf_score' key.
            - expanded_queries: the full list of queries used (original + variants),
              useful for logging / eval inspection.
        """
        candidate_k = top_k * 5   # wider pool for better fusion coverage

        expanded_queries = self.expander.transform(query)

        # ── Retrieve for each query variant ──────────────────────────────
        all_ranked_ids: List[List[str]] = []
        payload_map: Dict[str, Dict] = {}

        for q in expanded_queries:
            vec = self.embedder.embed_query(q)
            points = self.store.search(vec, k=candidate_k)
            payloads = [p.payload for p in points]

            ranked_ids = [self._chunk_id(p) for p in payloads]
            all_ranked_ids.append(ranked_ids)

            for payload in payloads:
                cid = self._chunk_id(payload)
                payload_map[cid] = payload   # last write wins (same chunk)

        # ── RRF fusion ────────────────────────────────────────────────────
        fused_scores: Dict[str, float] = rrf_score(all_ranked_ids, k=self.rrf_k)

        fused: List[Dict] = []
        for cid, score in fused_scores.items():
            if cid in payload_map:
                chunk = dict(payload_map[cid])
                chunk["rrf_score"] = score
                fused.append(chunk)

        fused.sort(key=lambda x: x["rrf_score"], reverse=True)

        return fused[:top_k], expanded_queries
