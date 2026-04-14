from typing import List, Dict

from rag_hub.eval.retrieval_metrics import rrf_score
from rag_hub.vectorstore.qdrant_store import QdrantStore
from rag_hub.retrievers.bm25_retriever import BM25Retriever


class HybridRRFRetriever:
    """
    Combines BM25 + Dense (Qdrant) retrieval using Reciprocal Rank Fusion.

    Both retrievers query independently with an expanded candidate pool
    (top_k * 5). Their ranked lists are fused via RRF, then the top_k
    final results are returned as plain payload dicts — same format as
    QdrantStore.search() payloads.
    """

    def __init__(self, store: QdrantStore, bm25: BM25Retriever, rrf_k: int = 60):
        """
        Args:
            store: QdrantStore instance for dense retrieval.
            bm25:  BM25Retriever instance (must already be built).
            rrf_k: RRF constant (default 60 — standard from the paper).
        """
        self.store = store
        self.bm25 = bm25
        self.rrf_k = rrf_k

    # ------------------------------------------------------------------
    # Stable chunk identifier  (doc + page + chunk_idx)
    # ------------------------------------------------------------------
    @staticmethod
    def _chunk_id(payload: Dict) -> str:
        return f"{payload['doc_name']}_p{payload['page']}_c{payload.get('chunk_idx', 0)}"

    # ------------------------------------------------------------------
    # Main search
    # ------------------------------------------------------------------
    def search(self, query: str, query_vec: List[float], top_k: int = 5) -> List[Dict]:
        """
        Args:
            query:     Raw query string — used by BM25.
            query_vec: Dense embedding vector — used by Qdrant.
            top_k:     Number of final results to return.

        Returns:
            List of payload dicts sorted by descending RRF score.
            Each dict has: doc_name, page, text, chunk_idx, rrf_score.
        """
        candidate_k = top_k * 5   # wider pool improves fusion quality

        # ------------------------------------------------------------------
        # 1. Dense retrieval  →  unwrap ScoredPoint objects into plain dicts
        # ------------------------------------------------------------------
        dense_points = self.store.search(query_vec, k=candidate_k)
        dense_payloads: List[Dict] = [p.payload for p in dense_points]

        # ------------------------------------------------------------------
        # 2. BM25 retrieval  →  already returns plain dicts
        # ------------------------------------------------------------------
        bm25_payloads: List[Dict] = self.bm25.search(query, k=candidate_k)

        # ------------------------------------------------------------------
        # 3. Build chunk-ID ranked lists for rrf_score()
        #    rrf_score expects List[List[str]]
        # ------------------------------------------------------------------
        dense_ids: List[str] = [self._chunk_id(p) for p in dense_payloads]
        bm25_ids:  List[str] = [self._chunk_id(p) for p in bm25_payloads]

        fused_scores: Dict[str, float] = rrf_score([dense_ids, bm25_ids], k=self.rrf_k)

        # ------------------------------------------------------------------
        # 4. Build lookup: chunk_id → payload dict
        #    Dense payloads take priority if the same chunk appears in both
        # ------------------------------------------------------------------
        item_map: Dict[str, Dict] = {}
        for payload in bm25_payloads + dense_payloads:   # dense overwrites BM25
            cid = self._chunk_id(payload)
            item_map[cid] = payload

        # ------------------------------------------------------------------
        # 5. Attach RRF score, sort, return top_k
        # ------------------------------------------------------------------
        fused: List[Dict] = []
        for cid, score in fused_scores.items():
            chunk = dict(item_map[cid])
            chunk["rrf_score"] = score
            fused.append(chunk)

        fused.sort(key=lambda x: x["rrf_score"], reverse=True)

        return fused[:top_k]
