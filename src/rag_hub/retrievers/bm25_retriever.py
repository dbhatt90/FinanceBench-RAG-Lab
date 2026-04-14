import re
import string
from typing import List, Dict, Optional

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    """
    Lowercase, strip punctuation, split on whitespace.
    Consistent tokenization must be applied to both corpus and queries.
    """
    if not text:
        return []
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


class BM25Retriever:
    """
    In-memory BM25 retriever built over chunks scrolled from Qdrant.

    Usage:
        retriever = BM25Retriever()
        retriever.build_from_qdrant(store)          # one-time corpus load
        results = retriever.search("revenue 2022", k=5)

    Return format mirrors QdrantStore.search payloads so the eval loop
    can swap retrievers without any changes:
        [{"doc_name": ..., "page": ..., "text": ..., "chunk_idx": ...}, ...]
    """

    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._corpus_chunks: List[Dict] = []   # ordered list of payload dicts
        self._corpus_tokens: List[List[str]] = []

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_from_qdrant(self, store, scroll_limit: int = 1000) -> None:
        """
        Scroll all points from a QdrantStore instance and build the BM25 index.

        Args:
            store:        QdrantStore instance (already connected).
            scroll_limit: Page size for each Qdrant scroll call (max 1000).
        """
        print("[BM25] Scrolling corpus from Qdrant …")

        chunks: List[Dict] = []
        offset = None   # None → start from beginning

        while True:
            results, next_offset = store.client.scroll(
                collection_name=store.collection,
                limit=scroll_limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,   # we only need text, not vectors
            )

            for point in results:
                chunks.append(point.payload)

            if next_offset is None:
                break
            offset = next_offset

        if not chunks:
            raise RuntimeError(
                f"[BM25] No chunks found in collection '{store.collection}'. "
                "Run build_index first."
            )

        print(f"[BM25] Loaded {len(chunks)} chunks from Qdrant.")

        self._corpus_chunks = chunks
        self._corpus_tokens = [_tokenize(c.get("text", "")) for c in chunks]
        self._bm25 = BM25Okapi(self._corpus_tokens)

        print("[BM25] Index built.")

    def build_from_chunks(self, chunks: List[Dict]) -> None:
        """
        Build directly from a list of payload dicts (useful for tests).

        Args:
            chunks: List of dicts, each must have at least a 'text' key.
        """
        if not chunks:
            raise ValueError("[BM25] Cannot build from empty chunk list.")

        self._corpus_chunks = chunks
        self._corpus_tokens = [_tokenize(c.get("text", "")) for c in chunks]
        self._bm25 = BM25Okapi(self._corpus_tokens)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Return top-k chunks ranked by BM25 score.

        Args:
            query: Raw query string (tokenized internally).
            k:     Number of results to return.

        Returns:
            List of payload dicts ordered by descending BM25 score.
            Each dict has: doc_name, page, text, chunk_idx, bm25_score.
        """
        if self._bm25 is None:
            raise RuntimeError(
                "[BM25] Index not built. Call build_from_qdrant() or build_from_chunks() first."
            )

        query_tokens = _tokenize(query)

        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)  # shape: (corpus_size,)

        # argsort descending, take top-k
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results = []
        for idx in top_indices:
            chunk = dict(self._corpus_chunks[idx])   # copy to avoid mutation
            chunk["bm25_score"] = float(scores[idx])
            results.append(chunk)

        return results

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def corpus_size(self) -> int:
        return len(self._corpus_chunks)

    def is_built(self) -> bool:
        return self._bm25 is not None
