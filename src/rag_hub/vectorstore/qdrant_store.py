import os
from typing import List, Dict, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    VectorParams,
    PointStruct,
)

from dotenv import load_dotenv

load_dotenv()


class QdrantStore:
    """
    Minimal Qdrant wrapper for RAG pipeline.
    """

    def __init__(self, url: str = None, collection: str = "rag_chunks"):
        self.url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.collection = collection

        self.client = QdrantClient(url=self.url)

    # -----------------------------
    # Collection setup
    # -----------------------------
    def ensure_collection(self, dim: int = 768, force: bool = False):
        """
        Create collection if it doesn't exist.

        Args:
            dim:   Vector dimensionality (must match embedding model).
            force: If True, delete and recreate (use for clean reindex only).
                   If False (default), leave existing collection + vectors intact
                   so incremental runs and checkpointing work correctly.
        """
        existing = [c.name for c in self.client.get_collections().collections]

        if self.collection in existing:
            if force:
                print(f"[INFO] force=True — deleting existing collection '{self.collection}'")
                self.client.delete_collection(self.collection)
            else:
                print(f"[INFO] Collection '{self.collection}' already exists — skipping creation")
                return  # leave existing vectors intact

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=dim,
                distance=Distance.COSINE,
            ),
        )
        print(f"[INFO] Created collection '{self.collection}' (dim={dim})")

    # -----------------------------
    # Upsert chunks + embeddings
    # -----------------------------
    def upsert(self, chunks: List[Dict], vectors: List[List[float]], batch_size: int = 32):
        """
        Inserts chunk embeddings into Qdrant in batches.

        Why batched? Each chunk carries its full text in the payload.
        A single upsert of 100+ long chunks can exceed Qdrant's 32MB
        request limit. batch_size=32 keeps each request well under the cap.

        Payload: doc_name, page, text, chunk_idx
        """
        points = [
            PointStruct(
                id=chunk["id"],
                vector=vec,
                payload={
                    "doc_name": chunk["doc_name"],
                    "page": chunk["page"],
                    "text": chunk["text"],
                    "chunk_idx": chunk["chunk_idx"],
                },
            )
            for chunk, vec in zip(chunks, vectors)
        ]

        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection,
                points=points[i : i + batch_size],
            )

    # -----------------------------
    # Vector search
    # -----------------------------
    def search(self, query_vec: List[float], k: int = 5, query_filter=None):
        """
        Search top-k similar chunks.
        """
        return self.client.query_points(
            collection_name=self.collection,
            query=query_vec,
            limit=k,query_filter=query_filter,
        ).points
        return self.client.search(
            collection_name=self.collection,
            query_vector=query_vec,
            limit=k,
            query_filter=query_filter,
        )
    

    def delete_by_doc_name(self, doc_name: str):
        """
        Deletes all vectors (and payloads) for a given document.
        """

        print(f"[INFO] Deleting all entries for doc: {doc_name}")

        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="doc_name",
                        match=MatchValue(value=doc_name)
                    )
                ]
            )
        )

        print(f"[DONE] Deleted entries for: {doc_name}")