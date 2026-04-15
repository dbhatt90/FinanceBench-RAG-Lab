"""
index_builder.py
----------------
Reusable pipeline: pages → chunker → embeddings → Qdrant.

Replaces build_index.py + build_index_extended.py.

Usage
-----
from rag_hub.indexing.index_builder import IndexBuilder
from rag_hub.chunking.recursive import chunk_pages

builder = IndexBuilder(
    collection_name="fb_recursive",
    eval_set_path="data/eval/smoke_20.jsonl",
)
builder.run(chunker=chunk_pages)
"""

import json
import os
from pathlib import Path
from typing import Callable, List, Dict, Optional

from tqdm import tqdm

from rag_hub.loaders.pdf_loader import load_pdf, load_cached_pages, save_cached_pages
from rag_hub.embeddings.gemini_001 import GeminiEmbeddingClient
from rag_hub.vectorstore.qdrant_store import QdrantStore


# ── defaults ────────────────────────────────────────────────────────────────
PDF_DIR = "data/raw/financebench/pdfs"
PAGES_CACHE_DIR = "data/processed/pages"
CHECKPOINTS_DIR = "data/eval/checkpoints"


class IndexBuilder:
    """
    Orchestrates: pages → chunker → embed → Qdrant upsert.

    Parameters
    ----------
    collection_name : str
        Qdrant collection to write into (one per chunking strategy).
    eval_set_path : str
        Path to a .jsonl file whose 'doc_name' fields define which docs to index.
    pdf_dir : str
        Root directory that contains raw PDF files.
    pages_cache_dir : str
        Directory for pre-parsed page JSON caches (avoids re-running pdfplumber).
    embed_dim : int
        Dimensionality expected by the Qdrant collection (must match embedder).
    batch_size : int
        How many chunk texts to send to the embedding API per request.
    """

    def __init__(
        self,
        collection_name: str,
        eval_set_path: str,
        pdf_dir: str = PDF_DIR,
        pages_cache_dir: str = PAGES_CACHE_DIR,
        embed_dim: int = 768,
        batch_size: int = 100,
    ):
        self.collection_name = collection_name
        self.eval_set_path = eval_set_path
        self.pdf_dir = pdf_dir
        self.pages_cache_dir = pages_cache_dir
        self.embed_dim = embed_dim
        self.batch_size = batch_size

        # per-collection checkpoint so parallel runs don't collide
        self.checkpoint_path = Path(CHECKPOINTS_DIR) / f"{collection_name}.json"

    # ── public API ────────────────────────────────────────────────────────────

    def run(
        self,
        chunker: Callable[[List[Dict]], List[Dict]],
        force: bool = False,
    ) -> Dict:
        """
        Main entry point.

        Parameters
        ----------
        chunker : callable
            Any function with signature  pages -> chunks
            where each chunk is a dict with at least {"id", "doc_name", "page", "text"}.
        force : bool
            If True, wipe Qdrant collection + checkpoint and start fresh.

        Returns
        -------
        dict  Summary stats: n_docs, n_pages, n_chunks, n_vectors.
        """
        doc_names = self._load_doc_names()
        checkpoint = self._load_checkpoint(force)
        completed = set(checkpoint.get("completed_docs", []))

        embedder = GeminiEmbeddingClient()
        store = QdrantStore(collection=self.collection_name)
        store.ensure_collection(dim=self.embed_dim, force=force)

        stats = {"n_docs": 0, "n_pages": 0, "n_chunks": 0, "n_vectors": 0}

        for doc in tqdm(doc_names, desc=f"[{self.collection_name}] indexing"):
            if doc in completed:
                print(f"  [SKIP] {doc}")
                continue

            pages = self._load_pages(doc)
            if pages is None:
                continue

            try:
                chunks = chunker(pages)
                if not chunks:
                    print(f"  [WARN] No chunks produced for {doc}")
                    completed.add(doc)
                    self._save_checkpoint(completed)
                    continue

                texts = [c["text"] for c in chunks]
                vectors = embedder.embed_documents(texts, batch_size=self.batch_size)
                store.upsert(chunks, vectors)

                stats["n_docs"] += 1
                stats["n_pages"] += len(pages)
                stats["n_chunks"] += len(chunks)
                stats["n_vectors"] += len(vectors)

                completed.add(doc)
                self._save_checkpoint(completed)
                print(f"  [DONE] {doc}  ({len(chunks)} chunks)")

            except Exception as exc:
                self._save_checkpoint(completed)  # save progress before raising
                raise RuntimeError(f"Failed on {doc}") from exc

        self._print_summary(stats)
        return stats

    # ── helpers ───────────────────────────────────────────────────────────────

    def _load_doc_names(self) -> List[str]:
        with open(self.eval_set_path, encoding="utf-8") as f:
            questions = [json.loads(line) for line in f]
        doc_names = sorted(set(q["doc_name"] for q in questions))
        print(f"[INFO] Eval set: {self.eval_set_path}  →  {len(doc_names)} unique docs")
        return doc_names

    def _load_pages(self, doc_name: str) -> Optional[List[Dict]]:
        """Load from cache if available, otherwise parse PDF and cache."""
        pages = load_cached_pages(doc_name, self.pages_cache_dir)
        if pages is not None:
            return pages

        pdf_path = os.path.join(self.pdf_dir, f"{doc_name}.pdf")
        if not os.path.exists(pdf_path):
            print(f"  [WARN] No PDF found: {pdf_path}")
            return None

        pages = load_pdf(pdf_path)
        save_cached_pages(doc_name, pages, self.pages_cache_dir)
        print(f"  [CACHE] Saved pages for {doc_name}")
        return pages

    def _load_checkpoint(self, force: bool) -> Dict:
        if force:
            print(f"[INFO] force=True — resetting checkpoint for '{self.collection_name}'")
            self._save_checkpoint(set())
            return {"completed_docs": []}

        if self.checkpoint_path.exists():
            return json.loads(self.checkpoint_path.read_text())
        return {"completed_docs": []}

    def _save_checkpoint(self, completed: set):
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.write_text(
            json.dumps({"completed_docs": list(completed)}, indent=2)
        )

    def _print_summary(self, stats: Dict):
        print("\n" + "=" * 40)
        print("INDEX BUILD COMPLETE")
        print("=" * 40)
        print(f"  Collection : {self.collection_name}")
        print(f"  #Docs      : {stats['n_docs']}")
        print(f"  #Pages     : {stats['n_pages']}")
        print(f"  #Chunks    : {stats['n_chunks']}")
        print(f"  #Vectors   : {stats['n_vectors']}")
        print("=" * 40)
