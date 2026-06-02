"""
Day 1 — Index FinanceBench PDFs into Qdrant.

Loads PDFs from the FinanceBench open-source set, applies recursive chunking
(chunk_size=1000, overlap=200), embeds with Gemini embedding-001, and upserts
into the Qdrant collection. Supports resumable indexing via checkpoints.

Usage:
    python scripts/day1_index.py [--force-reindex]
"""
import json
import argparse

from tqdm import tqdm

from rag_hub.config.settings import (
    PDF_DIR, PAGES_CACHE_DIR, SMOKE_20_PATH, SMOKE_50_PATH,
    QDRANT_URL, COLLECTION_NAME, EMBEDDING_DIM,
)
from rag_hub.eval.financebench import load_questions, sample_smoke_set
from rag_hub.loaders.pdf_loader import load_pdf, load_cached_pages, save_cached_pages
from rag_hub.chunking.recursive import chunk_pages
from rag_hub.embeddings.gemini_001 import GeminiEmbeddingClient
from rag_hub.vectorstore.qdrant_store import QdrantStore

# checkpoint_indexing.py lives in scripts/utils/ after refactor
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))
from checkpoint_indexing import load_checkpoint, save_checkpoint

_FINANCEBENCH_JSONL = "data/raw/financebench/data/financebench_open_source.jsonl"


def _save_jsonl(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Wipe the Qdrant collection and checkpoint, start from scratch.",
    )
    args = parser.parse_args()

    # Step 1: Build / reload the smoke-50 eval set
    questions = load_questions(_FINANCEBENCH_JSONL)

    with open(SMOKE_20_PATH) as f:
        smoke_questions = [json.loads(line) for line in f]

    smoke_ids = set(q["financebench_id"] for q in smoke_questions)
    remaining = [q for q in questions if q["financebench_id"] not in smoke_ids]
    new_questions = sample_smoke_set(remaining, n=30, seed=42)
    smoke_50_questions = smoke_questions + new_questions
    _save_jsonl(str(SMOKE_50_PATH), smoke_50_questions)
    print(f"[INFO] Saved smoke-50 → {SMOKE_50_PATH}")

    # Step 2: Collect unique doc names
    doc_names = sorted(set(q["doc_name"] for q in smoke_50_questions))
    print(f"[INFO] Unique docs: {len(doc_names)}")

    # Load / reset checkpoint
    if args.force_reindex:
        print("[INFO] --force-reindex: clearing checkpoint and Qdrant collection")
        checkpoint = {"completed_docs": []}
        save_checkpoint(checkpoint)
    else:
        checkpoint = load_checkpoint()

    completed_docs = set(checkpoint.get("completed_docs", []))
    print(f"[INFO] Already completed docs: {len(completed_docs)}")

    # Init systems
    embedder = GeminiEmbeddingClient()
    store = QdrantStore(url=QDRANT_URL, collection=COLLECTION_NAME)
    store.ensure_collection(dim=EMBEDDING_DIM, force=args.force_reindex)

    # Step 3: Process docs (resumable)
    total_pages = total_chunks = total_vectors = 0

    for doc in tqdm(doc_names, desc="Processing PDFs"):
        if doc in completed_docs:
            print(f"[SKIP] {doc}")
            continue

        pdf_path = os.path.join(str(PDF_DIR), f"{doc}.pdf")
        if not os.path.exists(pdf_path):
            print(f"[WARN] Missing PDF: {pdf_path}")
            continue

        try:
            pages = load_cached_pages(doc, str(PAGES_CACHE_DIR))
            if pages is None:
                pages = load_pdf(pdf_path)
                save_cached_pages(doc, pages, str(PAGES_CACHE_DIR))

            chunks = chunk_pages(pages)
            texts = [c["text"] for c in chunks]
            vectors = embedder.embed_documents(texts, batch_size=100)
            store.upsert(chunks, vectors)

            total_pages += len(pages)
            total_chunks += len(chunks)
            total_vectors += len(vectors)

            completed_docs.add(doc)
            checkpoint["completed_docs"] = list(completed_docs)
            save_checkpoint(checkpoint)
            print(f"[DONE] {doc}")

        except Exception as e:
            print(f"[ERROR] Failed on {doc}: {e}")
            checkpoint["completed_docs"] = list(completed_docs)
            save_checkpoint(checkpoint)
            raise

    print(f"\n{'='*30}")
    print("INDEX BUILD COMPLETE")
    print(f"{'='*30}")
    print(f"#Docs      : {len(doc_names)}")
    print(f"#Pages     : {total_pages}")
    print(f"#Chunks    : {total_chunks}")
    print(f"#Vectors   : {total_vectors}")
    print(f"Collection : {COLLECTION_NAME}")


if __name__ == "__main__":
    main()
