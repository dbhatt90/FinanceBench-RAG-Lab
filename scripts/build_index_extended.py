import os
import json
import argparse
from tqdm import tqdm

from rag_hub.eval.financebench import load_questions, sample_smoke_set
from rag_hub.loaders.pdf_loader import load_pdf, load_cached_pages, save_cached_pages
from rag_hub.chunking.recursive import chunk_pages
from rag_hub.embeddings.gemini_001 import GeminiEmbeddingClient
from rag_hub.vectorstore.qdrant_store import QdrantStore

from checkpoint_indexing import load_checkpoint, save_checkpoint


PDF_DIR = "data/raw/financebench/pdfs"
PAGES_CACHE_DIR = "data/processed/pages"
SMOKE_20_PATH = "data/eval/smoke_20.jsonl"
SMOKE_50_PATH = "data/eval/smoke_50.jsonl"
# CHECKPOINT_PATH = "data/eval/index_checkpoint.json"


def save_jsonl(path, data):
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

    # -----------------------------
    # Step 1: Load + sample questions
    # -----------------------------
    questions = load_questions(
        "data/raw/financebench/data/financebench_open_source.jsonl"
    )

    # smoke_questions = sample_smoke_set(questions, n=20, seed=42)
    with open(SMOKE_20_PATH) as f:
        smoke_questions = [json.loads(line) for line in f]

    smoke_ids = set(q["financebench_id"] for q in smoke_questions)

    remaining = [
        q for q in questions
        if q["financebench_id"] not in smoke_ids
    ]

    new_questions = sample_smoke_set(remaining, n=30, seed=42)

    smoke_50_questions = smoke_questions + new_questions

    save_jsonl(SMOKE_50_PATH, smoke_50_questions)
    print(f"\n[INFO] Saved smoke set → {SMOKE_50_PATH}")

    # -----------------------------
    # Step 2: Collect docs
    # -----------------------------
    doc_names = sorted(set(q["doc_name"] for q in smoke_50_questions))
    print(f"\n[INFO] Unique docs: {len(doc_names)}")

    # -----------------------------
    # Load checkpoint (or reset if --force-reindex)
    # -----------------------------
    if args.force_reindex:
        print("[INFO] --force-reindex: clearing checkpoint and Qdrant collection")
        checkpoint = {"completed_docs": []}
        save_checkpoint(checkpoint)
    else:
        checkpoint = load_checkpoint()

    completed_docs = set(checkpoint.get("completed_docs", []))
    print(f"[INFO] Already completed docs: {len(completed_docs)}")

    # -----------------------------
    # Init systems
    # -----------------------------
    embedder = GeminiEmbeddingClient()
    store = QdrantStore(collection="financebench_v1")

    # force=True only when explicitly requested — otherwise preserve existing vectors
    store.ensure_collection(dim=768, force=args.force_reindex)

    # -----------------------------
    # Step 3–5: Process docs (RESUMABLE)
    # -----------------------------
    total_pages = 0
    total_chunks = 0
    total_vectors = 0

    for doc in tqdm(doc_names, desc="Processing PDFs"):

        # -------------------------
        # SKIP IF ALREADY DONE
        # -------------------------
        if doc in completed_docs:
            print(f"[SKIP] {doc}")
            continue

        pdf_path = os.path.join(PDF_DIR, f"{doc}.pdf")

        if not os.path.exists(pdf_path):
            print(f"[WARN] Missing PDF: {pdf_path}")
            continue

        try:
            # -------------------------
            # Load + chunk
            # -------------------------
            pages = load_cached_pages(doc, PAGES_CACHE_DIR)

            if pages is None:
                pages = load_pdf(pdf_path)
                save_cached_pages(doc, pages, PAGES_CACHE_DIR)

            # pages = load_pdf(pdf_path)
            chunks = chunk_pages(pages)

            texts = [c["text"] for c in chunks]

            # -------------------------
            # Embeddings (batched + retry inside your class)
            # -------------------------
            vectors = embedder.embed_documents(texts, batch_size=100)

            # -------------------------
            # Upsert into Qdrant
            # -------------------------
            store.upsert(chunks, vectors)

            # -------------------------
            # Update stats
            # -------------------------
            total_pages += len(pages)
            total_chunks += len(chunks)
            total_vectors += len(vectors)

            # -------------------------
            # MARK SUCCESS (checkpoint)
            # -------------------------
            completed_docs.add(doc)
            checkpoint["completed_docs"] = list(completed_docs)
            save_checkpoint(checkpoint)

            print(f"[DONE] {doc}")

        except Exception as e:
            print(f"[ERROR] Failed on {doc}: {e}")

            # Save progress before crashing
            checkpoint["completed_docs"] = list(completed_docs)
            save_checkpoint(checkpoint)

            raise

    # -----------------------------
    # FINAL STATS
    # -----------------------------
    print("\n==============================")
    print("INDEX BUILD COMPLETE")
    print("==============================")
    print(f"#Docs      : {len(doc_names)}")
    print(f"#Pages     : {total_pages}")
    print(f"#Chunks    : {total_chunks}")
    print(f"#Vectors   : {total_vectors}")
    print(f"Collection : financebench_v1")


if __name__ == "__main__":
    main()