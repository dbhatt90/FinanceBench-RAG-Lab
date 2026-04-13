# import os
# import json
# from collections import defaultdict

# from tqdm import tqdm

# from rag_hub.eval.financebench import load_questions, sample_smoke_set
# from rag_hub.loaders.pdf_loader import load_pdf
# from rag_hub.chunking.recursive import chunk_pages
# from rag_hub.embeddings.gemini_001 import GeminiEmbeddingClient
# from rag_hub.vectorstore.qdrant_store import QdrantStore


# PDF_DIR = "data/raw/financebench/pdfs"
# SMOKE_PATH = "data/eval/smoke_20.jsonl"


# def save_jsonl(path, data):
#     os.makedirs(os.path.dirname(path), exist_ok=True)

#     with open(path, "w", encoding="utf-8") as f:
#         for item in data:
#             f.write(json.dumps(item) + "\n")


# def main():
#     # -----------------------------
#     # Step 1: Load + sample questions
#     # -----------------------------
#     questions = load_questions("data/raw/financebench/data/financebench_open_source.jsonl")
#     smoke_questions = sample_smoke_set(questions, n=20, seed=42)

#     save_jsonl(SMOKE_PATH, smoke_questions)

#     print(f"\n[INFO] Saved smoke set → {SMOKE_PATH}")

#     # -----------------------------
#     # Step 2: Collect unique docs
#     # -----------------------------
#     doc_names = sorted(set(q["doc_name"] for q in smoke_questions))

#     print(f"\n[INFO] Unique docs: {len(doc_names)}")

#     # -----------------------------
#     # Step 3: Load PDFs → pages → chunks
#     # -----------------------------
#     all_chunks = []
#     total_pages = 0

#     for doc in tqdm(doc_names, desc="Processing PDFs"):
#         pdf_path = os.path.join(PDF_DIR, f"{doc}.pdf")

#         if not os.path.exists(pdf_path):
#             print(f"[WARN] Missing PDF: {pdf_path}")
#             continue

#         pages = load_pdf(pdf_path)
#         total_pages += len(pages)

#         chunks = chunk_pages(pages)
#         all_chunks.extend(chunks)

#     print(f"\n[INFO] Total pages: {total_pages}")
#     print(f"[INFO] Total chunks: {len(all_chunks)}")

#     # -----------------------------
#     # Step 4: Embeddings
#     # -----------------------------
#     embedder = GeminiEmbeddingClient()

#     texts = [c["text"] for c in all_chunks]

#     print("\n[INFO] Generating embeddings...")

#     vectors = embedder.embed_documents(texts, batch_size=100)

#     print(f"[INFO] Embeddings generated: {len(vectors)}")

#     # -----------------------------
#     # Step 5: Qdrant indexing
#     # -----------------------------
#     store = QdrantStore(collection="financebench_v1")

#     store.ensure_collection(dim=768)
#     store.upsert(all_chunks, vectors)

#     # -----------------------------
#     # Step 6: Final stats
#     # -----------------------------
#     print("\n==============================")
#     print("INDEX BUILD COMPLETE")
#     print("==============================")
#     print(f"#Docs      : {len(doc_names)}")
#     print(f"#Pages     : {total_pages}")
#     print(f"#Chunks    : {len(all_chunks)}")
#     print(f"#Vectors   : {len(vectors)}")
#     print(f"Collection : financebench_v1")


# if __name__ == "__main__":
#     main()


import os
import json
import argparse
from tqdm import tqdm

from rag_hub.eval.financebench import load_questions, sample_smoke_set
from rag_hub.loaders.pdf_loader import load_pdf
from rag_hub.chunking.recursive import chunk_pages
from rag_hub.embeddings.gemini_001 import GeminiEmbeddingClient
from rag_hub.vectorstore.qdrant_store import QdrantStore

from checkpoint_indexing import load_checkpoint, save_checkpoint


PDF_DIR = "data/raw/financebench/pdfs"
SMOKE_PATH = "data/eval/smoke_20.jsonl"
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

    smoke_questions = sample_smoke_set(questions, n=20, seed=42)

    save_jsonl(SMOKE_PATH, smoke_questions)
    print(f"\n[INFO] Saved smoke set → {SMOKE_PATH}")

    # -----------------------------
    # Step 2: Collect docs
    # -----------------------------
    doc_names = sorted(set(q["doc_name"] for q in smoke_questions))
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
            pages = load_pdf(pdf_path)
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