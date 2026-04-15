"""
parent_child.py
---------------
Parent-child chunker.

Strategy
--------
1. Split each page into large "parent" chunks  (~2 000 chars)
2. Split each parent into small "child" chunks (~400 chars)
3. Index ONLY children in Qdrant (their embeddings are more precise)
4. At retrieval time, swap each child for its full parent text

The parent store is a plain dict  {parent_id -> parent_text}
serialised to  data/processed/parent_store/{doc_name}.json
so it survives restarts and can be loaded by the retriever.

Chunk payload gains:
    parent_id   : str   — links child back to its parent
    parent_text : str   — full parent text stored directly in payload
                         (avoids a separate lookup at retrieval time)
"""

import json
import os
import uuid
from typing import Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ── Size defaults ────────────────────────────────────────────────────────────
PARENT_CHUNK_SIZE  = 2_000
PARENT_OVERLAP     = 200
CHILD_CHUNK_SIZE   = 400
CHILD_OVERLAP      = 50

PARENT_STORE_DIR   = "data/processed/parent_store"


def chunk_pages(
    pages: List[Dict],
    parent_chunk_size: int = PARENT_CHUNK_SIZE,
    parent_overlap:    int = PARENT_OVERLAP,
    child_chunk_size:  int = CHILD_CHUNK_SIZE,
    child_overlap:     int = CHILD_OVERLAP,
    parent_store_dir:  str = PARENT_STORE_DIR,
) -> List[Dict]:
    """
    Parent-child chunker.

    Only the CHILD chunks are returned (to be embedded + upserted).
    Parent text is embedded inside each child's payload so the retriever
    can return it without a second lookup.

    Parameters
    ----------
    pages : list of page dicts from pdf_loader
    parent_chunk_size : chars per parent
    parent_overlap    : overlap between parents
    child_chunk_size  : chars per child
    child_overlap     : overlap between children of the same parent
    parent_store_dir  : directory to persist parent text (for inspection / debug)

    Returns
    -------
    List of child chunk dicts:
        {id, doc_name, page, text, chunk_idx, parent_id, parent_text}
    """
    if not pages:
        return []

    doc_name = pages[0]["doc_name"]

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_overlap,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_overlap,
    )

    child_chunks: List[Dict] = []
    parent_store: Dict[str, str] = {}
    chunk_idx = 0

    for page in pages:
        page_text = page.get("text", "")
        page_num  = page["page"]

        if not page_text.strip():
            continue

        parents = parent_splitter.split_text(page_text)

        for parent_text in parents:
            parent_id = str(uuid.uuid4())
            parent_store[parent_id] = parent_text

            children = child_splitter.split_text(parent_text)

            for child_text in children:
                if not child_text.strip():
                    continue
                child_chunks.append({
                    "id":          str(uuid.uuid4()),
                    "doc_name":    doc_name,
                    "page":        page_num,
                    "text":        child_text,       # what gets embedded
                    "chunk_idx":   chunk_idx,
                    "parent_id":   parent_id,
                    "parent_text": parent_text,      # what gets returned to the LLM
                })
                chunk_idx += 1

    # Persist parent store to disk (useful for debugging / retriever swap)
    _save_parent_store(doc_name, parent_store, parent_store_dir)

    return child_chunks


# ── Parent store persistence ─────────────────────────────────────────────────

def _save_parent_store(doc_name: str, store: Dict[str, str], store_dir: str):
    os.makedirs(store_dir, exist_ok=True)
    path = os.path.join(store_dir, f"{doc_name}_parents.json")
    tmp  = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(store, f)
    os.replace(tmp, path)


def load_parent_store(doc_name: str, store_dir: str = PARENT_STORE_DIR) -> Optional[Dict[str, str]]:
    """Load the parent store for a given document (used by retriever)."""
    path = os.path.join(store_dir, f"{doc_name}_parents.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
