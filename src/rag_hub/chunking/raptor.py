"""
raptor.py
---------
RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.

Algorithm (2 levels)
--------------------
Level 0  — leaf chunks (from recursive splitter, standard ~1 000 chars)
Level 1  — cluster leaves → summarise each cluster → summary chunks
Level 2  — cluster L1 summaries → summarise → top-level summary chunks

All levels are returned and indexed together in Qdrant.
The 'level' payload field lets you filter by abstraction depth at query time.

Dependencies
------------
    uv add umap-learn scikit-learn
    (Google Gemini LLM already available via rag_hub.generation.gemini_LLM)
"""

import uuid
from typing import List, Dict, Tuple

import numpy as np

from rag_hub.chunking.recursive import chunk_pages as recursive_chunk_pages
from rag_hub.embeddings.gemini_001 import GeminiEmbeddingClient
from rag_hub.generation.gemini_LLM import GeminiFlashGenerator


# ── Hyper-parameters ─────────────────────────────────────────────────────────
LEAF_CHUNK_SIZE      = 1_000
LEAF_OVERLAP         = 200
UMAP_N_COMPONENTS    = 10    # reduced dim for clustering
UMAP_N_NEIGHBORS     = 15
GMM_MAX_COMPONENTS   = 10    # upper bound; BIC selects the best k
MAX_RAPTOR_LEVELS    = 2
MIN_CLUSTER_SIZE     = 2     # don't summarise singleton clusters


# ── Public entry point ────────────────────────────────────────────────────────

def chunk_pages(
    pages: List[Dict],
    gemini_client: GeminiEmbeddingClient = None,
    llm: GeminiFlashGenerator = None,
    max_levels: int = MAX_RAPTOR_LEVELS,
) -> List[Dict]:
    """
    RAPTOR chunker.

    Returns leaf chunks + all summary chunks (all levels merged).
    Each chunk has an extra  'level'  field:
        0 = leaf, 1 = first summary layer, 2 = second summary layer, …

    Parameters
    ----------
    pages          : page dicts from pdf_loader
    gemini_client  : optional pre-built embedder (avoids re-init in sweep)
    llm            : optional pre-built LLM (avoids re-init)
    max_levels     : how many summarisation levels to build (1 or 2 recommended)
    """
    if not pages:
        return []

    if gemini_client is None:
        gemini_client = GeminiEmbeddingClient()
    if llm is None:
        llm = GeminiFlashGenerator()

    doc_name = pages[0]["doc_name"]

    # ── Level 0: leaf chunks ─────────────────────────────────────────────────
    leaf_chunks = recursive_chunk_pages(
        pages,
        chunk_size=LEAF_CHUNK_SIZE,
        chunk_overlap=LEAF_OVERLAP,
    )
    for c in leaf_chunks:
        c["level"] = 0

    all_chunks = list(leaf_chunks)
    current_level_chunks = leaf_chunks

    # ── Iterative summarisation levels ───────────────────────────────────────
    for level in range(1, max_levels + 1):
        texts = [c["text"] for c in current_level_chunks]
        if len(texts) < MIN_CLUSTER_SIZE:
            break

        print(f"  [RAPTOR] Level {level}: embedding {len(texts)} chunks …")
        vectors = gemini_client.embed_documents(texts, batch_size=100)
        vectors_np = np.array(vectors, dtype=np.float32)

        labels = _cluster(vectors_np)

        print(f"  [RAPTOR] Level {level}: {len(set(labels))} clusters found")

        summary_chunks = _summarise_clusters(
            chunks=current_level_chunks,
            labels=labels,
            llm=llm,
            doc_name=doc_name,
            level=level,
        )

        if not summary_chunks:
            break

        all_chunks.extend(summary_chunks)
        current_level_chunks = summary_chunks

    return all_chunks


# ── Clustering ────────────────────────────────────────────────────────────────

def _cluster(vectors: np.ndarray) -> List[int]:
    """
    UMAP dimensionality reduction → Gaussian Mixture Model clustering.
    BIC is used to select the optimal number of components.

    Returns a list of integer cluster labels (same length as vectors).
    """
    from umap import UMAP
    from sklearn.mixture import GaussianMixture

    n_samples = len(vectors)
    n_components_umap = min(UMAP_N_COMPONENTS, n_samples - 2)

    # UMAP reduction
    reducer = UMAP(
        n_components=n_components_umap,
        n_neighbors=min(UMAP_N_NEIGHBORS, n_samples - 1),
        metric="cosine",
        random_state=42,
    )
    reduced = reducer.fit_transform(vectors)

    # Select best k via BIC
    max_k = min(GMM_MAX_COMPONENTS, n_samples // MIN_CLUSTER_SIZE)
    best_gmm, best_bic = None, np.inf

    for k in range(2, max_k + 1):
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
        gmm.fit(reduced)
        bic = gmm.bic(reduced)
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm

    if best_gmm is None:
        # Fallback: single cluster
        return [0] * n_samples

    return best_gmm.predict(reduced).tolist()


# ── Summarisation ─────────────────────────────────────────────────────────────

_SUMMARY_PROMPT = (
    "You are a financial analyst. Summarise the following excerpts from a financial document "
    "into a single, dense paragraph (4-6 sentences). Preserve all numbers, dates, and key facts.\n\n"
    "Excerpts:\n{passages}\n\nSummary:"
)


def _summarise_clusters(
    chunks: List[Dict],
    labels: List[int],
    llm: GeminiFlashGenerator,
    doc_name: str,
    level: int,
) -> List[Dict]:
    """Group chunks by cluster label, summarise each group, return summary chunks."""
    from collections import defaultdict

    groups: Dict[int, List[Dict]] = defaultdict(list)
    for chunk, label in zip(chunks, labels):
        groups[label].append(chunk)

    summary_chunks = []

    for label, group in groups.items():
        if len(group) < MIN_CLUSTER_SIZE:
            continue

        # Build prompt
        passages = "\n\n---\n\n".join(c["text"] for c in group)
        prompt   = _SUMMARY_PROMPT.format(passages=passages[:8_000])  # cap tokens

        try:
            # GeminiFlashGenerator.generate() expects (question, chunks).
            # For RAPTOR we pass the full prompt as the "question" and no chunks.
            summary_text = llm.generate(question=prompt, chunks=[])
        except Exception as e:
            print(f"  [WARN] LLM summarisation failed for cluster {label}: {e}")
            continue

        # Use the median page from the cluster as the page reference
        pages_in_cluster = [c.get("page", 0) for c in group]
        representative_page = int(np.median(pages_in_cluster))

        summary_chunks.append({
            "id":        str(uuid.uuid4()),
            "doc_name":  doc_name,
            "page":      representative_page,
            "text":      summary_text,
            "chunk_idx": label,
            "level":     level,
            "cluster":   label,
            "n_sources": len(group),
        })

    return summary_chunks
