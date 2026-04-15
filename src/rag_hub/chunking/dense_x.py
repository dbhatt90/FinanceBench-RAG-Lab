"""
dense_x.py  (Dense Passage Retrieval with Propositions / "Dense-X")
--------------------------------------------------------------------
Multi-representation indexing strategy.

Idea
----
Raw chunk text is verbose — embedding it gives a noisy vector.
Instead, distil each chunk into a single atomic "proposition" sentence,
embed that, but store the FULL chunk text in the Qdrant payload.

At query time, the query vector matches the crisp proposition better,
but the LLM receives the full chunk for richer context.

Reference: "Dense X Retrieval: What Retrieval Granularity Should We Use?" (2023)

Implementation
--------------
1. Chunk with RecursiveCharacterTextSplitter (same sizes as recursive.py)
2. For each chunk, call Gemini to produce a 1-sentence proposition
3. Index: vector = embed(proposition), payload.text = full chunk text
           payload also stores proposition for inspection

The chunk dict returned has:
    text        : full chunk (what the LLM sees)
    proposition : distilled sentence (what gets embedded)

The IndexBuilder embeds  chunk["proposition"]  instead of  chunk["text"].
We handle this by overriding the text field before embedding and restoring it.
To keep IndexBuilder generic, we set:
    chunk["text"]        = proposition   ← gets embedded
    chunk["full_text"]   = original text ← stored in payload, returned to LLM

The QdrantStore.upsert payload stores whatever is in chunk["text"],
so we swap the fields.  The retriever should prefer "full_text" if present.
"""

import uuid
from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_hub.generation.gemini_LLM import GeminiFlashGenerator


# ── Defaults ─────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200

_PROPOSITION_PROMPT = (
    "Convert the following financial text into a single, self-contained declarative "
    "sentence that captures the most important fact or figure. "
    "Output ONLY the sentence, nothing else.\n\n"
    "Text:\n{text}\n\nProposition:"
)


def chunk_pages(
    pages: List[Dict],
    llm: GeminiFlashGenerator = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Dict]:
    """
    Dense-X chunker.

    Returns chunks where:
        chunk["text"]      = proposition sentence  (gets embedded by IndexBuilder)
        chunk["full_text"] = original chunk text   (stored in payload, used by LLM)

    Parameters
    ----------
    pages        : page dicts from pdf_loader
    llm          : optional pre-built GeminiFlashGenerator
    chunk_size   : chars per raw chunk before proposition extraction
    chunk_overlap: overlap between raw chunks
    """
    if not pages:
        return []

    if llm is None:
        llm = GeminiFlashGenerator()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    doc_name = pages[0]["doc_name"]
    chunks: List[Dict] = []
    chunk_idx = 0

    for page in pages:
        page_text = page.get("text", "")
        page_num  = page["page"]

        if not page_text.strip():
            continue

        sub_texts = splitter.split_text(page_text)

        for raw_text in sub_texts:
            if not raw_text.strip():
                continue

            proposition = _extract_proposition(raw_text, llm)

            chunks.append({
                "id":         str(uuid.uuid4()),
                "doc_name":   doc_name,
                "page":       page_num,
                "text":       proposition,   # embedded
                "full_text":  raw_text,      # returned to LLM
                "chunk_idx":  chunk_idx,
            })
            chunk_idx += 1

    return chunks


# ── Proposition extraction ────────────────────────────────────────────────────

def _extract_proposition(text: str, llm: GeminiFlashGenerator) -> str:
    """
    Call Gemini to distil the chunk into a single proposition sentence.
    Falls back to the first sentence of the chunk if the LLM call fails.
    """
    prompt = _PROPOSITION_PROMPT.format(text=text[:2000])  # cap to avoid token overflow

    try:
        result = llm.generate(question=prompt, chunks=[])
        # Strip leading/trailing whitespace and any accidental quotes
        result = result.strip().strip('"').strip("'")
        if result:
            return result
    except Exception as e:
        print(f"  [WARN] Proposition extraction failed: {e}")

    # Fallback: use first sentence
    first_sentence = text.split(".")[0].strip()
    return first_sentence if first_sentence else text[:200]
