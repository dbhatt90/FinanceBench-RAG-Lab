"""
section_aware.py
----------------
10-K section-aware chunker.

Detects major SEC 10-K section boundaries (Items 1A, 7, 8) via regex,
splits the full document at those boundaries, then further splits large
sections with RecursiveCharacterTextSplitter so no chunk is too large.

Chunk payload gains a  'section'  field (e.g. "risk_factors", "mdna",
"financial_statements", "other") — useful for section-filtered retrieval.
"""

import re
import uuid
from typing import List, Dict, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ── Section boundary patterns (ordered: more-specific first) ────────────────
# Each tuple: (regex, section_label)
# We match case-insensitively on the concatenated full-doc text.
SECTION_PATTERNS: List[Tuple[str, str]] = [
    (r"item\s+1a[\.\s\-–—]",   "risk_factors"),
    (r"item\s+7a[\.\s\-–—]",   "quantitative_disclosures"),   # often before item 7
    (r"item\s+7[\.\s\-–—]",    "mdna"),
    (r"item\s+8[\.\s\-–—]",    "financial_statements"),
    (r"item\s+9a[\.\s\-–—]",   "controls_procedures"),
]

# ── Chunking limits ──────────────────────────────────────────────────────────
MAX_SECTION_CHARS = 6_000   # sections larger than this get further split
SECTION_OVERLAP   = 200


def _concat_pages(pages: List[Dict]) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Concatenate all page texts into one string.
    Also returns a list of (char_offset, page_number) so we can map a char
    position back to the original page number.
    """
    parts = []
    page_map: List[Tuple[int, int]] = []  # (start_offset, page_num)
    offset = 0

    for p in pages:
        text = p.get("text", "")
        page_map.append((offset, p["page"]))
        parts.append(text)
        offset += len(text) + 1  # +1 for the newline separator

    return "\n".join(parts), page_map


def _offset_to_page(offset: int, page_map: List[Tuple[int, int]]) -> int:
    """Binary-search the page_map to find which page a char offset belongs to."""
    lo, hi = 0, len(page_map) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if page_map[mid][0] <= offset:
            lo = mid
        else:
            hi = mid - 1
    return page_map[lo][1]


def _find_section_boundaries(full_text: str) -> List[Tuple[int, str]]:
    """
    Scan the full doc for section headers.
    Returns a sorted list of (char_offset, section_label).

    TOC-skip heuristic
    ------------------
    10-Ks list all Item numbers in a Table of Contents near the front.
    We skip any match that falls in the first 15 % of the document
    AND take the LAST match per label (the body occurrence, not the TOC line).
    The body match is always further into the doc than the TOC line.
    """
    doc_len     = len(full_text)
    toc_cutoff  = int(doc_len * 0.15)   # ignore matches in first 15 %

    boundaries = []
    for pattern, label in SECTION_PATTERNS:
        for m in re.finditer(pattern, full_text, re.IGNORECASE):
            if m.start() >= toc_cutoff:
                boundaries.append((m.start(), label))

    if not boundaries:
        # Fallback: if nothing found after cutoff, use the last occurrence of each
        for pattern, label in SECTION_PATTERNS:
            matches = list(re.finditer(pattern, full_text, re.IGNORECASE))
            if matches:
                boundaries.append((matches[-1].start(), label))

    # Keep only the FIRST body-section match per label (sorted by position)
    seen: Dict[str, int] = {}
    for pos, label in sorted(boundaries):
        if label not in seen:
            seen[label] = pos

    return sorted((pos, label) for label, pos in seen.items())


def chunk_pages(
    pages: List[Dict],
    max_section_chars: int = MAX_SECTION_CHARS,
    overlap: int = SECTION_OVERLAP,
) -> List[Dict]:
    """
    Section-aware chunker.

    Parameters
    ----------
    pages : list of page dicts from pdf_loader
    max_section_chars : sections larger than this are further split
    overlap : overlap used when splitting large sections

    Returns
    -------
    List of chunk dicts:
        {id, doc_name, page, text, chunk_idx, section}
    """
    if not pages:
        return []

    doc_name = pages[0]["doc_name"]
    full_text, page_map = _concat_pages(pages)
    boundaries = _find_section_boundaries(full_text)

    # Build (start, end, section_label) triples
    segments: List[Tuple[int, int, str]] = []
    doc_end = len(full_text)

    if not boundaries:
        # No sections detected (e.g. earnings release, 10-Q) — treat as one block
        segments.append((0, doc_end, "other"))
    else:
        # Everything before the first detected section
        first_pos = boundaries[0][0]
        if first_pos > 0:
            segments.append((0, first_pos, "other"))

        for i, (pos, label) in enumerate(boundaries):
            end = boundaries[i + 1][0] if i + 1 < len(boundaries) else doc_end
            segments.append((pos, end, label))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_section_chars,
        chunk_overlap=overlap,
    )

    chunks: List[Dict] = []
    chunk_idx = 0

    for start, end, section in segments:
        section_text = full_text[start:end].strip()
        if not section_text:
            continue

        # Split large sections further
        sub_texts = splitter.split_text(section_text) if len(section_text) > max_section_chars else [section_text]

        for sub in sub_texts:
            if not sub.strip():
                continue
            # Approximate page: find offset of this sub-text in full_text
            sub_offset = full_text.find(sub[:80], start)  # search near section start
            page_num = _offset_to_page(
                sub_offset if sub_offset != -1 else start,
                page_map,
            )
            chunks.append({
                "id":        str(uuid.uuid4()),
                "doc_name":  doc_name,
                "page":      page_num,
                "text":      sub,
                "chunk_idx": chunk_idx,
                "section":   section,
            })
            chunk_idx += 1

    return chunks
