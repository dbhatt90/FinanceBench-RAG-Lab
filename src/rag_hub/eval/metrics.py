import re
import string
from typing import List, Dict, Set


# -----------------------------
# Text normalization
# -----------------------------
def normalize(text: str) -> str:
    """
    FinanceBench-safe normalization:
    - lowercase
    - remove punctuation
    - remove $, commas
    - collapse whitespace
    """

    if text is None:
        return ""

    text = text.lower()

    # remove currency symbols and commas
    text = text.replace("$", "").replace(",", "")

    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# -----------------------------
# Exact Match
# -----------------------------
def exact_match(pred: str, gold: str) -> bool:
    """
    FinanceBench exact match after normalization.
    """

    return normalize(pred) == normalize(gold)


# -----------------------------
# Hit@K (doc-aware + page-aware)
# -----------------------------
def _strip_pdf(name: str) -> str:
    """Normalise doc name by stripping .pdf suffix for comparison."""
    return name.removesuffix(".pdf") if name else ""


def hit_at_k(
    retrieved_chunks: List[Dict],
    gold_pages: Set[int],
    doc_name: str,
    k: int = 5
) -> bool:
    """
    Returns True if ANY of top-k retrieved chunks:
    - matches doc_name (ignoring .pdf suffix)
    - AND page is in gold_pages

    Note: Qdrant payloads store doc names with .pdf extension
    (e.g. 'CORNING_2021_10K.pdf') while FinanceBench ground truth
    omits it (e.g. 'CORNING_2021_10K'). We normalise both sides
    rather than re-indexing the entire corpus.
    """
    norm_gold = _strip_pdf(doc_name)

    for chunk in retrieved_chunks[:k]:
        if (
            _strip_pdf(chunk.get("doc_name", "")) == norm_gold
            and chunk.get("page") in gold_pages
        ):
            return True

    return False


# -----------------------------
# Token F1 (SQuAD-style)
# -----------------------------
def token_f1(pred: str, gold: str) -> float:
    """
    Token-level F1 score on normalized text.
    """

    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = set(pred_tokens) & set(gold_tokens)

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)