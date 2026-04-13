import json
import random
from collections import defaultdict
from typing import List, Dict, Set


# -----------------------------
# Step 1: Load JSONL questions
# -----------------------------
def load_questions(jsonl_path: str) -> List[Dict]:
    """
    Reads a JSONL file and returns a list of question dictionaries.
    Each line is expected to be a valid JSON object.
    """
    questions = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))

    return questions


# ---------------------------------------------
# Step 2: Extract gold evidence page numbers
# ---------------------------------------------
def gold_pages(record: Dict) -> Set[int]:
    """
    Extracts evidence page numbers from a FinanceBench record.

    Returns:
        Set of 0-indexed page numbers.
    """
    pages = set()

    for ev in record.get("evidence", []):
        page = ev.get("evidence_page_num")
        if page is not None:
            pages.add(int(page))

    return pages


# ---------------------------------------------------
# Step 3: Stratified smoke set (deterministic)
# ---------------------------------------------------
def sample_smoke_set(
    questions: List[Dict],
    n: int = 20,
    seed: int = 42
) -> List[Dict]:
    """
    Creates a stratified sample:
    - Stratified by question_type
    - Within each stratum, round-robin across company
    - Deterministic via seed
    """

    random.seed(seed)

    # Group by question_type
    by_type = defaultdict(list)
    for q in questions:
        by_type[q.get("question_type", "unknown")].append(q)

    # Shuffle within each type (deterministic)
    for t in by_type:
        random.Random(seed).shuffle(by_type[t])

    # Round-robin sampling across types
    selected = []
    pointers = {t: 0 for t in by_type}
    types = list(by_type.keys())

    while len(selected) < n:
        progress = False

        for t in types:
            i = pointers[t]
            if i < len(by_type[t]):
                selected.append(by_type[t][i])
                pointers[t] += 1
                progress = True

                if len(selected) == n:
                    break

        if not progress:
            break  # exhausted all data

    # -----------------------------
    # Sanity diagnostics
    # -----------------------------
    type_counts = defaultdict(int)
    doc_names = set()

    for q in selected:
        type_counts[q.get("question_type", "unknown")] += 1
        if "doc_name" in q:
            doc_names.add(q["doc_name"])

    print("\n=== Smoke Set Summary ===")
    print("Question type distribution:")
    for k, v in type_counts.items():
        print(f"  {k}: {v}")

    print("\nUnique doc_names:")
    print(f"  {len(doc_names)} documents")
    print(doc_names)

    return selected