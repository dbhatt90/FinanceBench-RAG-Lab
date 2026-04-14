import json
import os
from typing import List, Dict
import pdfplumber
from tqdm import tqdm


def load_pdf(path: str) -> List[Dict]:
    """
    Loads a PDF and returns a list of page-level documents.

    Each item:
    {
        "doc_name": str,
        "page": int,   # 0-indexed
        "text": str
    }

    Notes:
    - Keeps empty pages to preserve page alignment with ground truth
    - page numbering is converted from 1-indexed (pdfplumber) to 0-indexed
    - Logs progress for large documents
    """

    pages_data = []
    doc_name = path.split("/")[-1]

    with pdfplumber.open(path) as pdf:
        num_pages = len(pdf.pages)

        for i in tqdm(range(num_pages), desc=f"Loading {doc_name}"):
            page = pdf.pages[i]

            text = page.extract_text() or ""

            pages_data.append({
                "doc_name": doc_name,
                "page": i,  # already 0-indexed
                "text": text
            })

    return pages_data

CACHE_VERSION = "v1" 
def load_cached_pages(doc_name, cache_dir):
    path = os.path.join(cache_dir, f"{doc_name}_{CACHE_VERSION}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_cached_pages(doc_name, pages, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{doc_name}_{CACHE_VERSION}.json")
    tmp_path = path + ".tmp"

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(pages, f)

    os.replace(tmp_path, path)