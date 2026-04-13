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