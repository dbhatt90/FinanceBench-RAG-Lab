from typing import List, Dict
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(
    pages: List[Dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Dict]:
    """
    Chunk page-level documents using LangChain RecursiveCharacterTextSplitter.

    IMPORTANT DESIGN RULE:
    - Chunks NEVER cross page boundaries (critical for FinanceBench eval)
    - Each chunk inherits doc_name + page metadata

    Output:
    {
        "id": str,
        "doc_name": str,
        "page": int,
        "text": str,
        "chunk_idx": int
    }
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = []

    for page in pages:
        text = page.get("text", "")
        doc_name = page["doc_name"]
        page_num = page["page"]

        # Split ONLY within page boundary
        split_texts = splitter.split_text(text)

        for idx, chunk_text in enumerate(split_texts):
            chunks.append({
                "id": str(uuid.uuid4()),
                "doc_name": doc_name,
                "page": page_num,
                "text": chunk_text,
                "chunk_idx": idx
            })

    return chunks