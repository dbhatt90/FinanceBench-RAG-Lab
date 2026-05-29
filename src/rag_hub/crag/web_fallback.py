"""
Web search fallback for CRAG.

When the CRAG evaluator signals low retrieval confidence, this module queries
DuckDuckGo and formats results as chunk dicts — same schema as Qdrant payloads —
so downstream nodes (generate_node) need no special handling.

For financial RAG, this fires rarely: the corpus covers indexed filings.
Its main value is recovering from questions about post-filing events or
companies not in the index.
"""

from typing import List, Dict

from langchain_community.tools import DuckDuckGoSearchRun


class WebFallback:
    def __init__(self, max_results: int = 3):
        self.max_results = max_results
        self._tool = DuckDuckGoSearchRun()

    def search(self, query: str) -> List[Dict]:
        """
        Returns results as chunk dicts matching the corpus payload schema:
          {"doc_name": "web:<query_slug>", "page": 0, "text": <snippet>,
           "chunk_idx": i, "source": "web"}

        The doc_name encodes the query so it is traceable in logs.
        """
        try:
            raw = self._tool.run(query)
        except Exception:
            return []

        if not raw or raw.strip() == "":
            return []

        # DuckDuckGoSearchRun returns a single concatenated string.
        # Split on double-newline or period sequences to get snippet chunks.
        snippets = [s.strip() for s in raw.split("\n\n") if s.strip()]
        if not snippets:
            snippets = [raw.strip()]

        slug = query[:40].replace(" ", "_").replace("/", "-")
        results: List[Dict] = []
        for i, snippet in enumerate(snippets[: self.max_results]):
            results.append({
                "doc_name": f"web:{slug}",
                "page": 0,
                "text": snippet,
                "chunk_idx": i,
                "source": "web",
            })
        return results
