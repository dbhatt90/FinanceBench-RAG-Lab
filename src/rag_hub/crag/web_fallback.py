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


class WebFallback:
    def __init__(self, max_results: int = 3):
        self.max_results = max_results
        # Built lazily so a missing/broken search backend (e.g. the `ddgs`
        # package) degrades to "no web docs" instead of crashing graph startup.
        self._tool = None

    def _get_tool(self):
        if self._tool is None:
            from langchain_community.tools import DuckDuckGoSearchRun
            self._tool = DuckDuckGoSearchRun()
        return self._tool

    def search(self, query: str) -> List[Dict]:
        """
        Returns results as chunk dicts matching the corpus payload schema:
          {"doc_name": "web:<query_slug>", "page": 0, "text": <snippet>,
           "chunk_idx": i, "source": "web"}

        The doc_name encodes the query so it is traceable in logs.
        """
        try:
            raw = self._get_tool().run(query)
        except Exception as e:
            # DuckDuckGo throttles aggressively; degrade to no web docs rather
            # than crash the pipeline. Logged so throttling is visible.
            print(f"[WebFallback] search failed ({type(e).__name__}: {e}); returning no web docs")
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
