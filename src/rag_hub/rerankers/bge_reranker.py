"""
BGE cross-encoder reranker using BAAI/bge-reranker-v2-m3.

Takes a query and a list of retrieved chunk dicts, scores every (query, doc_text)
pair via a cross-encoder forward pass, and returns docs sorted by score descending.
Uses transformers directly — no sentence-transformers dependency needed.
"""

from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from rag_hub.config.settings import BGE_RERANKER_MODEL, get_torch_device


class BGEReranker:
    def __init__(
        self,
        model_name: str = BGE_RERANKER_MODEL,
        device: str = None,
        batch_size: int = 16,
        max_length: int = 512,
    ):
        self.device = device or get_torch_device()
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.model.to(device)

    def rerank(self, query: str, docs: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Score each (query, doc["text"]) pair via cross-encoder.
        Returns docs sorted by descending rerank_score, trimmed to top_k.
        Adds 'rerank_score' key to each returned dict (does not mutate input).
        """
        if not docs:
            return docs

        texts = [d.get("text", "") for d in docs]
        scores = self._score_pairs(query, texts)

        ranked = sorted(
            [dict(doc, rerank_score=float(score)) for doc, score in zip(docs, scores)],
            key=lambda x: x["rerank_score"],
            reverse=True,
        )
        return ranked[:top_k]

    def _score_pairs(self, query: str, texts: List[str]) -> List[float]:
        try:
            return self._score_pairs_on_device(query, texts)
        except RuntimeError as e:
            # Some models error on MPS ("Placeholder storage has not been
            # allocated on MPS device!"). Fall back to CPU once, permanently.
            if self.device != "cpu":
                print(f"[BGEReranker] {self.device} failed ({e}); falling back to CPU")
                self.device = "cpu"
                self.model.to("cpu")
                return self._score_pairs_on_device(query, texts)
            raise

    def _score_pairs_on_device(self, query: str, texts: List[str]) -> List[float]:
        all_scores: List[float] = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            encoded = self.tokenizer(
                [query] * len(batch_texts),
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**encoded).logits

            # Single logit (binary relevance) or first logit of multi-class — flatten to scalar
            scores = logits.squeeze(-1).tolist()
            if isinstance(scores, float):
                scores = [scores]
            all_scores.extend(scores)

        return all_scores
