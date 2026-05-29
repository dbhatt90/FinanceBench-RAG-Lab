"""
BGE cross-encoder reranker using BAAI/bge-reranker-v2-m3.

Takes a query and a list of retrieved chunk dicts, scores every (query, doc_text)
pair via a cross-encoder forward pass, and returns docs sorted by score descending.
Uses transformers directly — no sentence-transformers dependency needed.
"""

from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class BGEReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 512,
    ):
        self.device = device
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
