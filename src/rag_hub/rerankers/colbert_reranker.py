"""
ColBERT late-interaction reranker using colbert-ir/colbertv2.0.

Implements MaxSim scoring without ragatouille:
  1. Encode query tokens and document tokens independently into dense vectors.
  2. For each query token, find the max cosine similarity to any document token.
  3. Sum these per-token maxes → final relevance score.

This is ~5-10x slower than BGE cross-encoder per query (due to matrix ops per doc)
but captures fine-grained token-level matching — useful for numerical expressions
where "7.2 billion" and "$7,200M" share token-level similarity even when their
sentence-level embeddings differ.
"""

from typing import List, Dict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class ColBERTReranker:
    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        device: str = "cpu",
        max_query_len: int = 32,
        max_doc_len: int = 256,
    ):
        self.device = device
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(device)

        # ColBERT uses special prefix tokens: [Q] for queries, [D] for docs
        # We inject them as the first token after [CLS]
        self._q_token_id = self.tokenizer.convert_tokens_to_ids("[unused0]")
        self._d_token_id = self.tokenizer.convert_tokens_to_ids("[unused1]")

    def _encode(self, texts: List[str], is_query: bool) -> torch.Tensor:
        """
        Returns normalised token embeddings: shape (batch, seq_len, hidden_dim).
        Query tokens are padded with mask tokens to max_query_len (ColBERT convention).
        """
        prefix_id = self._q_token_id if is_query else self._d_token_id
        max_len = self.max_query_len if is_query else self.max_doc_len

        # Prepend prefix token id to each text
        prefixed = [f"[unused{'0' if is_query else '1'}] " + t for t in texts]

        encoded = self.tokenizer(
            prefixed,
            padding="max_length" if is_query else True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded)

        # Use last hidden state; mask padding tokens before normalisation
        hidden = outputs.last_hidden_state  # (batch, seq, hidden)
        attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
        hidden = hidden * attention_mask  # zero out padding positions
        hidden = F.normalize(hidden, p=2, dim=-1)  # L2-normalise per token

        return hidden  # (batch, seq_len, hidden_dim)

    def _maxsim_score(self, query_embs: torch.Tensor, doc_embs: torch.Tensor) -> float:
        """
        MaxSim: for each query token, find max cosine similarity to any doc token.
        Sum across query tokens → scalar score.

        query_embs: (q_len, hidden)
        doc_embs:   (d_len, hidden)
        """
        # sim[i, j] = cosine(query_token_i, doc_token_j)  — already L2-normalised
        sim = torch.matmul(query_embs, doc_embs.T)  # (q_len, d_len)
        max_sim = sim.max(dim=-1).values             # (q_len,)
        return max_sim.sum().item()

    def rerank(self, query: str, docs: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Returns docs sorted by ColBERT MaxSim score descending, trimmed to top_k.
        Adds 'colbert_score' key to each returned dict (does not mutate input).
        """
        if not docs:
            return docs

        # Encode query once
        query_embs = self._encode([query], is_query=True)[0]  # (q_len, hidden)

        scored = []
        texts = [d.get("text", "") for d in docs]

        # Encode all docs in one pass for efficiency
        doc_embs_batch = self._encode(texts, is_query=False)  # (n_docs, d_len, hidden)

        for doc, doc_embs in zip(docs, doc_embs_batch):
            score = self._maxsim_score(query_embs, doc_embs)
            scored.append(dict(doc, colbert_score=score))

        scored.sort(key=lambda x: x["colbert_score"], reverse=True)
        return scored[:top_k]
