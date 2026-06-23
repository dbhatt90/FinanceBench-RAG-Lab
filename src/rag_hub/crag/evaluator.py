"""
CRAG relevance evaluator.

Scores retrieved docs against a question using a single LLM call (Gemini Flash).
All top-N docs are batched into one prompt → one response with per-doc labels.

Label weights: relevant=1.0, partially_relevant=0.5, irrelevant=0.0
Confidence = weighted mean of top eval_top_n docs.
Threshold: confidence >= threshold → sufficient; < threshold → trigger fallback.
"""

import re
from typing import List, Dict, Tuple

from dotenv import load_dotenv

from rag_hub.config.settings import GEMINI_LLM_MODEL, CRAG_CONFIDENCE_THRESHOLD
from rag_hub.config.vertex_ai import make_chat_llm

load_dotenv()

_LABEL_WEIGHTS = {"relevant": 1.0, "partially_relevant": 0.5, "irrelevant": 0.0}

_EVAL_PROMPT = """\
You are evaluating whether retrieved document passages are relevant to a question.

Question: {question}

For each passage below, respond with exactly one label:
  relevant            — passage directly answers or strongly supports the question
  partially_relevant  — passage is on the same topic but does not directly answer
  irrelevant          — passage is off-topic or unhelpful

Respond with ONLY a numbered list, one label per line, in the same order as the passages.
Example format:
1. relevant
2. irrelevant
3. partially_relevant

Passages:
{passages}

Your labels:"""


class CRAGEvaluator:
    def __init__(
        self,
        threshold: float = CRAG_CONFIDENCE_THRESHOLD,
        eval_top_n: int = 3,
        model: str = GEMINI_LLM_MODEL,
    ):
        self.threshold = threshold
        self.eval_top_n = eval_top_n
        self.llm = make_chat_llm(model, temperature=0)

    def evaluate(self, question: str, docs: List[Dict]) -> Tuple[float, List[str]]:
        """
        Returns (confidence_score, per_doc_labels) for the top eval_top_n docs.
        confidence_score is in [0.0, 1.0]; >= threshold means retrieval is sufficient.
        """
        eval_docs = docs[: self.eval_top_n]
        if not eval_docs:
            return 0.0, []

        prompt = self._build_prompt(question, eval_docs)
        try:
            response = self.llm.invoke(prompt)
            labels = self._parse_labels(response.content, len(eval_docs))
        except Exception:
            # On LLM failure, assume partial relevance so we don't always fallback
            labels = ["partially_relevant"] * len(eval_docs)

        weights = [_LABEL_WEIGHTS.get(lbl, 0.5) for lbl in labels]
        confidence = sum(weights) / len(weights) if weights else 0.0
        return round(confidence, 4), labels

    def _build_prompt(self, question: str, docs: List[Dict]) -> str:
        passages = "\n\n".join(
            f"{i+1}. [{d.get('doc_name','?')} p.{d.get('page','?')}]\n{d.get('text','')[:400]}"
            for i, d in enumerate(docs)
        )
        return _EVAL_PROMPT.format(question=question, passages=passages)

    def _parse_labels(self, response: str, n_docs: int) -> List[str]:
        """
        Parses numbered list response into per-doc label strings.
        Robust to minor formatting variation (extra spaces, missing numbers).
        """
        valid = set(_LABEL_WEIGHTS.keys())
        labels: List[str] = []

        # Try to find lines like "1. relevant" or "relevant"
        for line in response.strip().splitlines():
            line = line.strip().lower()
            # Strip leading number and punctuation
            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            # Normalise: "partially relevant" → "partially_relevant"
            line = line.replace(" ", "_")
            if line in valid:
                labels.append(line)
            elif "partially" in line:
                labels.append("partially_relevant")
            elif "relevant" in line:
                labels.append("relevant")
            elif "irrelevant" in line:
                labels.append("irrelevant")

        # Pad or trim to match expected count
        if len(labels) < n_docs:
            labels.extend(["partially_relevant"] * (n_docs - len(labels)))
        return labels[:n_docs]
