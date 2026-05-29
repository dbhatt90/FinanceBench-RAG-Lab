"""
Hallucination detector: NLI-based sentence-level entailment check.
Flags sentences not entailed by the cited quotes as hallucinated.
"""
import re
from typing import List, Tuple

from transformers import pipeline as hf_pipeline

from rag_hub.generation.schemas import Answer


def _split_sentences(text: str) -> List[str]:
    """Split on sentence-ending punctuation; filter fragments shorter than 10 chars."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) >= 10]


class HallucinationDetector:
    """
    For each sentence in the answer, checks NLI entailment against cited quotes.
    Labels NEUTRAL or CONTRADICTION count as hallucinated.

    Model: cross-encoder/nli-deberta-v3-base (~180 MB, CPU-OK)
    """

    def __init__(
        self,
        model: str = "cross-encoder/nli-deberta-v3-base",
        device: int = -1,
        threshold: float = 0.25,
    ):
        self.threshold = threshold
        self._pipe = hf_pipeline(
            "text-classification",
            model=model,
            device=device,
            top_k=None,
        )

    def check(self, answer: Answer) -> Tuple[float, List[str]]:
        """
        Returns (hallucination_rate, per_sentence_labels).
        hallucination_rate = fraction of sentences not supported by citations.
        """
        sentences = _split_sentences(answer.text)
        if not sentences:
            return 0.0, []

        if not answer.citations:
            return 1.0, ["no_citation"] * len(sentences)

        premise = " ".join(c.quote for c in answer.citations)
        labels = []
        for sentence in sentences:
            raw = self._pipe({"text": premise, "text_pair": sentence})
            # raw is [[{"label": "ENTAILMENT", "score": 0.9}, ...]]
            candidates = raw[0] if isinstance(raw[0], list) else raw
            top = max(candidates, key=lambda x: x["score"])
            labels.append(top["label"].upper())

        hallucinated = sum(1 for lbl in labels if lbl in ("NEUTRAL", "CONTRADICTION"))
        return hallucinated / len(labels), labels
