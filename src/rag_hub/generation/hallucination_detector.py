"""
Hallucination detector: NLI-based sentence-level entailment check.
Flags sentences not entailed by the cited quotes as hallucinated.
"""
import re
from typing import List, Tuple

from transformers import pipeline as hf_pipeline

from rag_hub.config.settings import NLI_MODEL, get_torch_device
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
        model: str = NLI_MODEL,
        device: str = None,
        threshold: float = 0.25,
        batch_size: int = 16,
    ):
        self.threshold = threshold
        self.batch_size = batch_size
        self._pipe = hf_pipeline(
            "text-classification",
            model=model,
            device=device or get_torch_device(),
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
        # Batch all sentence pairs in a single call (one forward pass) instead
        # of one model call per sentence — the main per-question bottleneck.
        inputs = [{"text": premise, "text_pair": s} for s in sentences]
        raw = self._run(inputs)

        labels = []
        for item in raw:
            # With top_k=None each item is a list of {"label","score"} dicts.
            candidates = item if isinstance(item, list) else [item]
            top = max(candidates, key=lambda x: x["score"])
            labels.append(top["label"].upper())

        hallucinated = sum(1 for lbl in labels if lbl in ("NEUTRAL", "CONTRADICTION"))
        return hallucinated / len(labels), labels

    def _run(self, inputs):
        """Run the NLI pipeline, falling back to CPU once if the device errors."""
        try:
            return self._pipe(inputs, batch_size=self.batch_size)
        except RuntimeError as e:
            dev = getattr(self._pipe, "device", None)
            if dev is not None and getattr(dev, "type", str(dev)) != "cpu":
                import torch
                print(f"[HallucinationDetector] {dev} failed ({e}); falling back to CPU")
                self._pipe.model.to("cpu")
                self._pipe.device = torch.device("cpu")
                return self._pipe(inputs, batch_size=self.batch_size)
            raise
