"""
Generation quality metrics: ROUGE, BERTScore, RAGAS.
ROUGE runs locally (no API). BERTScore downloads model on first call (~1.5 GB).
RAGAS requires LLM API — call sparingly.
"""
from typing import Dict, List

from rouge_score import rouge_scorer


class GenerationMetrics:
    def __init__(self):
        self._rouge = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

    def rouge_scores(self, prediction: str, reference: str) -> Dict[str, float]:
        """Returns rouge1, rouge2, rougeL F-measures."""
        scores = self._rouge.score(reference, prediction)
        return {
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rouge2": round(scores["rouge2"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4),
        }

    def bert_score(self, prediction: str, reference: str) -> float:
        """BERTScore F1. Loads model on first call (~1.5 GB)."""
        from bert_score import score as _bert_score
        P, R, F1 = _bert_score(
            [prediction], [reference],
            lang="en",
            rescale_with_baseline=True,
            verbose=False,
        )
        return round(float(F1[0]), 4)

    def ragas_metrics(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str,
    ) -> Dict[str, float]:
        """Faithfulness + answer_relevancy + answer_correctness via RAGAS. LLM-backed."""
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, answer_correctness

        dataset = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth],
        })
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, answer_correctness])
        return {
            "faithfulness": round(float(result["faithfulness"]), 4),
            "answer_relevancy": round(float(result["answer_relevancy"]), 4),
            "answer_correctness": round(float(result["answer_correctness"]), 4),
        }
