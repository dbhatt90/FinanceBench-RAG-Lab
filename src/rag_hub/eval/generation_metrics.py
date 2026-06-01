"""
Generation quality metrics: ROUGE, BERTScore, RAGAS, Exact Match, Numeric Match.
ROUGE runs locally (no API). BERTScore downloads model on first call (~1.5 GB).
RAGAS requires LLM API — call sparingly.
"""
import re
from typing import Dict, List, Optional

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
        """BERTScore F1. Model is loaded once and cached on first call."""
        if not hasattr(self, "_bert_scorer"):
            from bert_score import BERTScorer
            self._bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
        P, R, F1 = self._bert_scorer.score([prediction], [reference])
        return round(float(F1[0]), 4)

    def exact_match(self, prediction: str, reference: str) -> bool:
        """True if the gold answer string appears anywhere in the prediction (case-insensitive)."""
        return reference.strip().lower() in prediction.strip().lower()

    def numeric_match(self, prediction: str, reference: str, tolerance: float = 0.01) -> Optional[bool]:
        """
        Extracts the first number from both strings and checks if they agree within
        `tolerance` (default 1%). Returns None if either string has no parseable number.
        """
        def extract(s: str) -> Optional[float]:
            s = s.replace(",", "").replace("%", "").replace("$", "")
            m = re.search(r"-?\d+(?:\.\d+)?", s)
            return float(m.group()) if m else None

        pred_val = extract(prediction)
        ref_val = extract(reference)
        if pred_val is None or ref_val is None:
            return None
        if ref_val == 0:
            return abs(pred_val) <= tolerance
        return abs(pred_val - ref_val) / abs(ref_val) <= tolerance

    def ragas_metrics(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str,
    ) -> Dict[str, float]:
        """
        Faithfulness + answer_relevancy + answer_correctness via RAGAS.
        Uses Vertex AI (Gemini) as the judge LLM — no OpenAI key required.
        Call sparingly: 3 LLM calls per question against free-tier quota.
        """
        import os
        import vertexai
        from datasets import Dataset
        from google.oauth2 import service_account
        from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.metrics import faithfulness, answer_relevancy, answer_correctness

        # Reuse the same Vertex AI credentials as the rest of the project
        credentials = service_account.Credentials.from_service_account_file(
            os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        vertexai.init(
            project=os.getenv("GCP_PROJECT_ID"),
            location=os.getenv("GCP_LOCATION", "us-central1"),
            credentials=credentials,
        )

        ragas_llm = LangchainLLMWrapper(ChatVertexAI(
            model_name="gemini-2.5-flash",
            temperature=0,
            project=os.getenv("GCP_PROJECT_ID"),
            location=os.getenv("GCP_LOCATION", "us-central1"),
            credentials=credentials,
        ))
        ragas_emb = LangchainEmbeddingsWrapper(VertexAIEmbeddings(
            model_name="text-embedding-004",
            project=os.getenv("GCP_PROJECT_ID"),
            location=os.getenv("GCP_LOCATION", "us-central1"),
            credentials=credentials,
        ))

        # Inject Vertex AI LLM/embeddings into each metric
        for metric in (faithfulness, answer_relevancy, answer_correctness):
            metric.llm = ragas_llm
        answer_relevancy.embeddings = ragas_emb

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
