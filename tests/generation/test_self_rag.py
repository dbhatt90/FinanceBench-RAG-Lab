from unittest.mock import MagicMock
from rag_hub.generation.schemas import Answer, Citation
from rag_hub.generation.self_rag import SelfRAGScorer, _ScoreResult

CHUNKS = [{"doc_name": "AAPL_2023_10K", "page": 5, "text": "Net revenue was $394.3 billion."}]
ANSWER = Answer(
    text="Apple's net revenue was $394.3 billion in fiscal 2022.",
    citations=[Citation(doc_id="AAPL_2023_10K", page=5, quote="Net revenue was $394.3 billion")],
)


def test_score_returns_confidence_from_llm():
    scorer = SelfRAGScorer.__new__(SelfRAGScorer)
    scorer._chain = MagicMock(invoke=MagicMock(return_value=_ScoreResult(
        is_relevant=True, is_supported=True, is_useful=True,
        confidence=0.92, reasoning="Answer directly matches context."
    )))
    assert scorer.score("What was Apple's revenue?", ANSWER, CHUNKS) == 0.92


def test_score_falls_back_to_0_5_on_error():
    scorer = SelfRAGScorer.__new__(SelfRAGScorer)
    scorer._chain = MagicMock(invoke=MagicMock(side_effect=Exception("LLM error")))
    assert scorer.score("Q?", ANSWER, CHUNKS) == 0.5
