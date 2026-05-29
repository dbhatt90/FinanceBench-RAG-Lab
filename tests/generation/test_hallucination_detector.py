from unittest.mock import MagicMock
from rag_hub.generation.schemas import Answer, Citation
from rag_hub.generation.hallucination_detector import HallucinationDetector, _split_sentences


def test_split_sentences_basic():
    sentences = _split_sentences("Revenue was $394B. iPhone grew 5%. Services hit record.")
    assert len(sentences) == 3


def test_split_sentences_filters_short():
    sentences = _split_sentences("OK. Revenue was $394B in fiscal 2022.")
    assert len(sentences) == 1


def _make_detector(label: str) -> HallucinationDetector:
    d = HallucinationDetector.__new__(HallucinationDetector)
    d.threshold = 0.25
    d._pipe = MagicMock(
        return_value=[[{"label": label, "score": 0.9}, {"label": "OTHER", "score": 0.1}]]
    )
    return d


def test_entailed_answer_zero_hallucination():
    answer = Answer(
        text="Revenue was $394B in fiscal 2022. iPhone grew 5%.",
        citations=[Citation(doc_id="AAPL", page=5, quote="Revenue was $394B in fiscal 2022")],
    )
    rate, labels = _make_detector("ENTAILMENT").check(answer)
    assert rate == 0.0


def test_contradicted_answer_full_hallucination():
    answer = Answer(
        text="Revenue was $394B. iPhone grew 5%.",
        citations=[Citation(doc_id="AAPL", page=5, quote="Revenue was $394B")],
    )
    rate, labels = _make_detector("CONTRADICTION").check(answer)
    assert rate == 1.0


def test_no_citation_returns_full_hallucination():
    answer = Answer(text="Revenue was $394B.", citations=[])
    d = HallucinationDetector.__new__(HallucinationDetector)
    d.threshold = 0.25
    d._pipe = MagicMock()
    rate, labels = d.check(answer)
    assert rate == 1.0
    d._pipe.assert_not_called()
