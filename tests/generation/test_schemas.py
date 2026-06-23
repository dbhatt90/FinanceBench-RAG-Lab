import pytest
from pydantic import ValidationError
from rag_hub.generation.schemas import Citation, CitedAnswer, Answer


def test_citation_requires_doc_id_page_quote():
    c = Citation(doc_id="AAPL_2023_10K", page=5, quote="Revenue was $394B")
    assert c.doc_id == "AAPL_2023_10K"
    assert c.page == 5
    assert c.quote == "Revenue was $394B"


def test_citation_quote_truncated_to_500_chars():
    # Quotes longer than 500 chars are silently truncated rather than rejected,
    # so the LLM returning a long quote never breaks structured output parsing.
    c = Citation(doc_id="X", page=1, quote="A" * 600)
    assert len(c.quote) == 500


def test_cited_answer_defaults_empty_citations():
    ca = CitedAnswer(text="Revenue was $394B.")
    assert ca.citations == []


def test_answer_defaults():
    a = Answer(text="test", citations=[])
    assert a.confidence == 0.0
    assert a.hallucination_rate == 0.0
    assert a.generation_iterations == 1


def test_answer_confidence_bounded():
    with pytest.raises(ValidationError):
        Answer(text="x", citations=[], confidence=1.5)
    with pytest.raises(ValidationError):
        Answer(text="x", citations=[], confidence=-0.1)
    with pytest.raises(ValidationError):
        Answer(text="x", citations=[], hallucination_rate=-0.1)
    with pytest.raises(ValidationError):
        Answer(text="x", citations=[], hallucination_rate=1.5)


def test_answer_from_cited_answer():
    ca = CitedAnswer(
        text="Revenue was $394B.",
        citations=[Citation(doc_id="AAPL_2023_10K", page=5, quote="Revenue was $394B")]
    )
    a = Answer(text=ca.text, citations=ca.citations)
    assert len(a.citations) == 1
    assert a.citations[0].doc_id == "AAPL_2023_10K"
