from unittest.mock import MagicMock, patch
from rag_hub.generation.schemas import Answer, Citation, CitedAnswer
from rag_hub.generation.citation_generator import CitationAwareGenerator


SAMPLE_CHUNKS = [
    {"doc_name": "AAPL_2023_10K", "page": 5, "text": "Net revenue was $394.3 billion for fiscal 2022."},
    {"doc_name": "AAPL_2023_10K", "page": 8, "text": "iPhone revenue totalled $205.5 billion."},
]


def _make_cited_answer():
    return CitedAnswer(
        text="Apple's net revenue was $394.3 billion in fiscal 2022.",
        citations=[Citation(doc_id="AAPL_2023_10K", page=5, quote="Net revenue was $394.3 billion")]
    )


def test_generate_returns_answer_with_citations():
    with patch("rag_hub.generation.citation_generator.ChatVertexAI") as MockLLM:
        mock_chain_output = _make_cited_answer()
        mock_instance = MagicMock()
        mock_instance.with_structured_output.return_value.__or__ = MagicMock(
            return_value=MagicMock(invoke=MagicMock(return_value=mock_chain_output))
        )
        MockLLM.return_value = mock_instance

        gen = CitationAwareGenerator.__new__(CitationAwareGenerator)
        gen._chain = MagicMock(invoke=MagicMock(return_value=mock_chain_output))
        gen._fallback_chain = MagicMock()

        answer = gen.generate("What was Apple's revenue?", SAMPLE_CHUNKS)

    assert isinstance(answer, Answer)
    assert "394" in answer.text
    assert len(answer.citations) == 1
    assert answer.citations[0].doc_id == "AAPL_2023_10K"


def test_generate_falls_back_on_structured_output_failure():
    gen = CitationAwareGenerator.__new__(CitationAwareGenerator)
    gen._chain = MagicMock(invoke=MagicMock(side_effect=Exception("structured output failed")))
    fallback_response = MagicMock()
    fallback_response.content = "Apple's net revenue was $394.3 billion."
    gen._fallback_chain = MagicMock(invoke=MagicMock(return_value=fallback_response))

    answer = gen.generate("What was Apple's revenue?", SAMPLE_CHUNKS)

    assert isinstance(answer, Answer)
    assert "394" in answer.text
    assert answer.citations == []


def test_format_context_includes_doc_id_and_page():
    gen = CitationAwareGenerator.__new__(CitationAwareGenerator)
    ctx = gen._format_context(SAMPLE_CHUNKS)
    assert "doc_id=AAPL_2023_10K" in ctx
    assert "page=5" in ctx
    assert "Net revenue" in ctx
