from unittest.mock import MagicMock
from rag_hub.generation.schemas import Answer, Citation
from rag_hub.generation.corrective_generator import CorrectiveGenerator

CHUNKS = [{"doc_name": "AAPL", "page": 5, "text": "Revenue was $394B."}]


def _make_answer(h_rate: float) -> Answer:
    return Answer(
        text="Revenue was $394B.",
        citations=[Citation(doc_id="AAPL", page=5, quote="Revenue was $394B.")],
        confidence=0.9,
        hallucination_rate=h_rate,
    )


def _make_gen(h_rates: list) -> CorrectiveGenerator:
    answers = [_make_answer(r) for r in h_rates]
    idx = {"n": 0}

    def fake_generate(q, docs):
        a = answers[min(idx["n"], len(answers) - 1)]
        idx["n"] += 1
        return a

    cg = MagicMock(); cg.generate.side_effect = fake_generate
    sr = MagicMock(); sr.score.return_value = 0.9
    det = MagicMock(); det.threshold = 0.25
    det.check.side_effect = lambda a: (a.hallucination_rate, ["ENTAILMENT"])

    return CorrectiveGenerator(
        citation_gen=cg, self_rag=sr, detector=det,
        retriever=None, embed_fn=None,
        max_iterations=2, hallucination_threshold=0.25, initial_top_k=5,
    )


def test_returns_immediately_under_threshold():
    assert _make_gen([0.1]).generate("Q?", CHUNKS).hallucination_rate == 0.1


def test_iterates_when_above_threshold():
    answer = _make_gen([0.8, 0.2]).generate("Q?", CHUNKS)
    assert answer.hallucination_rate == 0.2
    assert answer.generation_iterations == 2


def test_returns_best_answer():
    answer = _make_gen([0.8, 0.9]).generate("Q?", CHUNKS)
    assert answer.hallucination_rate == 0.8


def test_calls_retriever_on_second_iteration():
    cg = MagicMock()
    cg.generate.side_effect = [_make_answer(0.8), _make_answer(0.1)]
    sr = MagicMock(); sr.score.return_value = 0.9
    det = MagicMock(); det.threshold = 0.25
    det.check.side_effect = [(0.8, ["CONTRADICTION"]), (0.1, ["ENTAILMENT"])]
    retriever = MagicMock(); retriever.search.return_value = CHUNKS * 2
    embed_fn = MagicMock(return_value=[0.1] * 768)

    gen = CorrectiveGenerator(
        citation_gen=cg, self_rag=sr, detector=det,
        retriever=retriever, embed_fn=embed_fn,
        max_iterations=2, hallucination_threshold=0.25, initial_top_k=5,
    )
    answer = gen.generate("Q?", CHUNKS)
    retriever.search.assert_called_once_with("Q?", [0.1] * 768, top_k=10)
    assert answer.hallucination_rate == 0.1
