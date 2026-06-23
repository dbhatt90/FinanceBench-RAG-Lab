from rag_hub.eval.generation_metrics import GenerationMetrics


def test_rouge_perfect_match():
    m = GenerationMetrics()
    scores = m.rouge_scores("Revenue was $394B.", "Revenue was $394B.")
    assert scores["rouge1"] == 1.0
    assert scores["rougeL"] == 1.0


def test_rouge_partial_match():
    m = GenerationMetrics()
    scores = m.rouge_scores("Revenue was high.", "Revenue was $394B in fiscal 2022.")
    assert 0.0 < scores["rouge1"] < 1.0


def test_rouge_keys():
    m = GenerationMetrics()
    assert set(m.rouge_scores("a b c.", "a b c.").keys()) == {"rouge1", "rouge2", "rougeL"}
