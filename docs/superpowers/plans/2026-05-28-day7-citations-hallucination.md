# Day 7 — Citations, Hallucination Detection, Corrective Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the basic `GeminiFlashGenerator` with a citation-enforced, hallucination-checked, self-correcting generator; add ROUGE/BERTScore/RAGAS generation metrics; produce `eval_results/day_07_results.md`.

**Architecture:** A new `Day7Pipeline` bypasses the LangGraph router and runs: `HybridRRFRetriever → BGEReranker → CRAGEvaluator → CorrectiveGenerator`. `CorrectiveGenerator` wraps `CitationAwareGenerator` (Pydantic structured output) + `SelfRAGScorer` (LLM-as-judge) + `HallucinationDetector` (NLI cross-encoder). If hallucination rate exceeds threshold, it re-retrieves with 2× top-k and regenerates (max 2 iterations, keeps the best result).

**Tech Stack:** `langchain-google-vertexai` (ChatVertexAI.with_structured_output), `transformers` NLI pipeline (`cross-encoder/nli-deberta-v3-base`), `rouge-score`, `bert-score`, `ragas`, `pydantic v2`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/rag_hub/generation/schemas.py` | Pydantic models: `Citation`, `CitedAnswer`, `Answer` |
| Create | `src/rag_hub/generation/citation_generator.py` | `CitationAwareGenerator` — structured output with citation enforcement |
| Create | `src/rag_hub/generation/self_rag.py` | `SelfRAGScorer` — LLM-as-judge confidence scoring |
| Create | `src/rag_hub/generation/hallucination_detector.py` | `HallucinationDetector` — NLI entailment per sentence |
| Create | `src/rag_hub/generation/corrective_generator.py` | `CorrectiveGenerator` — orchestrates the corrective loop |
| Create | `src/rag_hub/generation/day7_pipeline.py` | `Day7Pipeline` — direct retrieval (no router) |
| Create | `src/rag_hub/eval/generation_metrics.py` | `GenerationMetrics` — ROUGE, BERTScore, RAGAS |
| Create | `scripts/day7_eval.py` | Eval runner — 50 FinanceBench Q, writes day_07_results.md |
| Create | `tests/generation/test_schemas.py` | Unit tests for Pydantic models |
| Create | `tests/generation/test_citation_generator.py` | Mocked LLM tests for CitationAwareGenerator |
| Create | `tests/generation/test_self_rag.py` | Mocked LLM tests for SelfRAGScorer |
| Create | `tests/generation/test_hallucination_detector.py` | Mocked NLI pipeline tests |
| Create | `tests/generation/test_corrective_generator.py` | Integration tests with stubs |
| Create | `tests/eval/test_generation_metrics.py` | Unit tests for ROUGE/BERTScore |

---

## Task 1: Pydantic Schemas

**Files:**
- Create: `src/rag_hub/generation/schemas.py`
- Create: `tests/generation/__init__.py` (empty)
- Create: `tests/generation/test_schemas.py`

- [ ] **Step 1.1: Write the failing test**

```python
# tests/generation/test_schemas.py
import pytest
from pydantic import ValidationError
from rag_hub.generation.schemas import Citation, CitedAnswer, Answer


def test_citation_requires_doc_id_page_quote():
    c = Citation(doc_id="AAPL_2023_10K", page=5, quote="Revenue was $394B")
    assert c.doc_id == "AAPL_2023_10K"
    assert c.page == 5
    assert c.quote == "Revenue was $394B"


def test_citation_quote_max_200_chars():
    with pytest.raises(ValidationError):
        Citation(doc_id="X", page=1, quote="A" * 201)


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
        Answer(text="x", citations=[], hallucination_rate=-0.1)


def test_answer_from_cited_answer():
    ca = CitedAnswer(
        text="Revenue was $394B.",
        citations=[Citation(doc_id="AAPL_2023_10K", page=5, quote="Revenue was $394B")]
    )
    a = Answer(text=ca.text, citations=ca.citations)
    assert len(a.citations) == 1
    assert a.citations[0].doc_id == "AAPL_2023_10K"
```

- [ ] **Step 1.2: Run test to confirm it fails**

```bash
cd /Users/devanshbhatt/Desktop/AI_projects/RAG
pytest tests/generation/test_schemas.py -v
```
Expected: `ModuleNotFoundError` or `ImportError` for `rag_hub.generation.schemas`

- [ ] **Step 1.3: Create the schemas module**

```python
# src/rag_hub/generation/schemas.py
from typing import List
from pydantic import BaseModel, Field


class Citation(BaseModel):
    doc_id: str = Field(..., description="doc_name from the chunk payload")
    page: int = Field(..., description="0-indexed page number")
    quote: str = Field(..., max_length=200, description="Verbatim excerpt from the source")


class CitedAnswer(BaseModel):
    """Structured output schema returned by CitationAwareGenerator's LLM call."""
    text: str = Field(..., description="Concise answer to the question")
    citations: List[Citation] = Field(
        default_factory=list,
        description="Source passages that support the answer"
    )


class Answer(BaseModel):
    """Enriched answer returned by CorrectiveGenerator with quality scores."""
    text: str
    citations: List[Citation] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    hallucination_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    generation_iterations: int = Field(default=1, ge=1)
```

- [ ] **Step 1.4: Create `tests/generation/__init__.py`**

```python
# tests/generation/__init__.py
```

- [ ] **Step 1.5: Run tests to confirm pass**

```bash
pytest tests/generation/test_schemas.py -v
```
Expected: 6 tests PASSED

- [ ] **Step 1.6: Commit**

```bash
git add src/rag_hub/generation/schemas.py tests/generation/__init__.py tests/generation/test_schemas.py
git commit -m "feat(day-07): add Citation/CitedAnswer/Answer Pydantic schemas"
```

---

## Task 2: CitationAwareGenerator

**Files:**
- Create: `src/rag_hub/generation/citation_generator.py`
- Create: `tests/generation/test_citation_generator.py`

- [ ] **Step 2.1: Write the failing test**

```python
# tests/generation/test_citation_generator.py
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
```

- [ ] **Step 2.2: Run test to confirm it fails**

```bash
pytest tests/generation/test_citation_generator.py -v
```
Expected: `ImportError` — `citation_generator` doesn't exist yet

- [ ] **Step 2.3: Implement CitationAwareGenerator**

```python
# src/rag_hub/generation/citation_generator.py
import os
from typing import List, Dict

import vertexai
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from google.oauth2 import service_account
from dotenv import load_dotenv

from rag_hub.generation.schemas import Answer, CitedAnswer

load_dotenv()

_credentials = service_account.Credentials.from_service_account_file(
    os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)
vertexai.init(
    project=os.getenv("GCP_PROJECT_ID"),
    location=os.getenv("GCP_LOCATION", "us-central1"),
    credentials=_credentials,
)

_PROMPT = ChatPromptTemplate.from_template(
    """You are a financial analyst answering questions from SEC filings.

Use ONLY the numbered context passages below. For each claim in your answer,
cite the passage by including its doc_id, page number, and a verbatim quote
(max 200 characters) from that passage.

If the answer is not present in the context, set text to "I don't know" and
return an empty citations list.

Context:
{context}

Question: {question}"""
)


class CitationAwareGenerator:
    """
    Generates structured answers with citation attribution using Gemini's
    structured output (function-calling) to enforce the CitedAnswer schema.

    Falls back to plain text generation if structured output fails.
    """

    def __init__(self, model: str = "gemini-2.5-flash"):
        base_llm = ChatVertexAI(
            model_name=model,
            temperature=0,
            project=os.getenv("GCP_PROJECT_ID"),
            location=os.getenv("GCP_LOCATION", "us-central1"),
            credentials=_credentials,
        )
        self._chain = _PROMPT | base_llm.with_structured_output(CitedAnswer)
        self._fallback_chain = _PROMPT | base_llm

    def _format_context(self, chunks: List[Dict]) -> str:
        lines = []
        for i, c in enumerate(chunks, 1):
            lines.append(
                f"[{i}] doc_id={c['doc_name']}  page={c['page']}\n{c['text']}"
            )
        return "\n\n".join(lines)

    def generate(self, question: str, chunks: List[Dict]) -> Answer:
        """Returns an Answer with text + citations. Falls back to no citations on LLM error."""
        context = self._format_context(chunks)
        try:
            cited: CitedAnswer = self._chain.invoke(
                {"context": context, "question": question}
            )
            return Answer(text=cited.text, citations=cited.citations)
        except Exception as e:
            print(f"[CitationGen] structured output failed ({e}), using fallback")
            response = self._fallback_chain.invoke(
                {"context": context, "question": question}
            )
            text = getattr(response, "content", str(response)).strip()
            return Answer(text=text, citations=[])
```

- [ ] **Step 2.4: Run tests to confirm pass**

```bash
pytest tests/generation/test_citation_generator.py -v
```
Expected: 3 tests PASSED

- [ ] **Step 2.5: Commit**

```bash
git add src/rag_hub/generation/citation_generator.py tests/generation/test_citation_generator.py
git commit -m "feat(day-07): CitationAwareGenerator with structured output and fallback"
```

---

## Task 3: SelfRAGScorer

**Files:**
- Create: `src/rag_hub/generation/self_rag.py`
- Create: `tests/generation/test_self_rag.py`

- [ ] **Step 3.1: Write the failing test**

```python
# tests/generation/test_self_rag.py
from unittest.mock import MagicMock
from rag_hub.generation.schemas import Answer, Citation
from rag_hub.generation.self_rag import SelfRAGScorer, _ScoreResult

CHUNKS = [
    {"doc_name": "AAPL_2023_10K", "page": 5, "text": "Net revenue was $394.3 billion for fiscal 2022."},
]
ANSWER = Answer(
    text="Apple's net revenue was $394.3 billion in fiscal 2022.",
    citations=[Citation(doc_id="AAPL_2023_10K", page=5, quote="Net revenue was $394.3 billion")],
)


def test_score_returns_float_between_0_and_1():
    scorer = SelfRAGScorer.__new__(SelfRAGScorer)
    mock_result = _ScoreResult(
        is_relevant=True, is_supported=True, is_useful=True,
        confidence=0.92, reasoning="Answer directly matches context."
    )
    scorer._chain = MagicMock(invoke=MagicMock(return_value=mock_result))

    score = scorer.score("What was Apple's revenue?", ANSWER, CHUNKS)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert score == 0.92


def test_score_falls_back_to_0_5_on_error():
    scorer = SelfRAGScorer.__new__(SelfRAGScorer)
    scorer._chain = MagicMock(invoke=MagicMock(side_effect=Exception("LLM error")))

    score = scorer.score("What was Apple's revenue?", ANSWER, CHUNKS)

    assert score == 0.5


def test_score_result_fields():
    r = _ScoreResult(
        is_relevant=True, is_supported=False, is_useful=True,
        confidence=0.3, reasoning="Not fully supported."
    )
    assert r.confidence == 0.3
    assert not r.is_supported
```

- [ ] **Step 3.2: Run test to confirm failure**

```bash
pytest tests/generation/test_self_rag.py -v
```
Expected: `ImportError`

- [ ] **Step 3.3: Implement SelfRAGScorer**

```python
# src/rag_hub/generation/self_rag.py
import os
from typing import List, Dict

import vertexai
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from google.oauth2 import service_account
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from rag_hub.generation.schemas import Answer

load_dotenv()

_credentials = service_account.Credentials.from_service_account_file(
    os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)
vertexai.init(
    project=os.getenv("GCP_PROJECT_ID"),
    location=os.getenv("GCP_LOCATION", "us-central1"),
    credentials=_credentials,
)


class _ScoreResult(BaseModel):
    is_relevant: bool = Field(..., description="Does the answer address the question?")
    is_supported: bool = Field(..., description="Are all claims supported by the context?")
    is_useful: bool = Field(..., description="Is the answer specific and non-vague?")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall support score")
    reasoning: str = Field(..., description="One-sentence explanation")


_PROMPT = ChatPromptTemplate.from_template(
    """You are evaluating a RAG system's answer against the retrieved context.

Question: {question}

Answer: {answer}

Context (retrieved passages):
{context}

Evaluate the answer on three criteria:
1. is_relevant: Does it directly address the question?
2. is_supported: Are ALL factual claims backed by the context passages?
3. is_useful: Is it specific and actionable (not "I don't know" or overly vague)?
4. confidence: Float 0.0–1.0. 1.0 = every claim is explicitly in the context. \
0.0 = no claim is supported.
5. reasoning: One sentence explaining the score."""
)


class SelfRAGScorer:
    """
    LLM-as-judge: evaluates whether an answer is grounded in the retrieved context.
    Returns a confidence score in [0, 1] that gets stored in Answer.confidence.
    """

    def __init__(self, model: str = "gemini-2.5-flash"):
        llm = ChatVertexAI(
            model_name=model,
            temperature=0,
            project=os.getenv("GCP_PROJECT_ID"),
            location=os.getenv("GCP_LOCATION", "us-central1"),
            credentials=_credentials,
        )
        self._chain = _PROMPT | llm.with_structured_output(_ScoreResult)

    def score(self, question: str, answer: Answer, chunks: List[Dict]) -> float:
        """Scores the answer's faithfulness to the context. Returns 0.5 on error."""
        context = "\n\n".join(
            f"[{c['doc_name']} p.{c['page']}] {c['text']}" for c in chunks[:5]
        )
        try:
            result: _ScoreResult = self._chain.invoke(
                {"question": question, "answer": answer.text, "context": context}
            )
            return result.confidence
        except Exception as e:
            print(f"[SelfRAG] scoring failed ({e}), defaulting to 0.5")
            return 0.5
```

- [ ] **Step 3.4: Run tests to confirm pass**

```bash
pytest tests/generation/test_self_rag.py -v
```
Expected: 3 tests PASSED

- [ ] **Step 3.5: Commit**

```bash
git add src/rag_hub/generation/self_rag.py tests/generation/test_self_rag.py
git commit -m "feat(day-07): SelfRAGScorer — LLM-as-judge faithfulness scoring"
```

---

## Task 4: HallucinationDetector

**Files:**
- Create: `src/rag_hub/generation/hallucination_detector.py`
- Create: `tests/generation/test_hallucination_detector.py`

- [ ] **Step 4.1: Write the failing test**

```python
# tests/generation/test_hallucination_detector.py
from unittest.mock import MagicMock, patch
from rag_hub.generation.schemas import Answer, Citation
from rag_hub.generation.hallucination_detector import HallucinationDetector, _split_sentences

# --- Unit tests for sentence splitter ---

def test_split_sentences_basic():
    text = "Revenue was $394B. iPhone segment grew 5%. Services hit an all-time high."
    sentences = _split_sentences(text)
    assert len(sentences) == 3
    assert all(len(s) > 10 for s in sentences)


def test_split_sentences_filters_short():
    text = "OK. Revenue was $394B in fiscal 2022."
    sentences = _split_sentences(text)
    assert len(sentences) == 1  # "OK." is < 10 chars, filtered out


# --- Integration tests with mocked NLI pipeline ---

def _make_detector_with_mock_pipe(label: str) -> HallucinationDetector:
    """Returns a detector whose NLI pipe always returns the given label."""
    detector = HallucinationDetector.__new__(HallucinationDetector)
    detector.threshold = 0.25
    # NLI pipeline returns [[{"label": ..., "score": 0.9}, ...]] per input
    detector._pipe = MagicMock(
        return_value=[[{"label": label, "score": 0.9}, {"label": "OTHER", "score": 0.1}]]
    )
    return detector


def test_entailed_answer_has_zero_hallucination():
    answer = Answer(
        text="Revenue was $394B in fiscal 2022. iPhone grew 5%.",
        citations=[Citation(doc_id="AAPL_10K", page=5, quote="Revenue was $394B in fiscal 2022")],
    )
    detector = _make_detector_with_mock_pipe("ENTAILMENT")
    rate, labels = detector.check(answer)
    assert rate == 0.0
    assert all(l == "ENTAILMENT" for l in labels)


def test_contradicted_answer_has_full_hallucination():
    answer = Answer(
        text="Revenue was $394B in fiscal 2022. iPhone grew 5%.",
        citations=[Citation(doc_id="AAPL_10K", page=5, quote="Revenue was $394B in fiscal 2022")],
    )
    detector = _make_detector_with_mock_pipe("CONTRADICTION")
    rate, labels = detector.check(answer)
    assert rate == 1.0


def test_no_citation_returns_full_hallucination():
    answer = Answer(text="Revenue was $394B.", citations=[])
    detector = HallucinationDetector.__new__(HallucinationDetector)
    detector.threshold = 0.25
    detector._pipe = MagicMock()  # should not be called

    rate, labels = detector.check(answer)
    assert rate == 1.0
    detector._pipe.assert_not_called()


def test_empty_text_returns_zero_rate():
    answer = Answer(text="", citations=[Citation(doc_id="X", page=1, quote="something")])
    detector = HallucinationDetector.__new__(HallucinationDetector)
    detector.threshold = 0.25
    detector._pipe = MagicMock()
    rate, labels = detector.check(answer)
    assert rate == 0.0
    assert labels == []
```

- [ ] **Step 4.2: Run test to confirm failure**

```bash
pytest tests/generation/test_hallucination_detector.py -v
```
Expected: `ImportError`

- [ ] **Step 4.3: Implement HallucinationDetector**

```python
# src/rag_hub/generation/hallucination_detector.py
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
    Sentence-level hallucination detection via NLI entailment.

    For each sentence in the answer, uses the cited quote(s) as the NLI premise
    and the sentence as the hypothesis. Labels NEUTRAL or CONTRADICTION are
    treated as hallucinated.

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
        Returns (hallucination_rate, per_sentence_label_list).

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
```

- [ ] **Step 4.4: Run tests to confirm pass**

```bash
pytest tests/generation/test_hallucination_detector.py -v
```
Expected: 6 tests PASSED

- [ ] **Step 4.5: Commit**

```bash
git add src/rag_hub/generation/hallucination_detector.py tests/generation/test_hallucination_detector.py
git commit -m "feat(day-07): HallucinationDetector — NLI sentence-level entailment check"
```

---

## Task 5: CorrectiveGenerator

**Files:**
- Create: `src/rag_hub/generation/corrective_generator.py`
- Create: `tests/generation/test_corrective_generator.py`

- [ ] **Step 5.1: Write the failing test**

```python
# tests/generation/test_corrective_generator.py
from unittest.mock import MagicMock
from rag_hub.generation.schemas import Answer, Citation
from rag_hub.generation.corrective_generator import CorrectiveGenerator

CHUNKS = [{"doc_name": "AAPL", "page": 5, "text": "Revenue was $394B."}]

def _make_answer(hallucination_rate: float) -> Answer:
    return Answer(
        text="Revenue was $394B.",
        citations=[Citation(doc_id="AAPL", page=5, quote="Revenue was $394B.")],
        confidence=0.9,
        hallucination_rate=hallucination_rate,
    )


def _make_generator(h_rates: list) -> CorrectiveGenerator:
    """Stub generator that produces answers with given hallucination rates in sequence."""
    answers = [_make_answer(r) for r in h_rates]
    call_count = {"n": 0}

    def fake_cite_generate(q, docs):
        a = answers[min(call_count["n"], len(answers) - 1)]
        call_count["n"] += 1
        return a

    citation_gen = MagicMock()
    citation_gen.generate.side_effect = fake_cite_generate

    self_rag = MagicMock()
    self_rag.score.return_value = 0.9

    detector = MagicMock()
    detector.threshold = 0.25
    detector.check.side_effect = lambda a: (a.hallucination_rate, ["ENTAILMENT"])

    return CorrectiveGenerator(
        citation_gen=citation_gen,
        self_rag=self_rag,
        detector=detector,
        retriever=None,
        embed_fn=None,
        max_iterations=2,
        hallucination_threshold=0.25,
        initial_top_k=5,
    )


def test_returns_immediately_when_under_threshold():
    gen = _make_generator([0.1])
    answer = gen.generate("Q?", CHUNKS)
    assert answer.hallucination_rate == 0.1
    assert answer.generation_iterations == 1


def test_iterates_when_above_threshold_no_retriever():
    """Without a retriever, still tries max_iterations with same docs."""
    gen = _make_generator([0.8, 0.2])
    answer = gen.generate("Q?", CHUNKS)
    # Should pick the best answer (0.2)
    assert answer.hallucination_rate == 0.2
    assert answer.generation_iterations == 2


def test_returns_best_answer_across_iterations():
    gen = _make_generator([0.8, 0.9])  # both bad, returns first (0.8 < 0.9)
    answer = gen.generate("Q?", CHUNKS)
    assert answer.hallucination_rate == 0.8


def test_calls_retriever_on_second_iteration():
    citation_gen = MagicMock()
    citation_gen.generate.side_effect = [_make_answer(0.8), _make_answer(0.1)]

    self_rag = MagicMock()
    self_rag.score.return_value = 0.9

    detector = MagicMock()
    detector.threshold = 0.25
    detector.check.side_effect = [
        (0.8, ["CONTRADICTION"]),
        (0.1, ["ENTAILMENT"]),
    ]

    expanded_docs = [{"doc_name": "AAPL", "page": 6, "text": "More context."}] * 10
    retriever = MagicMock()
    retriever.search.return_value = expanded_docs
    embed_fn = MagicMock(return_value=[0.1] * 768)

    gen = CorrectiveGenerator(
        citation_gen=citation_gen,
        self_rag=self_rag,
        detector=detector,
        retriever=retriever,
        embed_fn=embed_fn,
        max_iterations=2,
        hallucination_threshold=0.25,
        initial_top_k=5,
    )
    answer = gen.generate("Q?", CHUNKS)

    retriever.search.assert_called_once_with("Q?", [0.1] * 768, top_k=10)
    assert answer.hallucination_rate == 0.1
```

- [ ] **Step 5.2: Run test to confirm failure**

```bash
pytest tests/generation/test_corrective_generator.py -v
```
Expected: `ImportError`

- [ ] **Step 5.3: Implement CorrectiveGenerator**

```python
# src/rag_hub/generation/corrective_generator.py
from typing import List, Dict, Optional, Callable

from rag_hub.generation.schemas import Answer
from rag_hub.generation.citation_generator import CitationAwareGenerator
from rag_hub.generation.self_rag import SelfRAGScorer
from rag_hub.generation.hallucination_detector import HallucinationDetector


class CorrectiveGenerator:
    """
    Orchestrates a corrective generation loop:
      1. Generate a cited answer
      2. Score with Self-RAG (LLM judge)
      3. Check for hallucinations via NLI
      4. If hallucination_rate > threshold and iterations remain:
           expand retrieval (2× top-k) and regenerate
      5. Return the answer with the lowest hallucination rate seen

    Requires a retriever and embed_fn for re-retrieval. If not provided,
    still retries with the same docs (in case the LLM produces a better
    answer on a second attempt).
    """

    def __init__(
        self,
        citation_gen: CitationAwareGenerator,
        self_rag: SelfRAGScorer,
        detector: HallucinationDetector,
        retriever=None,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        max_iterations: int = 2,
        hallucination_threshold: float = 0.25,
        initial_top_k: int = 5,
    ):
        self.citation_gen = citation_gen
        self.self_rag = self_rag
        self.detector = detector
        self.retriever = retriever
        self.embed_fn = embed_fn
        self.max_iterations = max_iterations
        self.hallucination_threshold = hallucination_threshold
        self.initial_top_k = initial_top_k

    def generate(self, question: str, docs: List[Dict]) -> Answer:
        best: Optional[Answer] = None
        current_docs = docs

        for iteration in range(1, self.max_iterations + 1):
            answer = self.citation_gen.generate(question, current_docs)
            answer.confidence = self.self_rag.score(question, answer, current_docs)
            rate, _labels = self.detector.check(answer)
            answer.hallucination_rate = rate
            answer.generation_iterations = iteration

            if best is None or rate < best.hallucination_rate:
                best = answer

            if rate <= self.hallucination_threshold:
                break

            if iteration < self.max_iterations and self.retriever and self.embed_fn:
                expanded_k = self.initial_top_k * 2
                query_vec = self.embed_fn(question)
                current_docs = self.retriever.search(question, query_vec, top_k=expanded_k)

        return best
```

- [ ] **Step 5.4: Run tests to confirm pass**

```bash
pytest tests/generation/test_corrective_generator.py -v
```
Expected: 4 tests PASSED

- [ ] **Step 5.5: Commit**

```bash
git add src/rag_hub/generation/corrective_generator.py tests/generation/test_corrective_generator.py
git commit -m "feat(day-07): CorrectiveGenerator — hallucination-driven corrective loop"
```

---

## Task 6: GenerationMetrics

**Files:**
- Create: `src/rag_hub/eval/generation_metrics.py`
- Create: `tests/eval/test_generation_metrics.py`

- [ ] **Step 6.1: Write the failing test**

```python
# tests/eval/test_generation_metrics.py
from rag_hub.eval.generation_metrics import GenerationMetrics


def test_rouge_scores_perfect_match():
    m = GenerationMetrics()
    scores = m.rouge_scores("Revenue was $394B.", "Revenue was $394B.")
    assert scores["rouge1"] == 1.0
    assert scores["rouge2"] == 1.0
    assert scores["rougeL"] == 1.0


def test_rouge_scores_partial_match():
    m = GenerationMetrics()
    scores = m.rouge_scores("Revenue was high.", "Revenue was $394B in fiscal 2022.")
    # Some overlap but not perfect
    assert 0.0 < scores["rouge1"] < 1.0


def test_rouge_scores_no_match():
    m = GenerationMetrics()
    scores = m.rouge_scores("The sky is blue.", "Revenue was $394B.")
    # Very low overlap
    assert scores["rouge1"] < 0.3


def test_rouge_scores_keys():
    m = GenerationMetrics()
    scores = m.rouge_scores("test sentence here.", "test sentence here.")
    assert set(scores.keys()) == {"rouge1", "rouge2", "rougeL"}
    assert all(isinstance(v, float) for v in scores.values())
```

- [ ] **Step 6.2: Run test to confirm failure**

```bash
pytest tests/eval/test_generation_metrics.py -v
```
Expected: `ImportError` — `generation_metrics` doesn't exist

- [ ] **Step 6.3: Add `rouge-score` dependency**

```bash
uv add rouge-score bert-score sentencepiece
```

- [ ] **Step 6.4: Implement GenerationMetrics**

```python
# src/rag_hub/eval/generation_metrics.py
"""
Generation quality metrics for Day 7+.

ROUGE and BERTScore run locally (no API).
RAGAS metrics require a live LLM API call — call sparingly.
"""

from typing import Dict, List, Optional

from rouge_score import rouge_scorer


class GenerationMetrics:
    """
    Computes generation quality metrics.

    ROUGE: lexical overlap. Cheap, fast, no network.
    BERTScore: semantic similarity. Requires transformers (~1.5 GB model download on first call).
    RAGAS: faithfulness + answer_relevancy. Requires LLM API calls.
    """

    def __init__(self):
        self._rouge = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

    def rouge_scores(self, prediction: str, reference: str) -> Dict[str, float]:
        """Returns dict with rouge1, rouge2, rougeL F-measures."""
        scores = self._rouge.score(reference, prediction)
        return {
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rouge2": round(scores["rouge2"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4),
        }

    def bert_score(self, prediction: str, reference: str) -> float:
        """Returns BERTScore F1 (semantic similarity). Loads model on first call (~1.5 GB)."""
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
        """
        Runs RAGAS faithfulness + answer_relevancy + answer_correctness.
        Requires LLM API — avoid calling in tight loops on free-tier quotas.
        Returns dict with keys: faithfulness, answer_relevancy, answer_correctness.
        """
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, answer_correctness

        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth],
        }
        dataset = Dataset.from_dict(data)
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, answer_correctness],
        )
        return {
            "faithfulness": round(float(result["faithfulness"]), 4),
            "answer_relevancy": round(float(result["answer_relevancy"]), 4),
            "answer_correctness": round(float(result["answer_correctness"]), 4),
        }
```

- [ ] **Step 6.5: Run tests to confirm pass**

```bash
pytest tests/eval/test_generation_metrics.py -v
```
Expected: 4 tests PASSED

- [ ] **Step 6.6: Commit**

```bash
git add src/rag_hub/eval/generation_metrics.py tests/eval/test_generation_metrics.py pyproject.toml uv.lock
git commit -m "feat(day-07): GenerationMetrics — ROUGE, BERTScore, RAGAS wrappers"
```

---

## Task 7: Day7Pipeline

**Files:**
- Create: `src/rag_hub/generation/day7_pipeline.py`

No new tests needed — Day7Pipeline is a thin orchestration wrapper over already-tested components. The eval script (Task 8) serves as its integration test.

- [ ] **Step 7.1: Implement Day7Pipeline**

```python
# src/rag_hub/generation/day7_pipeline.py
"""
Day 7 RAG pipeline: direct retrieval (no LangGraph router).

Flow: embed query → HybridRRFRetriever → BGEReranker → CRAGEvaluator
      → CorrectiveGenerator → Answer

Bypasses the LangGraph router intentionally — Day 6 results showed direct
hybrid retrieval + BGE reranking already achieves good recall@5, and
the corrective generation loop handles quality correction.
"""

from typing import Dict

from rag_hub.embeddings.gemini_001 import GeminiEmbeddingClient
from rag_hub.vectorstore.qdrant_store import QdrantStore
from rag_hub.retrievers.bm25_retriever import BM25Retriever
from rag_hub.retrievers.hybrid_retriever import HybridRRFRetriever
from rag_hub.rerankers.bge_reranker import BGEReranker
from rag_hub.crag.evaluator import CRAGEvaluator
from rag_hub.generation.schemas import Answer
from rag_hub.generation.citation_generator import CitationAwareGenerator
from rag_hub.generation.self_rag import SelfRAGScorer
from rag_hub.generation.hallucination_detector import HallucinationDetector
from rag_hub.generation.corrective_generator import CorrectiveGenerator


class Day7Pipeline:
    """
    Self-contained Day 7 pipeline.

    Args:
        store:                  Qdrant vector store (already populated).
        bm25:                   BM25Retriever (already built from the same corpus).
        embedder:               GeminiEmbeddingClient for query embedding.
        retrieval_top_k:        Candidate pool size fed into the reranker (default 10).
        rerank_top_k:           Docs passed to CRAG + generator after reranking (default 5).
        crag_threshold:         Confidence below which web fallback would fire (not used
                                here — corrective loop handles quality, not CRAG fallback).
        hallucination_threshold: Fraction of hallucinated sentences that triggers re-retrieval.
    """

    def __init__(
        self,
        store: QdrantStore,
        bm25: BM25Retriever,
        embedder: GeminiEmbeddingClient,
        retrieval_top_k: int = 10,
        rerank_top_k: int = 5,
        crag_threshold: float = 0.5,
        hallucination_threshold: float = 0.25,
    ):
        self.embedder = embedder
        self._retriever = HybridRRFRetriever(store=store, bm25=bm25)
        self._reranker = BGEReranker()
        self._crag = CRAGEvaluator(threshold=crag_threshold)

        self._corrective = CorrectiveGenerator(
            citation_gen=CitationAwareGenerator(),
            self_rag=SelfRAGScorer(),
            detector=HallucinationDetector(threshold=hallucination_threshold),
            retriever=self._retriever,
            embed_fn=self.embedder.embed_query,
            hallucination_threshold=hallucination_threshold,
            initial_top_k=rerank_top_k,
        )
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_k = rerank_top_k

    def run(self, question: str) -> Dict:
        """
        Runs the full pipeline and returns a result dict.

        Keys: question, answer (Answer), docs (List[Dict]),
              crag_confidence (float), crag_labels (List[str]).
        """
        query_vec = self.embedder.embed_query(question)
        docs = self._retriever.search(question, query_vec, top_k=self.retrieval_top_k)
        reranked = self._reranker.rerank(question, docs, top_k=self.rerank_top_k)
        crag_confidence, crag_labels = self._crag.evaluate(question, reranked)

        answer: Answer = self._corrective.generate(question, reranked)

        return {
            "question": question,
            "answer": answer,
            "docs": reranked,
            "crag_confidence": crag_confidence,
            "crag_labels": crag_labels,
        }
```

- [ ] **Step 7.2: Confirm import works**

```bash
cd /Users/devanshbhatt/Desktop/AI_projects/RAG
python -c "from rag_hub.generation.day7_pipeline import Day7Pipeline; print('OK')"
```
Expected: `OK` (no import errors; actual model loading deferred to instantiation)

- [ ] **Step 7.3: Commit**

```bash
git add src/rag_hub/generation/day7_pipeline.py
git commit -m "feat(day-07): Day7Pipeline — direct hybrid retrieval + corrective generation"
```

---

## Task 8: Eval Script + Results

**Files:**
- Create: `scripts/day7_eval.py`
- Create: `eval_results/day_07_results.md` (written by the script)

- [ ] **Step 8.1: Implement the eval script**

```python
# scripts/day7_eval.py
"""
Day 7 evaluation: generation quality on FinanceBench.

Runs Day7Pipeline on up to --limit questions and computes:
  - hallucination_rate (NLI-based)
  - Self-RAG confidence
  - ROUGE-L
  - BERTScore F1
  - RAGAS faithfulness + answer_relevancy (sampled subset, API-expensive)

Writes: eval_results/day_07_results.md
"""

import sys
import os
import json
import argparse
from datetime import datetime
from typing import List, Dict
from statistics import mean, stdev

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag_hub.eval.financebench import load_questions
from rag_hub.eval.generation_metrics import GenerationMetrics
from rag_hub.embeddings.gemini_001 import GeminiEmbeddingClient
from rag_hub.vectorstore.qdrant_store import QdrantStore
from rag_hub.retrievers.bm25_retriever import BM25Retriever
from rag_hub.generation.day7_pipeline import Day7Pipeline

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SMOKE_PATH = "data/eval/smoke_50.jsonl"
RESULTS_DIR = "eval_results"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = "financebench_v1"
BM25_PATH = "data/processed/bm25_index.pkl"

# RAGAS is expensive (LLM call per question). Limit to first N questions.
RAGAS_LIMIT = 10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=50, help="Number of questions to evaluate")
    p.add_argument("--no-ragas", action="store_true", help="Skip RAGAS metrics (faster)")
    return p.parse_args()


def run_eval(questions: List[Dict], pipeline: Day7Pipeline, metrics: GenerationMetrics, run_ragas: bool, ragas_limit: int):
    rows = []
    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] {q['question'][:70]}...")
        result = pipeline.run(q["question"])
        answer = result["answer"]
        prediction = answer.text
        reference = q.get("answer", "")

        rouge = metrics.rouge_scores(prediction, reference)

        row = {
            "question": q["question"],
            "prediction": prediction,
            "reference": reference,
            "hallucination_rate": answer.hallucination_rate,
            "confidence": answer.confidence,
            "generation_iterations": answer.generation_iterations,
            "num_citations": len(answer.citations),
            "rouge1": rouge["rouge1"],
            "rouge2": rouge["rouge2"],
            "rougeL": rouge["rougeL"],
            "bert_score": None,
            "ragas_faithfulness": None,
            "ragas_answer_relevancy": None,
        }

        # BERTScore — heavier, skip if answer is "I don't know"
        if prediction.lower() != "i don't know" and reference:
            try:
                row["bert_score"] = metrics.bert_score(prediction, reference)
            except Exception as e:
                print(f"  [WARN] BERTScore failed: {e}")

        # RAGAS — LLM calls, limited sample
        if run_ragas and i < ragas_limit and reference:
            contexts = [d.get("text", "") for d in result["docs"][:5]]
            try:
                ragas = metrics.ragas_metrics(q["question"], prediction, contexts, reference)
                row["ragas_faithfulness"] = ragas["faithfulness"]
                row["ragas_answer_relevancy"] = ragas["answer_relevancy"]
            except Exception as e:
                print(f"  [WARN] RAGAS failed: {e}")

        rows.append(row)

    return rows


def write_results(rows: List[Dict], args):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    def safe_mean(vals):
        v = [x for x in vals if x is not None]
        return round(mean(v), 4) if v else None

    def safe_std(vals):
        v = [x for x in vals if x is not None]
        return round(stdev(v), 4) if len(v) > 1 else None

    corrective_fired = sum(1 for r in rows if r["generation_iterations"] > 1)

    lines = [
        f"# Day 7 Eval Results — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"**Questions evaluated:** {len(rows)}",
        f"**Corrective loop fired:** {corrective_fired}/{len(rows)} ({100*corrective_fired/len(rows):.1f}%)",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Mean | Std |",
        "|--------|------|-----|",
        f"| Hallucination Rate | {safe_mean([r['hallucination_rate'] for r in rows])} | {safe_std([r['hallucination_rate'] for r in rows])} |",
        f"| Self-RAG Confidence | {safe_mean([r['confidence'] for r in rows])} | {safe_std([r['confidence'] for r in rows])} |",
        f"| ROUGE-1 | {safe_mean([r['rouge1'] for r in rows])} | {safe_std([r['rouge1'] for r in rows])} |",
        f"| ROUGE-2 | {safe_mean([r['rouge2'] for r in rows])} | {safe_std([r['rouge2'] for r in rows])} |",
        f"| ROUGE-L | {safe_mean([r['rougeL'] for r in rows])} | {safe_std([r['rougeL'] for r in rows])} |",
        f"| BERTScore F1 | {safe_mean([r['bert_score'] for r in rows])} | {safe_std([r['bert_score'] for r in rows])} |",
        f"| RAGAS Faithfulness (n={RAGAS_LIMIT}) | {safe_mean([r['ragas_faithfulness'] for r in rows])} | — |",
        f"| RAGAS Answer Relevancy (n={RAGAS_LIMIT}) | {safe_mean([r['ragas_answer_relevancy'] for r in rows])} | — |",
        "",
        "## Citation Coverage",
        "",
        f"- Questions with ≥1 citation: {sum(1 for r in rows if r['num_citations'] > 0)}/{len(rows)}",
        f"- Average citations per answer: {safe_mean([r['num_citations'] for r in rows])}",
        "",
        "## Example Answer (first question)",
        "",
    ]

    if rows:
        ex = rows[0]
        lines += [
            f"**Question:** {ex['question']}",
            "",
            f"**Answer:** {ex['prediction']}",
            "",
            f"**Reference:** {ex['reference']}",
            "",
            f"**Metrics:** hallucination_rate={ex['hallucination_rate']}, confidence={ex['confidence']}, rouge-L={ex['rougeL']}",
            "",
        ]

    path = os.path.join(RESULTS_DIR, "day_07_results.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))

    # Also dump raw JSON for dashboard
    json_path = os.path.join(RESULTS_DIR, "day_07_raw.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"\nResults written to {path}")
    print(f"Raw JSON: {json_path}")


def main():
    args = parse_args()
    questions = load_questions(SMOKE_PATH)[: args.limit]
    print(f"Loaded {len(questions)} questions from {SMOKE_PATH}")

    embedder = GeminiEmbeddingClient()
    store = QdrantStore(url=QDRANT_URL, collection_name=COLLECTION)
    bm25 = BM25Retriever.load(BM25_PATH)
    pipeline = Day7Pipeline(store=store, bm25=bm25, embedder=embedder)
    metrics = GenerationMetrics()

    rows = run_eval(questions, pipeline, metrics, run_ragas=not args.no_ragas, ragas_limit=RAGAS_LIMIT)
    write_results(rows, args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 8.2: Verify the script loads without errors**

```bash
cd /Users/devanshbhatt/Desktop/AI_projects/RAG
python -c "import scripts.day7_eval" 2>&1 | head -5
```
Expected: no `ImportError` or `SyntaxError`

- [ ] **Step 8.3: Run smoke eval on 3 questions (no RAGAS)**

```bash
python scripts/day7_eval.py --limit 3 --no-ragas
```
Expected: prints per-question progress, writes `eval_results/day_07_results.md`

- [ ] **Step 8.4: Inspect the results file**

```bash
cat eval_results/day_07_results.md
```
Confirm: metrics table present, example answer section populated, citation count > 0 for at least one answer.

- [ ] **Step 8.5: Run full 50-question eval (background-friendly)**

```bash
python scripts/day7_eval.py --limit 50 --no-ragas
```

- [ ] **Step 8.6: Commit everything**

```bash
git add scripts/day7_eval.py eval_results/day_07_results.md eval_results/day_07_raw.json
git commit -m "feat(day-07): eval script + day_07_results.md with generation metrics"
```

---

## Task 9: Run Full Test Suite

- [ ] **Step 9.1: Run all tests to confirm no regressions**

```bash
pytest tests/ -v --tb=short
```
Expected: all existing tests pass; new generation + eval tests pass

- [ ] **Step 9.2: Final commit with day tag**

```bash
git tag day-07
git push origin feat/day7 --tags
```

---

## Verification Checklist

1. `pytest tests/` — all tests green (no regressions, new tests pass)
2. `python -c "from rag_hub.generation.corrective_generator import CorrectiveGenerator; print('OK')"` — clean import
3. `cat eval_results/day_07_results.md` — metric table present, corrective loop firing rate visible
4. At least one answer in results has `num_citations >= 1` (structured output working)
5. Force corrective loop: `python -c "from rag_hub.generation.hallucination_detector import HallucinationDetector; d = HallucinationDetector(threshold=0.0); print(d.threshold)"` — confirms threshold is settable
