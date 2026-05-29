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
