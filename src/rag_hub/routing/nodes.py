"""
LangGraph node functions for the Day 5 routing pipeline.

Each node receives the shared RouterState dict and returns a partial update.
LangGraph merges the update back into state before calling the next node.

DAG shape:

    [classify_node]
         |
         | conditional_edge (reads state["route"])
         |
    ┌────┴─────┬──────────────┬──────────────┐
    ▼          ▼              ▼              ▼
[direct]   [hyde_node]  [decompose_node] [stepback_node]
    └────┬─────┘              └──────────────┘
         ▼
   [generate_node]
         ▼
      answer

State keys written by each node:
  classify_node   → route
  *_node          → docs   (list of chunk dicts)
  generate_node   → answer

All retrieval nodes share the same interface so generate_node only needs
to read state["docs"] — it doesn't care which branch produced them.
"""

import os
from typing import List, Dict

import vertexai
from google.oauth2 import service_account
from dotenv import load_dotenv

from rag_hub.routing.router import QuestionRouter
from rag_hub.routing.state import RouterState
from rag_hub.embeddings.gemini_001 import GeminiEmbeddingClient
from rag_hub.vectorstore.qdrant_store import QdrantStore
from rag_hub.query.hyde import HyDETransform
from rag_hub.query.decomposition import DecompositionTransform
from rag_hub.query.step_back import StepBackTransform
from rag_hub.generation.gemini_LLM import GeminiFlashGenerator

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

# ---------------------------------------------------------------------------
# Lazy singletons — initialised on first node call, reused thereafter.
# Lazy init avoids slow network calls at import time when only the router
# is needed (e.g. routing-accuracy eval with FULL_PIPELINE=False).
# ---------------------------------------------------------------------------

_singletons: dict = {}


def _get(key: str):
    if key not in _singletons:
        if key == "embedder":
            _singletons[key] = GeminiEmbeddingClient()
        elif key == "store":
            _singletons[key] = QdrantStore(collection="financebench_v1")
        elif key == "router":
            _singletons[key] = QuestionRouter(verbose=True)
        elif key == "hyde":
            _singletons[key] = HyDETransform(verbose=False)
        elif key == "decompose":
            _singletons[key] = DecompositionTransform(verbose=False)
        elif key == "stepback":
            _singletons[key] = StepBackTransform(verbose=False)
        elif key == "generator":
            _singletons[key] = GeminiFlashGenerator()
    return _singletons[key]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _embed_and_search(queries: List[str], top_k: int = 5) -> List[Dict]:
    """
    Embed each query with RETRIEVAL_QUERY task type, search Qdrant,
    and deduplicate results by (doc_name, page, chunk_idx).

    Used by all four retrieval nodes — they only differ in what queries
    they pass in.
    """
    seen: set[str] = set()
    chunks: List[Dict] = []

    for q in queries:
        vec = _get("embedder").embed_query(q)
        results = _get("store").search(vec, k=top_k)
        for point in results:
            payload = point.payload
            key = f"{payload['doc_name']}_p{payload['page']}_c{payload.get('chunk_idx', 0)}"
            if key not in seen:
                seen.add(key)
                chunks.append(payload)

    return chunks


# ---------------------------------------------------------------------------
# Node 1 — Classify
# ---------------------------------------------------------------------------

def classify_node(state: RouterState) -> dict:
    """
    Reads state["question"], writes state["route"].

    This is the entry node. It runs the QuestionRouter (rules → LLM fallback)
    and stores the routing decision. LangGraph's conditional_edge then reads
    state["route"] to branch to the correct retrieval node.

    No retrieval happens here — classification is cheap (often zero LLM calls).
    """
    route = _get("router").route(state["question"])
    return {"route": route}


# ---------------------------------------------------------------------------
# Node 2a — Direct retrieval
# ---------------------------------------------------------------------------

def direct_node(state: RouterState) -> dict:
    """
    Reads state["question"], writes state["docs"].

    Plain vector search: embed the raw question with RETRIEVAL_QUERY task type
    and retrieve the top-k closest chunks.

    Best for: single-fact lookups ("What was revenue in FY2022?").
    The question and answer live in the same embedding subspace, so no
    query transformation is needed.
    """
    docs = _embed_and_search([state["question"]])
    print(f"[DirectNode] Retrieved {len(docs)} chunks")
    return {"docs": docs}


# ---------------------------------------------------------------------------
# Node 2b — HyDE retrieval
# ---------------------------------------------------------------------------

def hyde_node(state: RouterState) -> dict:
    """
    Reads state["question"], writes state["docs"].

    HyDE pipeline:
      1. LLM generates a hypothetical 10-K passage that would answer the question.
      2. That passage is embedded with RETRIEVAL_DOCUMENT task type (not RETRIEVAL_QUERY).
      3. The passage embedding is used to search Qdrant.

    Why RETRIEVAL_DOCUMENT for the hypothetical passage?
      Gemini's embedding model uses different task types for queries vs. documents.
      The index was built with RETRIEVAL_DOCUMENT. By embedding the hypothetical
      passage with the same task type, we search in the same vector subspace as
      the corpus — closing the query-document gap.

    Best for: explanatory / conceptual questions where the answer is prose.
    """
    hypothetical_passages = _get("hyde").transform(state["question"])  # → [passage]

    # Embed with RETRIEVAL_DOCUMENT — same task type as the indexed corpus.
    seen: set[str] = set()
    chunks: List[Dict] = []
    for passage in hypothetical_passages:
        vec = _get("embedder").embed_documents([passage])[0]
        results = _get("store").search(vec, k=5)
        for point in results:
            payload = point.payload
            key = f"{payload['doc_name']}_p{payload['page']}_c{payload.get('chunk_idx', 0)}"
            if key not in seen:
                seen.add(key)
                chunks.append(payload)

    print(f"[HyDENode] Retrieved {len(chunks)} chunks via hypothetical passage")
    return {"docs": chunks}


# ---------------------------------------------------------------------------
# Node 2c — Decomposition retrieval
# ---------------------------------------------------------------------------

def decompose_node(state: RouterState) -> dict:
    """
    Reads state["question"], writes state["docs"].

    Decomposition pipeline:
      1. LLM splits the complex question into atomic sub-questions.
      2. Each sub-question is embedded and searched independently.
      3. Results are deduplicated and merged.

    Why retrieve per sub-question?
      A multi-hop question like "Did margin improve from 2021-2022 and why?"
      has two parts: a numerical lookup AND a qualitative explanation. A single
      embedding averages across both meanings and retrieves mediocre results
      for each. Retrieving separately maximises recall for each hop.

    Best for: comparative, multi-year, conditional, or two-part questions.
    """
    sub_questions = _get("decompose").transform(state["question"])
    docs = _embed_and_search(sub_questions, top_k=4)
    print(f"[DecomposeNode] {len(sub_questions)} sub-questions → {len(docs)} chunks")
    return {"docs": docs}


# ---------------------------------------------------------------------------
# Node 2d — Step-back retrieval
# ---------------------------------------------------------------------------

def stepback_node(state: RouterState) -> dict:
    """
    Reads state["question"], writes state["docs"].

    Step-back pipeline:
      1. LLM abstracts the specific question into a broader general question.
      2. Both the broad question and the original are embedded and searched.
      3. Results are merged — broad context + specific lookup.

    Why retrieve for both?
      The broad question retrieves MD&A / Risk Factors context that sets up
      the answer. The specific question retrieves the precise data point.
      Together they give the generator both background and specifics.

    Best for: "why" questions, risk/exposure reasoning, strategic context.
    """
    queries = _get("stepback").transform(state["question"])  # → [broad, original]
    docs = _embed_and_search(queries, top_k=4)
    print(f"[StepBackNode] {len(queries)} queries → {len(docs)} chunks")
    return {"docs": docs}


# ---------------------------------------------------------------------------
# Node 3 — Generate
# ---------------------------------------------------------------------------

def generate_node(state: RouterState) -> dict:
    """
    Reads state["question"] + state["docs"], writes state["answer"].

    This node is shared by all four branches — it doesn't know or care which
    retrieval strategy was used. It just formats the docs into context and
    calls the LLM generator.

    Keeping generation separate from retrieval means we can swap the generator
    (e.g. GPT-4 vs Gemini) without touching any retrieval node.
    """
    answer = _get("generator").generate(
        question=state["question"],
        chunks=state["docs"],
    )
    return {"answer": answer}
