import re
import time
from typing import List

import vertexai
from google import genai
from google.genai import types
from google.oauth2 import service_account
from dotenv import load_dotenv
import os

load_dotenv()

# Initialise Vertex AI once at import time.
# vertexai.init() registers the project/location/credentials globally so that
# genai.Client(vertexai=True) routes all calls through Vertex AI (GCP billing)
# instead of the Gemini API free tier.
_credentials = service_account.Credentials.from_service_account_file(
    os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)
vertexai.init(
    project=os.getenv("GCP_PROJECT_ID"),
    location=os.getenv("GCP_LOCATION", "us-central1"),
    credentials=_credentials,
)


class GeminiEmbeddingClient:
    """
    Embedding client using the new unified Google GenAI SDK routed via Vertex AI.

    SDK:   google-genai  (replaces both google-generativeai and vertexai language_models)
    Route: genai.Client(vertexai=True) → aiplatform.googleapis.com → GCP billing
           (NOT ai.google.dev free tier)

    Model: gemini-embedding-001
      - Stable GA on Vertex AI
      - Dimensionality: 768 (explicit via output_dimensionality)
      - MTEB-competitive retrieval performance

    task_type matters:
      RETRIEVAL_DOCUMENT — indexing corpus chunks
      RETRIEVAL_QUERY    — embedding user queries at search time
      The model produces different vector spaces for each; mixing them hurts recall.
    """

    def __init__(self, model: str = "gemini-embedding-001", dim: int = 768):
        self.model = model
        self.dim = dim
        # vertexai=True routes through Vertex AI using the vertexai.init() config above
        self._client = genai.Client(
            vertexai=True,
            project=os.getenv("GCP_PROJECT_ID"),
            location=os.getenv("GCP_LOCATION", "us-central1"),
            credentials=_credentials,   # explicit — don't let it auto-detect GEMINI_API_KEY
        )



    # -----------------------------
    # Batch embedding (documents)
    # -----------------------------
    def embed_documents(
        self,
        texts: List[str],
        batch_size: int = 100,
    ) -> List[List[float]]:
        """
        Embed corpus chunks in batches.
        Each batch = 1 API call with task_type=RETRIEVAL_DOCUMENT.
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                result = self._client.models.embed_content(
                    model=self.model,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=self.dim,
                    ),
                )
                all_embeddings.extend([e.values for e in result.embeddings])
                time.sleep(1)

            except Exception as e:
                delay = self._parse_retry_delay(str(e), default=30)
                print(f"[WARN] Batch {i}-{i+batch_size} failed: {e}")
                print(f"[INFO] Retrying after {delay}s...")
                time.sleep(delay)

                result = self._client.models.embed_content(
                    model=self.model,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=self.dim,
                    ),
                )
                all_embeddings.extend([e.values for e in result.embeddings])

        return all_embeddings

    # -----------------------------
    # Query embedding
    # -----------------------------
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query at retrieval time.
        Uses RETRIEVAL_QUERY — different task type from documents, intentionally.
        """
        try:
            result = self._client.models.embed_content(
                model=self.model,
                contents=[text],   # list, not bare string
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=self.dim,
                ),
            )
            return result.embeddings[0].values



        except Exception as e:
            delay = self._parse_retry_delay(str(e), default=10)
            print(f"[WARN] Query embedding failed, retrying after {delay}s: {e}")
            time.sleep(delay)
            result = self._client.models.embed_content(
                model=self.model,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=self.dim,
                ),
            )
            return result.embeddings[0].values

    # -----------------------------
    # Helpers
    # -----------------------------
    @staticmethod
    def _parse_retry_delay(error_str: str, default: int = 30) -> int:
        """Extract retryDelay seconds from a quota error. Falls back to default."""
        match = re.search(r"retry in (\d+)", error_str, re.IGNORECASE)
        return int(match.group(1)) + 5 if match else default
