"""Initialize Vertex AI once. Subsequent calls are no-ops."""
import os

import vertexai
from google.oauth2 import service_account

_initialized = False


def init_vertex_ai() -> None:
    global _initialized
    if _initialized:
        return
    credentials = service_account.Credentials.from_service_account_file(
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"],
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    from rag_hub.config.settings import GCP_PROJECT_ID, GCP_LOCATION

    vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION, credentials=credentials)
    _initialized = True


def make_chat_llm(model: str = None, temperature: float = 0, **kwargs):
    """Build a ChatVertexAI with a sane shared retry policy.

    Ensures Vertex AI is initialised, then returns a ChatVertexAI reading project
    /location from the vertexai.init() globals. Caps max_retries at 2 (the library
    default of 6 with exponential backoff can turn a rate-limited call into minutes
    of waiting). Project/location are intentionally NOT passed — passing an empty
    project string bypasses the library's fallback to the init globals.
    """
    from langchain_google_vertexai import ChatVertexAI
    from rag_hub.config.settings import GEMINI_LLM_MODEL

    init_vertex_ai()
    kwargs.setdefault("max_retries", 2)
    return ChatVertexAI(
        model_name=model or GEMINI_LLM_MODEL,
        temperature=temperature,
        **kwargs,
    )
