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
