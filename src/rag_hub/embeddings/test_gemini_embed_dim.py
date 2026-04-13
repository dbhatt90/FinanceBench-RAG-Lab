from gemini_001 import GeminiEmbeddingClient

emb = GeminiEmbeddingClient()

vec = emb.embed_query("test sentence")

print(len(vec))
