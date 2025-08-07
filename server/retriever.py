from typing import List
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import os

COLLECTION = os.getenv("COLLECTION_NAME", "docs")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
_client = QdrantClient(url=QDRANT_URL)

def fetch_relevant(query: str, k: int = 4) -> List[str]:
    emb = _model.encode(query)
    hits = _client.search(
        collection_name=COLLECTION,
        query_vector=emb,
        limit=k,
        with_payload=True
    )
    return [hit.payload["text"] for hit in hits]
