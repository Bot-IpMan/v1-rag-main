#!/usr/bin/env python
import argparse, os, glob, pathlib
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

def load_docs(path):
    paths = glob.glob(f"{path}/**/*", recursive=True)
    for p in paths:
        if os.path.isfile(p) and p.lower().endswith((".md", ".txt")):
            yield pathlib.Path(p).read_text(encoding="utf-8"), p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_dir", default="data/knowledge")
    ap.add_argument("--collection", default="docs")
    ap.add_argument("--url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    args = ap.parse_args()

    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    client = QdrantClient(url=args.url)

    # (re)create collection
    dim = model.get_sentence_embedding_dimension()
    client.recreate_collection(
        collection_name=args.collection,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
    )

    points = []
    for idx, (text, path) in enumerate(load_docs(args.source_dir)):
        emb = model.encode(text)
        payload = {"text": text, "source": path}
        points.append(models.PointStruct(id=idx, vector=emb, payload=payload))

    client.upsert(collection_name=args.collection, points=points)
    print(f"✅  Upserted {len(points)} documents into {args.collection}")

if __name__ == "__main__":
    main()
