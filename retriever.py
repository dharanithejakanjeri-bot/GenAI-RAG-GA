"""
utils/retriever.py — Vector similarity retrieval from FAISS index
"""

import os
import json
import numpy as np
import faiss
from pathlib import Path
from openai import OpenAI

VECTOR_STORE_DIR = Path("vector_store")
INDEX_FILE = VECTOR_STORE_DIR / "index.faiss"
META_FILE = VECTOR_STORE_DIR / "metadata.json"

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _embed_query(query: str) -> np.ndarray:
    response = client.embeddings.create(model=EMBED_MODEL, input=[query])
    vec = np.array([response.data[0].embedding], dtype="float32")
    faiss.normalize_L2(vec)
    return vec


def retrieve_context(query: str, doc_ids: list[str], top_k: int = 4) -> list[dict]:
    """
    Search the FAISS index for the top_k most relevant chunks
    belonging to the given doc_ids.

    Returns list of chunk dicts with an added 'score' field.
    """
    if not INDEX_FILE.exists() or not META_FILE.exists():
        return []

    index = faiss.read_index(str(INDEX_FILE))
    meta = json.loads(META_FILE.read_text())
    all_chunks = meta.get("chunks", [])

    if not all_chunks:
        return []

    query_vec = _embed_query(query)

    # Search broadly, then filter by doc_id
    search_k = min(top_k * 10, index.ntotal)
    scores, indices = index.search(query_vec, search_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(all_chunks):
            continue
        chunk = all_chunks[idx]
        if chunk["doc_id"] not in doc_ids:
            continue
        results.append({**chunk, "score": float(score)})
        if len(results) >= top_k:
            break

    return results
