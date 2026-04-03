"""
ingest.py — Document ingestion pipeline

Flow:
  1. Parse PDF / DOCX / TXT  →  raw text
  2. Split into overlapping chunks
  3. Embed each chunk via OpenAI
  4. Persist to FAISS vector store + metadata JSON
"""

import os
import json
import uuid
import hashlib
from pathlib import Path
from typing import IO

import faiss
import numpy as np
from openai import OpenAI
from pypdf import PdfReader
from docx import Document as DocxDocument

# ── Config ───────────────────────────────────────────────────────────────────
VECTOR_STORE_DIR = Path("vector_store")
VECTOR_STORE_DIR.mkdir(exist_ok=True)

META_FILE = VECTOR_STORE_DIR / "metadata.json"
INDEX_FILE = VECTOR_STORE_DIR / "index.faiss"

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
CHUNK_SIZE = 512        # characters
CHUNK_OVERLAP = 64      # characters

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── Metadata helpers ─────────────────────────────────────────────────────────
def _load_meta() -> dict:
    if META_FILE.exists():
        return json.loads(META_FILE.read_text())
    return {"documents": [], "chunks": []}


def _save_meta(meta: dict) -> None:
    META_FILE.write_text(json.dumps(meta, indent=2))


def list_documents() -> list[dict]:
    """Return list of ingested documents for the sidebar."""
    meta = _load_meta()
    return meta.get("documents", [])


# ── Text extraction ──────────────────────────────────────────────────────────
def _extract_text(file: IO, filename: str) -> str:
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        reader = PdfReader(file)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)

    elif ext == ".docx":
        doc = DocxDocument(file)
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())

    elif ext == ".txt":
        return file.read().decode("utf-8", errors="ignore")

    raise ValueError(f"Unsupported file type: {ext}")


# ── Chunking ─────────────────────────────────────────────────────────────────
def _chunk_text(text: str, doc_id: str, filename: str) -> list[dict]:
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "chunk_id": f"{doc_id}_{idx}",
                "doc_id": doc_id,
                "doc_name": filename,
                "text": chunk_text,
                "char_start": start,
                "char_end": end,
            })
        start += CHUNK_SIZE - CHUNK_OVERLAP
        idx += 1
    return chunks


# ── Embedding ────────────────────────────────────────────────────────────────
def _embed_chunks(chunks: list[dict]) -> np.ndarray:
    texts = [c["text"] for c in chunks]
    # Batch in groups of 100 (API limit)
    all_embeddings = []
    for i in range(0, len(texts), 100):
        batch = texts[i : i + 100]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        all_embeddings.extend([e.embedding for e in response.data])
    return np.array(all_embeddings, dtype="float32")


# ── FAISS index helpers ──────────────────────────────────────────────────────
def _load_index() -> faiss.IndexFlatIP:
    if INDEX_FILE.exists():
        return faiss.read_index(str(INDEX_FILE))
    index = faiss.IndexFlatIP(EMBED_DIM)   # Inner-product (cosine after norm)
    return index


def _save_index(index: faiss.IndexFlatIP) -> None:
    faiss.write_index(index, str(INDEX_FILE))


# ── Public API ───────────────────────────────────────────────────────────────
def ingest_document(file: IO) -> str | None:
    """
    Ingest a document file object.
    Returns doc_id on success, None on failure.
    """
    filename = file.name
    raw_bytes = file.read()
    file_hash = hashlib.md5(raw_bytes).hexdigest()

    meta = _load_meta()

    # Skip if already ingested (same hash)
    if any(d["hash"] == file_hash for d in meta["documents"]):
        existing = next(d for d in meta["documents"] if d["hash"] == file_hash)
        return existing["id"]

    doc_id = str(uuid.uuid4())[:8]

    # Extract text
    import io
    text = _extract_text(io.BytesIO(raw_bytes), filename)
    if not text.strip():
        return None

    # Chunk
    chunks = _chunk_text(text, doc_id, filename)

    # Embed
    embeddings = _embed_chunks(chunks)

    # Normalise for cosine similarity
    faiss.normalize_L2(embeddings)

    # Add to FAISS
    index = _load_index()
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    _save_index(index)

    # Persist metadata
    meta["documents"].append({
        "id": doc_id,
        "name": filename,
        "hash": file_hash,
        "chunks": len(chunks),
        "size": f"{len(raw_bytes) // 1024} KB",
    })
    meta["chunks"].extend(chunks)
    _save_meta(meta)

    return doc_id
