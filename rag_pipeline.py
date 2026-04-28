"""
RAG Pipeline — FAISS-backed semantic search with bill tracking.

Optimisations over original:
- Smarter overlap-based chunking (avoids cutting mid-sentence)
- Normalised embeddings cached per session
- Safer FAISS index guards
- Cleaner count / company extraction logic
"""

from __future__ import annotations

import re
import logging
from functools import lru_cache

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from llm import ask_llm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding model (loaded once at import time)
# ---------------------------------------------------------------------------
EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
_embed_model = SentenceTransformer(EMBED_MODEL_NAME)
DIMENSION = 768

# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------
index = faiss.IndexFlatIP(DIMENSION)   # inner-product on L2-normalised vecs
documents: list[str] = []
bill_registry: list[str] = []          # unique uploaded doc names


# ---------------------------------------------------------------------------
# Chunking helper
# ---------------------------------------------------------------------------
def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split *text* into overlapping chunks, preferring sentence boundaries."""
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # Try to break at the last sentence-ending punctuation within the window
        if end < len(text):
            last_period = text.rfind(".", start, end)
            last_newline = text.rfind("\n", start, end)
            break_at = max(last_period, last_newline)
            if break_at > start:
                end = break_at + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap  # slide back by *overlap* chars

    return chunks


# ---------------------------------------------------------------------------
# Encode helper (thin wrapper for consistency)
# ---------------------------------------------------------------------------
def _encode(texts: list[str]) -> np.ndarray:
    """Return L2-normalised float32 embeddings."""
    vecs = _embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype="float32")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def add_documents(texts: list[str], doc_name: str = "Bill") -> int:
    """Embed *texts*, add to FAISS index, and return count of chunks added."""
    global documents

    if doc_name not in bill_registry:
        bill_registry.append(doc_name)

    labeled = [f"[{doc_name}] {t}" for t in texts]
    documents.extend(labeled)

    embeddings = _encode(labeled)
    index.add(embeddings)
    logger.info("Added %d chunks from '%s' (index total: %d)", len(labeled), doc_name, index.ntotal)
    return len(labeled)


def add_raw_text(text: str, doc_name: str = "Bill", chunk_size: int = 500, overlap: int = 50) -> int:
    """Convenience: chunk raw text then add to the index."""
    chunks = _chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    return add_documents(chunks, doc_name=doc_name)


def retrieve(query: str, k: int = 4) -> list[str]:
    """Return the top-*k* most relevant chunks for *query*."""
    if not documents or index.ntotal == 0:
        return []

    k = min(k, index.ntotal)  # can't ask for more than we have
    query_vec = _encode([query])
    distances, indices = index.search(query_vec, k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx != -1:
            results.append(documents[idx])
    return results


# ---------------------------------------------------------------------------
# Count-query helpers
# ---------------------------------------------------------------------------
_COUNT_KEYWORDS = [
    # English
    "how many", "count", "number of", "total number",
    "total bills", "total invoices",
    # Spanish
    "cuantos", "cuantas",
    # Portuguese
    "quantos", "quantas",
    # French
    "combien", "nombre de",
    # Hindi (romanised)
    "kitne bill", "kitni", "koyti",
    # Russian (romanised)
    "skolko", "kolichestvo",
    # Japanese (romanised)
    "ikutsu", "nanmai", "nanken",
]

_AMOUNT_KEYWORDS = {
    "gst", "tax", "amount", "price", "discount",
    "cost", "fee", "charge", "rate", "value", "sum",
    "kitna", "kitne", "kitni",
}


def is_count_query(query: str) -> bool:
    """Return True when *query* is asking for a *count* of bills."""
    q = query.lower()
    if any(kw in q for kw in _COUNT_KEYWORDS):
        return True
    if any(w in q for w in _AMOUNT_KEYWORDS):
        return False
    return False


def extract_company(query: str) -> str:
    """Use the LLM to extract a company/brand name from *query*."""
    prompt = (
        "Extract the company or brand name from this query.\n"
        "If NO company/brand is mentioned, return exactly: NONE\n"
        "Return ONLY the name or NONE, nothing else.\n\n"
        f"Query: {query}"
    )
    raw = ask_llm(prompt, temperature=0.0, max_tokens=64)
    return raw.strip()


def count_bills(query: str) -> str:
    """Answer a count-type question about uploaded bills."""
    company_raw = extract_company(query)
    company = company_raw.lower().strip()

    ignore = {"none", "", "no company", "no brand", "outlook", "n/a"}
    if company in ignore:
        total = len(bill_registry)
        names = "\n".join(f"  • {n}" for n in bill_registry) if bill_registry else "  (none)"
        return f"📄 **Total bills uploaded: {total}**\n{names}"

    # Filter bills whose *text* mentions the company
    seen: set[str] = set()
    for doc in documents:
        if company in doc.lower():
            doc_name = doc.split("]")[0].replace("[", "").strip()
            seen.add(doc_name)

    if not seen:
        return f"No bills found for **{company_raw}**."
    names = "\n".join(f"  • {n}" for n in seen)
    return f"📄 **Bills for {company_raw}: {len(seen)}**\n{names}"