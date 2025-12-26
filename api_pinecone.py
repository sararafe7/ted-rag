# api_pinecone.py
# Purpose: reusable Pinecone retrieval for your FastAPI endpoints.
# Returns a context list in the assignment-style shape:
#   [{ "id": ..., "score": ..., "metadata": {...} }, ...]
# Also supports selecting distinct talks (by talk_id) and optionally
# building richer context from the top-k (multiple chunks per selected talk).

import os
from typing import Any, Dict, List, Optional, Set

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

# --- Clients (created once) ---
_client = OpenAI()  # uses OPENAI_API_KEY (+ optional OPENAI_BASE_URL) from .env
_pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
_index = _pc.Index(os.getenv("PINECONE_INDEX"))

EMBED_MODEL = os.getenv("EMBED_MODEL", "RPRTHPB-text-embedding-3-small")

# Keep the same TOP_K for everything
TOP_K = int(os.getenv("TOP_K", "15"))


def embed_query(text: str) -> List[float]:
    """Embed a query string into a dense vector."""
    resp = _client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


def retrieve_context(
    question: str,
    top_k: int = TOP_K,
    include_metadata: bool = True,
) -> List[Dict[str, Any]]:
    """
    Retrieve top_k matching chunks from Pinecone.
    Returns JSON-friendly items: {"id","score","metadata"}.
    """
    qvec = embed_query(question)

    res = _index.query(
        vector=qvec,
        top_k=top_k,
        include_metadata=include_metadata,
    )

    matches = res.get("matches", [])

    context: List[Dict[str, Any]] = []
    for m in matches:
        context.append(
            {
                "id": m.get("id"),
                "score": m.get("score"),
                "metadata": m.get("metadata", {}) if include_metadata else {},
            }
        )
    return context


def select_top_talk_ids(
    context: List[Dict[str, Any]],
    n_talks: int = 3,
) -> List[Any]:
    """
    Select up to n_talks distinct talk_ids from the already-retrieved context,
    in descending score order (since Pinecone returns matches sorted by score).
    """
    seen: Set[Any] = set()
    talk_ids: List[Any] = []

    for item in context:
        md = item.get("metadata") or {}
        tid = md.get("talk_id")
        if tid is None:
            continue
        if tid in seen:
            continue
        seen.add(tid)
        talk_ids.append(tid)
        if len(talk_ids) >= n_talks:
            break

    return talk_ids


def dedupe_context_by_talk(
    context: List[Dict[str, Any]],
    max_distinct_talks: int = 3,
) -> List[Dict[str, Any]]:
    """
    Keep only the highest-scoring chunk per distinct talk_id,
    returning up to max_distinct_talks items.
    """
    talk_ids = select_top_talk_ids(context, n_talks=max_distinct_talks)
    if not talk_ids:
        return []

    out: List[Dict[str, Any]] = []
    seen: Set[Any] = set()
    for item in context:
        tid = (item.get("metadata") or {}).get("talk_id")
        if tid in talk_ids and tid not in seen:
            out.append(item)
            seen.add(tid)
        if len(out) >= max_distinct_talks:
            break
    return out


def build_context_for_talks(
    context: List[Dict[str, Any]],
    talk_ids: List[Any],
    chunks_per_talk: int = 2,
) -> List[Dict[str, Any]]:
    """
    From an already retrieved top-k context, keep up to chunks_per_talk chunks
    for each talk_id in talk_ids (preserving score order).
    This is useful for Type-2 recommendation questions: stronger evidence
    without increasing top_k or making extra Pinecone calls.
    """
    buckets: Dict[Any, List[Dict[str, Any]]] = {tid: [] for tid in talk_ids}

    for item in context:
        tid = (item.get("metadata") or {}).get("talk_id")
        if tid in buckets and len(buckets[tid]) < chunks_per_talk:
            buckets[tid].append(item)

    out: List[Dict[str, Any]] = []
    for tid in talk_ids:
        out.extend(buckets[tid])
    return out


def get_top_titles(
    question: str,
    top_k: int = TOP_K,
    n_titles: int = 3,
) -> List[str]:
    """
    Convenience helper for "return up to 3 titles" question type.
    Uses the same top_k (15) as everything else, then dedupes by talk_id.
    """
    ctx = retrieve_context(
        question=question,
        top_k=top_k,
        include_metadata=True,
    )

    talk_ids = select_top_talk_ids(ctx, n_talks=n_titles)
    if not talk_ids:
        return []

    titles: List[str] = []
    seen_titles: Set[str] = set()

    for item in ctx:
        md = item.get("metadata") or {}
        if md.get("talk_id") not in talk_ids:
            continue
        title = md.get("title")
        if not title:
            continue
        if title in seen_titles:
            continue
        seen_titles.add(title)
        titles.append(title)
        if len(titles) >= n_titles:
            break

    return titles

def best_talk_id(context: List[Dict[str, Any]]) -> Optional[Any]:
    """Return the talk_id of the highest-scoring match."""
    for item in context:
        tid = (item.get("metadata") or {}).get("talk_id")
        if tid is not None:
            return tid
    return None


def _matches_keyword(item: Dict[str, Any], keyword: str) -> bool:
    """Keyword check across title/topics/chunk. Very light filter."""
    if not keyword:
        return True
    md = item.get("metadata") or {}
    hay = " ".join([
        str(md.get("title", "")),
        str(md.get("topics", "")),
        str(md.get("chunk", "")),
    ]).lower()
    return keyword.lower() in hay


def get_best_talk_chunks(
    question: str,
    top_k: int = TOP_K,
    chunks_per_talk: int = 3,
    keyword: str = "",
) -> List[Dict[str, Any]]:
    """
    Retrieve top_k matches, then return up to chunks_per_talk chunks
    from the single best-matching talk (same talk_id).
    Optional keyword filter reduces irrelevant intro/joke chunks.
    """
    ctx = retrieve_context(question=question, top_k=top_k, include_metadata=True)
    tid = best_talk_id(ctx)
    if tid is None:
        return []

    out: List[Dict[str, Any]] = []
    for item in ctx:
        md = item.get("metadata") or {}
        if md.get("talk_id") != tid:
            continue
        if keyword and not _matches_keyword(item, keyword):
            continue
        out.append(item)
        if len(out) >= chunks_per_talk:
            break

    # fallback: if keyword filter removed everything, return unfiltered chunks
    if not out:
        for item in ctx:
            md = item.get("metadata") or {}
            if md.get("talk_id") == tid:
                out.append(item)
                if len(out) >= chunks_per_talk:
                    break

    return out
