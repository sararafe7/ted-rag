# api_prompt.py
# Returns JSON exactly as required by the homework:
# {
#   "response": "...",
#   "context": [{"talk_id": "...", "title": "...", "chunk": "...", "score": 0.123}],
#   "Augmented_prompt": {"System": "...", "User": "..."}
# }

import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from dotenv import load_dotenv
from openai import OpenAI

from api_pinecone import retrieve_context

load_dotenv()
client = OpenAI()

GPT_MODEL = "RPRTHPB-gpt-5-mini"

# Internal retrieval vs returned context size
TOP_K = int(os.getenv("TOP_K", "15"))      # internal retrieval for most questions
TOP_K_LIST = 30                            # internal retrieval for list questions (max allowed) :contentReference[oaicite:1]{index=1}
CONTEXT_RETURN_K = 5                       # how many context items you RETURN (all types)

# =======================
# REQUIRED SYSTEM PROMPT
# (DO NOT CHANGE)
# =======================
SYSTEM_PROMPT = (
    "You are a TED Talk assistant that answers questions strictly and "
    "only based on the TED dataset context provided to you (metadata "
    "and transcript passages). You must not use any external "
    "knowledge, the open internet, or information that is not explicitly "
    "contained in the retrieved context. If the answer cannot be "
    "determined from the provided context, respond: “I don’t know "
    "based on the provided TED data.” Always explain your answer "
    "using the given context, quoting or paraphrasing the relevant "
    "transcript or metadata when helpful."
)

# ---------- Helpers ----------

def normalize(s: str) -> str:
    return (s or "").lower().strip()


def extract_quoted_title(question: str) -> str:
    m = re.search(r"[\"']([^\"']+)[\"']", question)
    return m.group(1) if m else ""


def parse_requested_count(question: str) -> Optional[int]:
    """
    Extract requested list size for Type 2.
    Supports: 'exactly 3', '3 titles', 'three titles', etc.
    Caps at 3 per assignment. :contentReference[oaicite:2]{index=2}
    """
    q = normalize(question)

    # digits
    m = re.search(r"\bexactly\s+(\d+)\b", q)
    if m:
        return min(int(m.group(1)), 3)

    m = re.search(r"\b(\d+)\s+(?:talks|titles)\b", q)
    if m:
        return min(int(m.group(1)), 3)

    # words
    word_to_num = {"one": 1, "two": 2, "three": 3}
    m = re.search(r"\bexactly\s+(one|two|three)\b", q)
    if m:
        return word_to_num[m.group(1)]

    m = re.search(r"\b(one|two|three)\s+(?:talks|titles)\b", q)
    if m:
        return word_to_num[m.group(1)]

    return None


def detect_type(question: str) -> int:
    """
    Type 1: find a talk / provide title+speaker OR metadata fact like speaker/url/topics
    Type 2: list exactly N titles (N<=3)
    Type 3: summary / key idea
    Type 4: recommend / suggest
    Default: 4 (safe, still constrained by SYSTEM_PROMPT)
    """
    q = normalize(question)

    # Type 2: list exactly N titles (most specific)
    n = parse_requested_count(question)
    if n is not None and ("title" in q or "titles" in q or "talk" in q or "talks" in q):
        return 2

    # Type 3: summarize
    if any(k in q for k in ["summarize", "summary", "key idea", "main idea", "in a few sentences"]):
        return 3

    # Type 4: recommend
    if any(k in q for k in ["recommend", "suggest", "which talk should i watch", "what should i watch"]):
        return 4

    # Type 1: factual / find talk + title/speaker
    if any(k in q for k in [
        "who is the speaker", "speaker of", "url of", "topics of", "provide the title and speaker",
        "title and speaker", "find a ted talk", "find a talk", "which talk talks about"
    ]):
        return 1

    # Default
    return 4


def make_context_output(matches, limit: int):
    """
    Convert Pinecone matches into EXACT homework context format:
    [{talk_id,title,chunk,score}, ...]
    """
    out = []
    for m in matches[:limit]:
        meta = m.get("metadata", {})
        out.append({
            "talk_id": str(meta.get("talk_id", "")),
            "title": str(meta.get("title", "")),
            "chunk": str(meta.get("chunk", "")),
            "score": float(m.get("score", 0.0)),
        })
    return out


def group_matches_by_talk(matches) -> Dict[str, List[dict]]:
    """
    Group Pinecone matches by talk_id, preserving order.
    """
    groups = defaultdict(list)
    for m in matches:
        meta = m.get("metadata", {}) or {}
        tid = str(meta.get("talk_id", "")).strip()
        if tid:
            groups[tid].append(m)
    return groups


def best_talk_id(matches) -> Optional[str]:
    """
    Choose the talk_id whose best match score is highest.
    """
    best_tid = None
    best_score = -1.0
    for m in matches:
        meta = m.get("metadata", {}) or {}
        tid = str(meta.get("talk_id", "")).strip()
        score = float(m.get("score", 0.0))
        if tid and score > best_score:
            best_score = score
            best_tid = tid
    return best_tid


def format_excerpts_for_gpt(matches: List[dict], max_chunks: int, excerpt_chars: int) -> str:
    """
    Build numbered excerpts so we can require [1]/[2]/[3] references in Type 3/4.
    """
    out = []
    for i, m in enumerate(matches[:max_chunks], start=1):
        meta = m.get("metadata", {}) or {}
        excerpt = (meta.get("chunk", "") or "")[:excerpt_chars].replace("\n", " ")
        out.append(
            f"[{i}] Title: {meta.get('title','')}\n"
            f"Speaker: {meta.get('speaker','')}\n"
            f"Topics: {meta.get('topics','')}\n"
            f"Transcript excerpt: {excerpt}"
        )
    return "\n\n".join(out)


def extract_simple_keyword(question: str) -> Optional[str]:
    """
    Very light heuristic for Type 2 filtering.
    Finds 'about X' or a single strong keyword.
    """
    q = normalize(question)
    m = re.search(r"\babout\s+([a-z][a-z\s\-]{2,40})", q)
    if m:
        kw = m.group(1).strip()
        # take first phrase chunk
        return kw.split(",")[0].strip()
    # fallback: none
    return None


def matches_keyword(m: dict, keyword: str) -> bool:
    """
    Check keyword presence in title/topics/chunk (case-insensitive).
    """
    if not keyword:
        return True
    meta = m.get("metadata", {}) or {}
    hay = " ".join([
        str(meta.get("title", "")),
        str(meta.get("topics", "")),
        str(meta.get("chunk", "")),
    ]).lower()
    return keyword.lower() in hay


def question_overlap_score(question: str, chunk: str) -> int:
    q_words = set(re.findall(r"[a-z]{4,}", question.lower()))
    c_words = set(re.findall(r"[a-z]{4,}", (chunk or "").lower()))
    return len(q_words & c_words)


# ======================
# MAIN ENTRY POINT
# ======================
def answer_question(question: str):
    qtype = detect_type(question)

    # 1) Retrieve (internal)
    internal_top_k = TOP_K_LIST if qtype == 2 else TOP_K
    matches = retrieve_context(question, top_k=internal_top_k, include_metadata=True) or []

    # 2) Return ONLY a small number of chunks (homework wants concise)
    context_out = make_context_output(matches, limit=CONTEXT_RETURN_K)

    # If retrieval returns nothing
    if not matches:
        return {
            "response": "I don’t know based on the provided TED data.",
            "context": context_out,
            "Augmented_prompt": {"System": SYSTEM_PROMPT, "User": f"QUESTION:\n{question}\n\nCONTEXT:\n"}
        }

    # ---------- TYPE 1 ----------
    # Type 1 includes:
    #  (a) metadata fact queries: speaker/url/topics of a specific titled talk
    #  (b) semantic "find a talk about X" -> return title + speaker (based on retrieval)
    if qtype == 1:
        q = normalize(question)
        quoted = extract_quoted_title(question)

        chosen = None

        # If user quoted a title, try to find that exact title in matches
        if quoted:
            for m in matches:
                title = normalize((m.get("metadata") or {}).get("title", ""))
                if title == normalize(quoted):
                    chosen = m
                    break

        # Otherwise choose best matching talk overall (top match)
        if not chosen:
            chosen = matches[0]

        meta = chosen.get("metadata", {}) or {}

        # Metadata fact cases
        if "speaker" in q and quoted:
            response_text = f"The speaker of '{meta.get('title')}' is {meta.get('speaker')}."
        elif "url" in q and quoted:
            response_text = meta.get("url", "I don’t know based on the provided TED data.")
        elif "topics" in q and quoted:
            response_text = meta.get("topics", "I don’t know based on the provided TED data.")
        else:
            # Semantic "find a talk about..." case
            response_text = f"{meta.get('title')} — {meta.get('speaker')}"

        return {
            "response": response_text,
            "context": context_out,
            "Augmented_prompt": {"System": SYSTEM_PROMPT, "User": f"QUESTION:\n{question}\n\nCONTEXT:\n{format_excerpts_for_gpt(matches, 3, 450)}"}
        }

    # ---------- TYPE 2: exactly N titles (NO GPT) ----------
    if qtype == 2:
        requested_n = parse_requested_count(question) or 3
        requested_n = min(requested_n, 3)  # assignment cap :contentReference[oaicite:3]{index=3}

        keyword = extract_simple_keyword(question)
        distinct_titles = []
        seen_talks = set()

        for m in matches:
            meta = m.get("metadata", {}) or {}
            tid = meta.get("talk_id")
            title = meta.get("title")

            if not tid or not title:
                continue
            if tid in seen_talks:
                continue
            if keyword and not matches_keyword(m, keyword):
                continue

            seen_talks.add(tid)
            distinct_titles.append(title)

            if len(distinct_titles) >= requested_n:
                break

        if len(distinct_titles) == 0:
            response_text = "I don’t know based on the provided TED data."
        elif len(distinct_titles) < requested_n:
            response_text = (
                f"I found only {len(distinct_titles)} matching talk(s) in the provided TED data:\n"
                + "\n".join(f"{i + 1}) {t}" for i, t in enumerate(distinct_titles))
            )
        else:
            response_text = f"Here are {requested_n} TED talk titles:\n" + "\n".join(
                f"{i + 1}) {t}" for i, t in enumerate(distinct_titles)
            )

        return {
            "response": response_text,
            "context": context_out,
            "Augmented_prompt": {"System": "retrieval-only", "User": question}
        }

    # ---------- TYPE 3 & 4: GPT ----------
    # Choose one best talk for coherence, then pass multiple chunks from that talk.
    tid = best_talk_id(matches)
    groups = group_matches_by_talk(matches)
    talk_matches = groups.get(tid, matches)

    # Re-rank chunks within the chosen talk by overlap with the question (generic)
    talk_matches = sorted(
        talk_matches,
        key=lambda m: question_overlap_score(question, (m.get("metadata") or {}).get("chunk", "")),
        reverse=True
    )

    if qtype == 3:
        # Summary: more chunks helps produce a coherent key idea
        context_for_gpt = format_excerpts_for_gpt(talk_matches, max_chunks=5, excerpt_chars=500)
        user_prompt = (
            f"QUESTION:\n{question}\n\n"
            f"CONTEXT:\n{context_for_gpt}\n\n"
            "INSTRUCTIONS:\n"
            "- Pick ONE talk from the provided CONTEXT.\n"
            "- Output exactly these fields:\n"
            "  Title:\n"
            "  Speaker:\n"
            "  Key idea summary: (2–4 sentences)\n"
            "- Use only the CONTEXT; if not enough information, say: I don’t know based on the provided TED data.\n"
            "- Ground your summary in the excerpts (quote or paraphrase)."
        )

    else:
        # Type 4 recommendation: fewer but strong excerpts + force evidence markers
        context_for_gpt = format_excerpts_for_gpt(talk_matches, max_chunks=3, excerpt_chars=600)
        user_prompt = (
            f"QUESTION:\n{question}\n\n"
            f"CONTEXT:\n{context_for_gpt}\n\n"
            "INSTRUCTIONS:\n"
            "- Recommend ONE talk from the provided CONTEXT.\n"
            "- Then justify with 2–4 bullet points.\n"
            "- Each bullet MUST cite the excerpt number like [1], [2], or [3].\n"
            "- Use only the CONTEXT; if not enough information, say: I don’t know based on the provided TED data."
        )

    completion = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
    )

    answer_text = completion.choices[0].message.content.strip()

    return {
        "response": answer_text,
        "context": context_out,
        "Augmented_prompt": {"System": SYSTEM_PROMPT, "User": user_prompt}
    }
