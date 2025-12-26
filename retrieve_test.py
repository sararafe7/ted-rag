# retrieve_test.py
# Purpose: debug retrieval after ingestion (no GPT answering here)
# - Embeds the question (cheap)
# - Queries Pinecone
# - Prints top matches with title/speaker/score + chunk snippet
# - Also prints "top distinct talks" (dedup by talk_id) for the "3 titles" requirement

import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

# ---------- Config ----------
EMBED_MODEL = "RPRTHPB-text-embedding-3-small"
TOP_K = int(os.getenv("TOP_K", "10"))          # debug: 10 is fine
SNIPPET_CHARS = 220                             # how much chunk text to print
DEDUP_TALKS_TO_SHOW = 3                         # show top 3 distinct talks

# ---------- Clients ----------
client = OpenAI()  # uses OPENAI_API_KEY and OPENAI_BASE_URL from .env
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))   # e.g., "ted-rag"

# ---------- Test question ----------
question = "Who is the speaker of 'One Laptop per Child'?"
# Try other questions too:
# question = "Recommend a TED talk about climate change and justify it."
# question = "Which TED talks are about business and decision-making? Return exactly 3 titles."

# ---------- 1) Embed question ----------
emb_resp = client.embeddings.create(
    model=EMBED_MODEL,
    input=question
)
qvec = emb_resp.data[0].embedding

# ---------- 2) Query Pinecone ----------
res = index.query(
    vector=qvec,
    top_k=TOP_K,
    include_metadata=True
)

matches = res.get("matches", [])
print(f"\nQuestion: {question}\n")
print(f"Returned matches: {len(matches)} (top_k={TOP_K})\n")

if not matches:
    print("No matches returned. Check that you ingested data and the index name is correct.")
    raise SystemExit(0)

# ---------- 3) Print raw matches (chunks) ----------
print("=== Top matching CHUNKS (may include multiple chunks from same talk) ===\n")
for rank, m in enumerate(matches, start=1):
    meta = m.get("metadata", {})
    title = meta.get("title", "<no title>")
    speaker = meta.get("speaker", "<no speaker>")
    talk_id = meta.get("talk_id", "<no talk_id>")
    chunk_index = meta.get("chunk_index", "<no chunk_index>")
    score = m.get("score", 0.0)

    chunk_text = meta.get("chunk", "")
    chunk_snip = (chunk_text[:SNIPPET_CHARS].replace("\n", " ") + "...") if chunk_text else "<no chunk text>"

    print(f"{rank}. {title} | speaker: {speaker} | talk_id: {talk_id} | chunk_index: {chunk_index} | score={score:.4f}")
    print(f"   snippet: {chunk_snip}\n")

# ---------- 4) Print top DISTINCT talks (dedupe by talk_id) ----------
print("=== Top DISTINCT TALKS (dedup by talk_id) ===\n")
seen = set()
distinct_rank = 0

for m in matches:
    meta = m.get("metadata", {})
    talk_id = meta.get("talk_id")
    if not talk_id or talk_id in seen:
        continue

    seen.add(talk_id)
    distinct_rank += 1

    title = meta.get("title", "<no title>")
    speaker = meta.get("speaker", "<no speaker>")
    score = m.get("score", 0.0)

    print(f"{distinct_rank}. {title} | speaker: {speaker} | score={score:.4f}")

    if distinct_rank >= DEDUP_TALKS_TO_SHOW:
        break

# ---------- 5) Special helper: answer "Who is the speaker of X?" using metadata (no GPT) ----------
# If your top distinct talk is clearly the one asked about, we can extract speaker directly.
top_meta = matches[0].get("metadata", {})
if top_meta.get("title") and top_meta.get("speaker"):
    print("\n=== Metadata-based answer (no GPT) ===")
    print(f"Speaker of '{top_meta['title']}': {top_meta['speaker']}\n")
