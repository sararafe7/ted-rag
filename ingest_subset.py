import os
import ast
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

from chunking import chunk_text
from config import CHUNK_SIZE, OVERLAP_RATIO

load_dotenv()

# --- Clients ---
client = OpenAI()  # uses OPENAI_API_KEY + OPENAI_BASE_URL from .env
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))  # e.g. "ted-rag"

# --- Load subset ---
df = pd.read_csv("data/50 talks.csv")

# If you want all 50 rows, use df (no slicing).
# If you want a smaller test slice, change these numbers.
df_subset = df  # <-- simplest: ingest all rows in the file

required_cols = ["talk_id", "title", "speaker_1", "transcript"]
missing = [c for c in required_cols if c not in df_subset.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

BATCH_SIZE = 50
vectors_batch = []
total_chunks = 0

def parse_topics(x):
    """
    Your CSV topics sometimes look like "['business','culture']" (string).
    We normalize to a single string "business, culture".
    """
    if pd.isna(x):
        return ""
    if isinstance(x, list):
        return ", ".join([str(t) for t in x])
    if isinstance(x, str):
        s = x.strip()
        # Try to parse python-list-like string safely
        if s.startswith("[") and s.endswith("]"):
            try:
                arr = ast.literal_eval(s)
                if isinstance(arr, list):
                    return ", ".join([str(t) for t in arr])
            except Exception:
                pass
        return s
    return str(x)

for _, row in df_subset.iterrows():
    talk_id = str(row.get("talk_id", "")).strip()
    title = str(row.get("title", "")).strip()
    speaker = str(row.get("speaker_1", "")).strip()

    transcript = row.get("transcript", "")
    if not isinstance(transcript, str) or not transcript.strip():
        continue

    # NEW metadata fields (agreed set)
    description = str(row.get("description", "")).strip()
    url = str(row.get("url", "")).strip()
    topics = parse_topics(row.get("topics", ""))

    # Chunk transcript
    chunks = chunk_text(transcript, chunk_size=CHUNK_SIZE, overlap_ratio=OVERLAP_RATIO)

    for i, raw_chunk in enumerate(chunks):
        # Keep SAME embedding approach as before (good for retrieval)
        chunk_for_embedding = (
            f"Title: {title}\n"
            f"Speaker: {speaker}\n"
            f"Topics: {topics}\n"
            f"Description: {description}\n\n"
            f"{raw_chunk}"
        )

        # Embed
        try:
            emb_resp = client.embeddings.create(
                model="RPRTHPB-text-embedding-3-small",
                input=chunk_for_embedding
            )
            emb = emb_resp.data[0].embedding
        except Exception as e:
            print(f"[Embedding ERROR] talk_id={talk_id} chunk={i} error={e}")
            continue

        # IMPORTANT: keep SAME ID format so upsert overwrites old vectors
        vectors_batch.append({
            "id": f"{talk_id}_{i}",
            "values": emb,
            "metadata": {
                "talk_id": talk_id,
                "title": title,
                "speaker": speaker,
                "topics": topics,
                "description": description,
                "url": url,
                "chunk_index": i,
                "chunk": raw_chunk[:1200],
            }
        })

        total_chunks += 1

        if len(vectors_batch) >= BATCH_SIZE:
            index.upsert(vectors=vectors_batch)
            print(f"Upserted batch of {len(vectors_batch)} vectors (total chunks so far: {total_chunks})")
            vectors_batch = []

if vectors_batch:
    index.upsert(vectors=vectors_batch)
    print(f"Upserted final batch of {len(vectors_batch)} vectors (total chunks: {total_chunks})")

print("✅ Subset ingestion done (metadata updated).")
