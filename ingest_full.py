import os
import ast
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

from chunking import chunk_text
from config import CHUNK_SIZE, OVERLAP_RATIO

load_dotenv()

# ---------------- Clients ----------------
client = OpenAI()  # uses OPENAI_API_KEY + OPENAI_BASE_URL
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

EMBED_MODEL = os.getenv("EMBED_MODEL", "RPRTHPB-text-embedding-3-small")

# ---------------- Settings ----------------
FULL_CSV_PATH = "data/ted_talks_en.csv"
BATCH_SIZE = 50
CHUNK_STORE_LIMIT = 1200

# 👉 SAFETY LIMIT
LIMIT = 500

# ---------------- Helpers ----------------
def parse_topics(x):
    if pd.isna(x):
        return ""
    if isinstance(x, list):
        return ", ".join(map(str, x))
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                arr = ast.literal_eval(s)
                if isinstance(arr, list):
                    return ", ".join(map(str, arr))
            except Exception:
                pass
        return s
    return str(x)

def talk_already_ingested(talk_id: str) -> bool:
    """Check if talk_id_0 already exists in Pinecone."""
    try:
        res = index.fetch(ids=[f"{talk_id}_0"])
        return f"{talk_id}_0" in (res.get("vectors") or {})
    except Exception:
        return False

# ---------------- Load data ----------------
df = pd.read_csv(FULL_CSV_PATH).head(LIMIT)

required_cols = ["talk_id", "title", "speaker_1", "transcript"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# ---------------- Ingest loop ----------------
vectors_batch = []
total_chunks = 0
processed_talks = 0
skipped_talks = 0

for _, row in df.iterrows():
    talk_id = str(row.get("talk_id", "")).strip()
    if not talk_id:
        continue

    if talk_already_ingested(talk_id):
        skipped_talks += 1
        continue

    title = str(row.get("title", "")).strip()
    speaker = str(row.get("speaker_1", "")).strip()
    transcript = row.get("transcript", "")

    if not isinstance(transcript, str) or not transcript.strip():
        continue

    description = str(row.get("description", "")).strip()
    url = str(row.get("url", "")).strip()
    topics = parse_topics(row.get("topics", ""))

    chunks = chunk_text(transcript, CHUNK_SIZE, OVERLAP_RATIO)

    for i, raw_chunk in enumerate(chunks):
        text_for_embedding = (
            f"Title: {title}\n"
            f"Speaker: {speaker}\n"
            f"Topics: {topics}\n"
            f"Description: {description}\n\n"
            f"{raw_chunk}"
        )

        try:
            emb = client.embeddings.create(
                model=EMBED_MODEL,
                input=text_for_embedding
            ).data[0].embedding
        except Exception as e:
            print(f"[Embedding error] talk_id={talk_id} chunk={i}: {e}")
            continue

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
                "chunk": raw_chunk[:CHUNK_STORE_LIMIT],
            }
        })

        total_chunks += 1

        if len(vectors_batch) >= BATCH_SIZE:
            index.upsert(vectors=vectors_batch)
            print(
                f"Upserted {len(vectors_batch)} | "
                f"talks={processed_talks} | "
                f"skipped={skipped_talks} | "
                f"chunks={total_chunks}"
            )
            vectors_batch = []

    processed_talks += 1

# ---------------- Flush ----------------
if vectors_batch:
    index.upsert(vectors=vectors_batch)

print("\n✅ INGEST COMPLETE")
print(f"Processed talks: {processed_talks}")
print(f"Skipped talks:   {skipped_talks}")
print(f"Total chunks:    {total_chunks}")
