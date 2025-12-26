import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX", "ted-rag-full")
DIMENSIONS = 1536

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

existing = [idx["name"] for idx in pc.list_indexes()]
if INDEX_NAME in existing:
    print(f"Index '{INDEX_NAME}' already exists ✅")
else:
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSIONS,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=os.getenv("PINECONE_CLOUD", "aws"),
            region=os.getenv("PINECONE_REGION", "us-east-1")
        )
    )
    print(f"Creating index '{INDEX_NAME}'...")

# Wait until ready
while True:
    desc = pc.describe_index(INDEX_NAME)
    if desc.status["ready"]:
        print(f"Index '{INDEX_NAME}' is ready ✅")
        break
