from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

resp = client.embeddings.create(
    model="RPRTHPB-text-embedding-3-small",
    input="hello from ted rag"
)

embedding = resp.data[0].embedding
print("Embedding length:", len(embedding))
print("First 5 numbers:", embedding[:5])
