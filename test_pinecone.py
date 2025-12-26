import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = os.getenv("PINECONE_INDEX")  # should be "ted-rag"
print("Index name from env:", index_name)

index = pc.Index(index_name)
stats = index.describe_index_stats()
print(stats)
