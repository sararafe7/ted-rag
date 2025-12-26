import pandas as pd
from chunking import chunk_text
from config import CHUNK_SIZE, OVERLAP_RATIO

df = pd.read_csv("data/50 talks.csv")   # your 50-talk csv
row = df.iloc[1]
chunks = chunk_text(row["transcript"], CHUNK_SIZE, OVERLAP_RATIO)

print("Title:", row["title"])
print("Chunks:", len(chunks))

print("\n--- chunk 0 last 250 ---\n", chunks[0][-250:])
print("\n--- chunk 1 first 250 ---\n", chunks[1][:250])
