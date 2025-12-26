def chunk_text(text, chunk_size=1024, overlap_ratio=0.2):
    overlap = int(chunk_size * overlap_ratio)
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks
