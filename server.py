from fastapi import FastAPI
from pydantic import BaseModel

from config import CHUNK_SIZE, OVERLAP_RATIO, TOP_K
from api_pinecone import retrieve_context
from api_prompt import answer_question

app = FastAPI(title="TedRag", version="1.0")


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"status": "ok", "message": "Go to /docs for Swagger UI"}


@app.get("/api/stats")
def stats():
    return {
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K
    }


@app.post("/api/pinecone")
def pinecone_search(q: QuestionRequest):
    context = retrieve_context(question=q.question, top_k=TOP_K, include_metadata=True)
    return {"context": context}


@app.post("/api/prompt")
def prompt(q: QuestionRequest):
    # Now Swagger will show {"question": "string"} instead of additionalProp1
    return answer_question(q.question)
