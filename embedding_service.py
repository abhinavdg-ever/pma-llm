from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

class TextPayload(BaseModel):
    text: str

app = FastAPI()

# Updated to use all-mpnet-base-v2 for better semantic understanding
# This produces 768-dimensional vectors (vs 384 for MiniLM)
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

@app.post("/embed")
async def embed(payload: TextPayload):
    embedding = model.encode(payload.text).tolist()
    return {"embedding": embedding}

