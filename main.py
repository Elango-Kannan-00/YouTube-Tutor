from fastapi import FastAPI
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import vertexai
from vertexai.language_models import TextGenerationModel
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Google Vertex AI
vertexai.init(project="your-gcp-project-id", location="us-central1")
gemini_model = TextGenerationModel.from_pretrained("gemini-1.0-pro")

# Global variables
stored_chunks = []
index = None

# Request schema
class QARequest(BaseModel):
    video_url: str
    question: str

# Helper functions
def get_transcript(video_url):
    video_id = video_url.split("v=")[-1]
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return " ".join([t['text'] for t in transcript])

def split_text(text, max_tokens=200):
    words = text.split()
    return [" ".join(words[i:i+max_tokens]) for i in range(0, len(words), max_tokens)]

def embed_chunks(chunks):
    return embedding_model.encode(chunks, show_progress_bar=False)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    idx = faiss.IndexFlatL2(dim)
    idx.add(np.array(embeddings))
    return idx

def get_relevant_chunks(question, index, chunks, top_k=3):
    q_embed = embedding_model.encode([question])[0]
    D, I = index.search(np.array([q_embed]), top_k)
    return [chunks[i] for i in I[0]]

def ask_gemini(context_chunks, question):
    context = "\n\n".join(context_chunks)
    prompt = f"""
You are an AI tutor. Based on the following YouTube lecture transcript, answer the student's question accurately.

Transcript:
{context}

Question:
{question}

Answer:
"""
    response = gemini_model.predict(prompt, temperature=0.3, max_output_tokens=512)
    return response.text

# API endpoint
@app.post("/ask")
async def ask_question(data: QARequest):
    global stored_chunks, index

    transcript = get_transcript(data.video_url)
    chunks = split_text(transcript)
    embeddings = embed_chunks(chunks)
    index = build_faiss_index(np.array(embeddings))
    stored_chunks = chunks

    relevant = get_relevant_chunks(data.question, index, stored_chunks)
    answer = ask_gemini(relevant, data.question)

    return {"answer": answer}

# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
