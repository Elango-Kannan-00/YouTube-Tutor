import faiss
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import os

model = SentenceTransformer('all-MiniLM-L6-v2')

# Load Gemini API key from environment variable
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm = genai.GenerativeModel("gemini-pro")

def build_vector_store(embeddings):
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index

def get_top_chunks(question, chunks, chunk_embeddings, index, k=3):
    q_embedding = model.encode([question])
    D, I = index.search(np.array(q_embedding), k)
    return [chunks[i] for i in I[0]]

def ask_gemini(question, context_chunks):
    prompt = f"""
You're an educational AI assistant. Answer the following question using the context below.

Context:
{''.join(context_chunks)}

Question: {question}
Answer:"""
    response = llm.generate_content(prompt)
    return response.text
