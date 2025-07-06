from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

def embed_and_store(text):
    sentences = text.split('. ')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(sentences)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, sentences, model