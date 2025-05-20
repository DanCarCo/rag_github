from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def create_vector_store(chunks):
    vectors = model.encode(chunks)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors))
    return index, vectors, chunks

def search_similar(query, index, chunks, top_k=3):
    query_vector = model.encode([query])
    D, I = index.search(query_vector, top_k)
    return [chunks[i] for i in I[0]]
