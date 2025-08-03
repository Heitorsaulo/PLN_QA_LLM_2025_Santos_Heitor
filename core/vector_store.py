import faiss
import numpy as np
from typing import List, Tuple
from core.models import Chunk

class FAISSVectorStore:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.chunks = []

    def add_chunks(self, chunks: List[Chunk]):
        embeddings = np.array([c.embedding for c in chunks]).astype('float32')
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[Chunk, float]]:
        query = np.array([query_embedding]).astype('float32')
        D, I = self.index.search(query, top_k)
        results = []
        for idx, score in zip(I[0], D[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], score))
        return results
