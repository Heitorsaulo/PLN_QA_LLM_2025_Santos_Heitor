from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingModel:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=False)
        # Garante que cada embedding está na CPU e como lista de floats
        return [emb.cpu().numpy().tolist() if hasattr(emb, 'cpu') else emb for emb in embeddings]
