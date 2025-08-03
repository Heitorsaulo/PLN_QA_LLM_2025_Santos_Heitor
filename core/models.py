from pydantic import BaseModel
from typing import List, Optional

class Document(BaseModel):
    id: str
    content: str
    metadata: Optional[dict] = None

class Chunk(BaseModel):
    id: str
    document_id: str
    content: str
    embedding: Optional[List[float]] = None

class Query(BaseModel):
    question: str
    top_k: int = 3

class RetrievedDocument(BaseModel):
    chunk: Chunk
    relevance_score: float

class Answer(BaseModel):
    answer: str
    retrieved_documents: List[RetrievedDocument]
    processing_time: Optional[float] = None
