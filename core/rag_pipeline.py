from core.document_loader import DocumentLoader
from core.embedding import EmbeddingModel
from core.vector_store import FAISSVectorStore
from core.llm import HuggingFaceLLM
from core.models import Query, Chunk, RetrievedDocument, Answer, Document
from typing import List
import time

class RAGPipeline:
    def __init__(self, embedding_model_name: str = None, llm_model_name: str = None, embedding_dim: int = 384):
        self.embedding = EmbeddingModel(model_name=embedding_model_name or 'sentence-transformers/all-MiniLM-L6-v2')
        self.vector_store = FAISSVectorStore(embedding_dim)
        self.llm = HuggingFaceLLM(model_name=llm_model_name or 'google/flan-t5-base')
        self.documents = []

    def add_documents(self, paths: List[str]):
        docs = DocumentLoader.load_documents(paths)
        chunks = []
        for doc in docs:
            # Dividir documento em chunks menores
            doc_chunks = self._split_document(doc)
            for i, chunk_content in enumerate(doc_chunks):
                chunk = Chunk(id=f"{doc.id}_chunk_{i}", document_id=doc.id, content=chunk_content)
                chunk.embedding = self.embedding.embed([chunk.content])[0]
                chunks.append(chunk)
        self.vector_store.add_chunks(chunks)
        self.documents.extend(docs)
        return docs

    def _split_document(self, doc: Document, chunk_size: int = 1000) -> List[str]:
        """Divide um documento em chunks menores baseado no número de caracteres"""
        content = doc.content
        chunks = []
        
        # Dividir por parágrafos primeiro (se possível)
        paragraphs = content.split('\n\n')
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Se o parágrafo sozinho já é muito grande, dividi-lo
            if len(paragraph) > chunk_size:
                # Salvar chunk atual se não estiver vazio
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # Dividir parágrafo grande em pedaços menores
                words = paragraph.split()
                temp_chunk = ""
                for word in words:
                    if len(temp_chunk + " " + word) > chunk_size and temp_chunk:
                        chunks.append(temp_chunk.strip())
                        temp_chunk = word
                    else:
                        temp_chunk += " " + word if temp_chunk else word
                
                if temp_chunk.strip():
                    chunks.append(temp_chunk.strip())
            
            # Se adicionar este parágrafo não exceder o limite
            elif len(current_chunk + "\n\n" + paragraph) <= chunk_size:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
            else:
                # Salvar chunk atual e começar novo
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
        
        # Adicionar último chunk se não estiver vazio
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [content[:chunk_size]]

    def query(self, question: str, top_k: int = 5) -> Answer:
        start = time.time()
        query_embedding = self.embedding.embed([question])[0]
        results = self.vector_store.search(query_embedding, top_k=top_k)
        retrieved = [RetrievedDocument(chunk=chunk, relevance_score=score) for chunk, score in results]
        
        # Limitar o contexto para não exceder o limite do modelo
        context_parts = []
        total_length = 0
        max_context_length = 400  # Deixar espaço para pergunta e resposta
        
        for r in retrieved:
            chunk_text = r.chunk.content
            if total_length + len(chunk_text) <= max_context_length:
                context_parts.append(chunk_text)
                total_length += len(chunk_text)
            else:
                # Adicionar parte do chunk se couber
                remaining = max_context_length - total_length
                if remaining > 50:  # Só adicionar se for significativo
                    context_parts.append(chunk_text[:remaining] + "...")
                break
        
        context = '\n'.join(context_parts)
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        answer_text = self.llm.generate(prompt)
        end = time.time()
        return Answer(answer=answer_text, retrieved_documents=retrieved, processing_time=end-start)
