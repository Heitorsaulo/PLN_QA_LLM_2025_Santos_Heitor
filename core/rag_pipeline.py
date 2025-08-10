import torch
from core.document_loader import DocumentLoader
from core.embedding import EmbeddingModel
from core.vector_store import FAISSVectorStore
from core.llm import HuggingFaceLLM
from core.models import Query, Chunk, RetrievedDocument, Answer, Document
from typing import List
import time

class RAGPipeline:
    def __init__(self, embedding_model_name: str = None, llm_model_name: str = None, embedding_dim: int = 384, pipeline_name: str = 'text2text-generation'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.embedding = EmbeddingModel(model_name=embedding_model_name or 'sentence-transformers/all-MiniLM-L6-v2')
        self.vector_store = FAISSVectorStore(embedding_dim)
        self.llm = HuggingFaceLLM(model_name=llm_model_name or 'google/flan-t5-large', device=self.device)
        self.documents = []

    def add_documents(self, paths: List[str]):
        docs = DocumentLoader.load_documents(paths)
        chunks = []
        for doc in docs:
            if doc.metadata and doc.metadata.get("chunked", False):
                chunk = Chunk(
                    id=doc.id,
                    document_id=doc.id.split("_chunk_")[0],
                    content=doc.content
                )
                chunk.embedding = self.embedding.embed([chunk.content])[0]
                chunks.append(chunk)
            else:
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
            if len(paragraph) > chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
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
            elif len(current_chunk + "\n\n" + paragraph) <= chunk_size:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [content[:chunk_size]]

    def query(self, question: str, top_k: int = 5) -> Answer:
        start = time.time()
        query_embedding = self.embedding.embed([question])[0]
        results = self.vector_store.search(query_embedding, top_k=top_k)
        retrieved = [RetrievedDocument(chunk=chunk, relevance_score=score) for chunk, score in results]

        # Reserva de tokens para resposta
        reserved_output_tokens = 512
        max_input_tokens = self.llm.max_model_input_tokens
        allowed_input_tokens = max_input_tokens - reserved_output_tokens

        # Tokens da pergunta (para reservar espaço)
        question_tokens = len(self.llm.tokenizer(question)["input_ids"])

        context_parts = []
        total_tokens = 0

        for r in retrieved:
            chunk_text = r.chunk.content
            chunk_tokens = len(self.llm.tokenizer(chunk_text)["input_ids"])

            if total_tokens + chunk_tokens + question_tokens <= allowed_input_tokens:
                context_parts.append(chunk_text)
                total_tokens += chunk_tokens
            else:
                remaining = allowed_input_tokens - total_tokens - question_tokens
                if remaining > 20:
                    enc = self.llm.tokenizer(chunk_text, truncation=True, max_length=remaining, return_tensors='pt')
                    truncated_chunk = self.llm.tokenizer.decode(enc['input_ids'][0], skip_special_tokens=True)
                    context_parts.append(truncated_chunk + "...")
                break

        context = "\n\n".join(context_parts)
        print(context)

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                        Você é um assistente especializado em responder perguntas apenas com base nas informações do CONTEXTO fornecido.
                        Siga estas regras:
                        - Use exclusivamente as informações do CONTEXTO.
                        - Não adicione informações externas nem invente detalhes.
                        - Responda de forma clara, direta e completa.
                        <|eot_id|><|start_header_id|>user<|end_header_id|>
                        [CONTEXTO]
                        {context}
                        
                        [PERGUNTA]
                        {question}
                        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                """

        answer_text = self.llm.generate(prompt, max_new_tokens=reserved_output_tokens)
        end = time.time()
        return Answer(answer=answer_text, retrieved_documents=retrieved, processing_time=end-start)
