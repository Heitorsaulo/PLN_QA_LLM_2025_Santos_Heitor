import sys
from core.rag_pipeline import RAGPipeline
from config.settings import settings
from utils.helpers import setup_logging, format_duration

if __name__ == "__main__":
    setup_logging()
    rag = RAGPipeline(
        embedding_model_name=settings.embedding_model_name,
        llm_model_name=settings.llm_model_name,
        embedding_dim=settings.embedding_dim
    )
    # Exemplo: adicionar documentos
    paths = ["data/doencas_respiratorias_cronicas.pdf"]
    rag.add_documents(paths)
    # Exemplo: consulta
    question = "Qual é a prevalência e o impacto das principais doenças respiratórias crônicas no Brasil, segundo dados do Ministério da Saúde?"
    answer = rag.query(question)
    print(f"Resposta: {answer.answer}")
    print(f"Tempo de processamento: {format_duration(answer.processing_time)}")
    print(f"Documentos recuperados: {len(answer.retrieved_documents)}")
