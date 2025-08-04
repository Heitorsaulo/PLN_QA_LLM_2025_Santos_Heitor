import sys
from core.rag_pipeline import RAGPipeline
from config.settings import settings
from utils.helpers import setup_logging, format_duration
import torch
import os

def run_test(model_name, question, file):
    rag = RAGPipeline(
        embedding_model_name=settings.embedding_model_name,
        llm_model_name=model_name,
        embedding_dim=settings.embedding_dim,
    )
    paths = ["data/doencas_respiratorias_cronicas.pdf"]
    rag.add_documents(paths)
    answer = rag.query(question)
    file.write(f"Modelo: {model_name}\n")
    file.write(f"Pergunta: {question}\n")
    file.write(f"Resposta: {answer.answer}\n\n")
    print(f"Modelo: {model_name}")
    print(f"Resposta: {answer.answer}")

if __name__ == "__main__":
    os.environ['HF_HOME'] = 'D:/cache/huggingface'
    setup_logging()
    lista_modelos = ["bigscience/mt0-xl", "google/flan-t5-large", "google/flan-t5-base", "google/flan-t5-small", "google/flan-t5-xl"]
    lista_perguntas = question = ["Quais são os principais fatores de risco evitáveis para o desenvolvimento de doenças respiratórias crônicas?", "Qual é a relação entre rinite alérgica e asma segundo o conceito de “via aérea única”?", "Quais são os critérios utilizados para classificar a segurança do uso de medicamentos durante a gravidez, segundo o FDA?"]

    with open("resultados_teste.txt", "w", encoding="utf-8") as file:
        for question in lista_perguntas:
            for modelo in lista_modelos:
                run_test(modelo, question, file)