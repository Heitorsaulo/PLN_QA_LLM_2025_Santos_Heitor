# RAG Pipeline com HuggingFace

Sistema de perguntas e respostas baseado em documentos usando RAG (Retrieval-Augmented Generation). 

**Participantes da equipe:**

- Heitor Saulo Dantas Santos
- Itor Carlos Souza Queiroz
- Lanna Luara Novaes Silva
- Lavínia Louise Rosa Santos
- Rômulo Menezes De Santana

## Modelos utilizados

O objetivo deste tutorial é apresentar uma comparação de desempenho de três modelos do Hugging Face ao executar a tarefa de Question Answer (QA). Os modelos utilizados foram:
- [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)
- [google/flan-t5-xl](https://huggingface.co/google/flan-t5-x)
- [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruc)

## Funcionalidades

- Carregamento de documentos PDF
- Geração de embeddings com sentence-transformers
- Busca vetorial com FAISS
- Geração de respostas com modelos HuggingFace

## Instalação

```bash
pip install -r requirements.txt
```
## Fluxo da Aplicação

1. Carregamento e Pré-processamento dos documentos
2. Divisão do texto em chunks
3. Geração dos embeddings dos chunks
4. Armazenamento e indexação dos embeddings no banco vetorial
5. Geração do embedding da pergunta
6. Busca dos chunks mais relevantes
7. Montagem do contexto com os chunks recuperados
8. Geração da Resposta pelo LLM

## Estrutura do Projeto

```
qa_pln_huggingface/
├── core/
│   ├── document_loader.py    # Carregamento de PDFs
│   ├── embedding.py          # Geração de embeddings
│   ├── vector_store.py       # Armazenamento vetorial (FAISS)
│   ├── llm.py               # Modelo de linguagem
│   ├── rag_pipeline.py      # Pipeline principal
│   └── models.py            # Modelos de dados
├── config/
│   └── settings.py          # Configurações
├── data/                    # Documentos PDF
└── main.py                  # Arquivo principal
```

## Uso

1. Coloque seus arquivos PDF na pasta `data/`
2. Execute o sistema:

```bash
python main.py
```

3. Digite sua pergunta quando solicitado

## Configuração

Edite `config/settings.py` para alterar:
- Modelo de embedding
- Modelo de linguagem
- Caminho dos documentos

## Exemplo

```
Pergunta: O que é machine learning?
Resposta: [Baseada no conteúdo dos documentos carregados]
```

## Dependências Principais

- Transformers
- Sentence-transformers
- Faiss-cpu
- PyPDF2
- torch
