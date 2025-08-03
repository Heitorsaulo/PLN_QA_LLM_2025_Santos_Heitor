# RAG Pipeline com HuggingFace

Sistema de perguntas e respostas baseado em documentos usando RAG (Retrieval-Augmented Generation).

## Funcionalidades

- Carregamento de documentos PDF
- Geração de embeddings com sentence-transformers
- Busca vetorial com FAISS
- Geração de respostas com modelos HuggingFace

## Instalação

```bash
pip install -r requirements.txt
```

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

- transformers
- sentence-transformers
- faiss-cpu
- PyPDF2
- torch