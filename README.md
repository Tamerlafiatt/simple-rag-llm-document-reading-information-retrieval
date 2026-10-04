# Simple RAG LLM Usage for Document Reading and Information Retrieval

A small educational **Retrieval-Augmented Generation (RAG)** project that reads PDF documents, retrieves the most relevant passages for a question, and uses a **local LLM** to generate a grounded answer.

The project was built step by step to make each RAG component visible instead of hiding the pipeline behind a high-level framework.

## Architecture

```text
PDF document
    ↓
PyMuPDF text extraction
    ↓
LangChain recursive chunking
    ↓
BAAI/bge-small-en-v1.5 embeddings
    ↓
FAISS vector index
    ↓
Top-k semantic retrieval
    ↓
Retrieved context + user question
    ↓
Qwen3:4B through Ollama
    ↓
Grounded answer + source pages
```

## What the project demonstrates

- PDF ingestion and page-level metadata
- recursive text chunking with overlap
- local sentence embeddings
- semantic similarity search
- FAISS nearest-neighbor retrieval
- Retrieval-Augmented Generation
- local LLM inference with Ollama
- source/page tracking for retrieved evidence

## Technologies

| Component | Tool |
|---|---|
| Language | Python 3.11 |
| PDF extraction | PyMuPDF |
| Text splitting | LangChain Text Splitters |
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Vector search | FAISS |
| Local LLM | `qwen3:4b` |
| LLM runtime | Ollama |
| Experimentation | Jupyter Notebook |

## Project structure

```text
simple-rag-llm-document-reading-information-retrieval/
├── data/
│   └── documents/
│       └── .gitkeep
├── notebooks/
│   └── simple_rag.ipynb
├── src/
│   └── rag.py
├── .gitignore
├── README.md
└── requirements.txt
```

PDF files are ignored by Git by default so private documents are not accidentally committed.

## How RAG works in this project

### 1. Document ingestion

PyMuPDF opens the PDF and extracts the text page by page.

Each page keeps metadata such as:

```python
{
    "page": 7,
    "source": "document.pdf"
}
```

### 2. Chunking

Large documents are split into smaller overlapping passages using:

```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
```

The overlap helps preserve context across chunk boundaries.

### 3. Embeddings

Each chunk is converted into a semantic vector with:

```text
BAAI/bge-small-en-v1.5
```

The vectors are normalized so inner-product search can be used as cosine-style semantic similarity.

### 4. Vector retrieval

FAISS stores all chunk embeddings.

A user question is embedded using the same model and FAISS returns the top-k nearest chunks.

### 5. Augmentation

The retrieved passages are combined into a context block and inserted into the prompt together with the user question.

### 6. Generation

A local `qwen3:4b` model running through Ollama produces the final answer.

The prompt explicitly asks the model to use only the retrieved context and to say when the answer is not available in the document.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/simple-rag-llm-document-reading-information-retrieval.git
cd simple-rag-llm-document-reading-information-retrieval
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install Ollama

Install Ollama, then download the local model:

```bash
ollama pull qwen3:4b
```

You can test it with:

```bash
ollama run qwen3:4b
```

## Run the project

Place a PDF at:

```text
data/documents/document.pdf
```

Then run:

```bash
python src/rag.py
```

Or open:

```text
notebooks/simple_rag.ipynb
```

and execute the notebook step by step.

## Example

```python
from src.rag import SimpleRAG

rag = SimpleRAG("data/documents/document.pdf")

result = rag.answer(
    "What dataset is used in the document?"
)

print(result["answer"])

for source in result["sources"]:
    print(source)
```

Example output:

```text
The document uses ...

Sources:
- document.pdf, page 7
- document.pdf, page 8
```

## Retrieval function

The core retriever performs:

```text
question
   ↓
embedding
   ↓
FAISS nearest-neighbor search
   ↓
top-k relevant chunks
```

The retrieval score is a similarity score, **not a probability or model confidence score**.

## Why use a local LLM?

Using Ollama allows the generation step to run locally.

Benefits include:

- no LLM API key required
- local/private inference
- easier experimentation with open models
- useful foundation for offline document assistants

The first download of the embedding model and Qwen model still requires internet access.

## Current limitations

This is intentionally a simple learning implementation.

Current limitations include:

- one PDF at a time
- in-memory FAISS index
- no reranker
- no formal RAG evaluation
- no hybrid lexical + semantic search
- no persistent vector database
- no web UI
- retrieved chunks are supplied directly to the LLM without advanced context compression

## Possible next steps

- multi-document ingestion
- persistent FAISS or Qdrant storage
- metadata filtering
- reranking
- RAGAS evaluation
- citation validation
- FastAPI backend
- Streamlit or React interface
- LangGraph / agentic RAG
- query rewriting and retrieval retry
- hybrid BM25 + vector retrieval

## Goal

The goal of this repository is to understand the mechanics of RAG from first principles:

**retrieve relevant information first, then let the LLM generate an answer from that evidence.**
