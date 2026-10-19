# Splinter: Local Document Q&A with Hybrid RAG

Ask questions about a PDF or text file and get answers based only on its content.
Everything runs **locally** (no API keys): embeddings with Hugging Face, answers with [Ollama](https://ollama.com).

The retrieval step combines **semantic search** (vector embeddings) and **keyword search** (BM25),
and uses an LLM to rewrite each question into several variations before searching.

## Features

- Loads PDF and `.txt` files
- Splits text into overlapping chunks that respect paragraph and sentence boundaries
- **Hybrid retrieval**: vector search (Chroma) + keyword search (BM25), merged with an ensemble retriever
- **Multi-query**: the LLM generates alternative versions of the question to improve recall
- Answers only from the retrieved context, and says "I don't know" when the answer is missing
- Streams the answer as it is generated
- Runs fully offline after the models are downloaded

## How it works

```
PDF / TXT
   |
   v
Load  ->  Split into chunks (1000 chars, 200 overlap)
   |
   v
Embed chunks (all-MiniLM-L6-v2)  ->  Chroma vector store
   |                                      |
   |                                      v
   +--------------------->  BM25 index    Vector retriever (+ MultiQuery)
                                |                 |
                                v                 v
                         BM25 retriever (+ MultiQuery)
                                \               /
                                 Ensemble retriever (50/50)
                                        |
                                        v
                      Top chunks + question -> prompt -> Ollama (llama3.2)
                                        |
                                        v
                                 Streamed answer
```

## Tech stack

| Part | Tool |
|---|---|
| Language | Python |
| Framework | LangChain |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face) |
| Vector store | Chroma |
| Keyword search | BM25 (`rank_bm25`) |
| LLM | `llama3.2` via Ollama |
| PDF parsing | PyPDF |

## Setup

1. Install [Ollama](https://ollama.com) and download the model:

   ```bash
   ollama pull llama3.2
   ```

2. Clone the project and install the dependencies:

   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   pip install langchain langchain-classic langchain-community langchain-chroma \
               langchain-huggingface langchain-ollama langchain-text-splitters \
               chromadb sentence-transformers pypdf rank_bm25 ipython jupyter
   ```

3. Put your PDF next to the notebook (the PDF is not included in this repository).

The embedding model is downloaded automatically from Hugging Face on the first run.

## Usage

Open `ask_doc.ipynb`, or use the helper functions directly:

```python
from rag_utils import temps

# 1. Load and split the document
docs = temps.load_documents("./your_file.pdf")
chunks = temps.split_documents(docs, chunk_size=1000, chunk_overlap=200)

# 2. Build the hybrid retriever
vector_store = temps.get_vector_store(chunks)
retriever = temps.get_EnsembleRetriever(
    vector_retriever=temps.get_retriever(get_vector_store=vector_store),
    BM25_retriever=temps.get_BM25retriever(chunks=chunks),
)

# 3. Ask a question
question = "Histogram Equalization"
result = retriever.invoke(question)
context = "\n\n".join(d.page_content for d in result)

temps.ask_ollama(
    prom=(
        "Answer the question using only the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    ),
    result=result,
)
```

The answer is streamed to the output as the model generates it.

## Results

> **TODO:** fill this table with your own measurements.
> Pick 5 questions whose answer page you know in the PDF. For each retriever setup,
> record whether the correct page was returned and how long the query took.

| Setup | Correct page found (out of 5) | Avg. time per query |
|---|---|---|
| Vector only | TODO | TODO |
| Vector + BM25 | TODO | TODO |
| Vector + BM25 + MultiQuery (current) | TODO | TODO |

## Known limitations

- The vector store is **in memory**: the index is rebuilt every time the notebook runs.
- Answers do not show their **source pages** yet.
- Built for **specific questions**. Requests like "summarize the whole document" work poorly,
  because retrieval only returns a few chunks.
- `PyPDFLoader` may not preserve **tables** well, and scanned PDFs (images) are not supported.
- MultiQuery adds extra LLM calls for every question, so responses are slower on local models.
- No automated evaluation yet.

## Roadmap

- [ ] Persistent Chroma store
- [ ] Show source file and page number with each answer
- [ ] Evaluate retrieval and answers with [RAGAS](https://docs.ragas.io)
- [ ] Add a reranking step after retrieval
- [ ] Route summary requests to a map-reduce summarizer
- [ ] Simple web interface (Streamlit)

## Project structure

```
.
├── rag_utils.py     # helper functions: load, split, embed, retrievers, ask
├── ask_doc.ipynb    # example notebook
└── README.md
```
