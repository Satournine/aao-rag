**🧾 AAO-RAG: Retrieval-Augmented Legal Assistant for EB-1 Visa Petitions**

AAO-RAG is a Retrieval-Augmented Generation (RAG) pipeline designed to help users explore USCIS AAO (Administrative Appeals Office) decisions for EB-1 Extraordinary Ability visa petitions. The system combines PDF parsing, vector embedding, semantic search, and local language model generation with citation-aware answers.

---

### 📦 Features

* Crawls and downloads EB-1 AAO decision PDFs from the USCIS website
* Parses AAO decision PDFs and extracts clean legal text with metadata
* Embeds legal content using BAAI/bge-small-en-v1.5
* Indexes text chunks with FAISS for fast semantic search
* Answers legal questions using mistralai/Mistral-7B-Instruct-v0.2
* Returns grounded, concise responses with citation metadata
* Fully local, no API calls needed

---

### 📁 Project Structure

```
├── data/                     # Raw PDFs, processed JSONL chunks
│   ├── raw/                  # Downloaded PDFs
│   └── processed/            # Cleaned, chunked, and structured text
├── models/                  # Place for quantized LLM GGUF files
├── notebooks/               # Development notebooks (e.g., Ask_RAG.ipynb)
├── scripts/                 # All processing scripts
│   ├── crawl_aao.py         # Crawl AAO website and download PDFs
│   ├── process_pdfs.py      # Parse and chunk PDF documents
│   ├── embed_index.py       # Embed and create FAISS index
│   └── ask_rag.py           # Query engine CLI interface
├── data/faiss/              # FAISS index and metadata
│   ├── faiss_index.index    # Saved FAISS vector index
│   └── metadata.pkl         # Corresponding metadata
├── requirements.txt         # Python dependencies
├── LICENSE                  # License info
└── README.md                # You're here
```

---

### 🚀 Getting Started

#### 🔧 Environment Setup

1. Clone the repo:

```bash
git clone https://github.com/yourusername/aao-rag.git
cd aao-rag
```

2. (Optional) Create a virtual environment:

```bash
python3 -m venv casenv
source casenv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

### 🏗️ Pipeline Steps

#### 1. PDF Crawling, Processing & Indexing

```bash
python scripts/main.py
```

* Crawls and downloads AAO PDFs to `data/raw/`
* Parses PDFs and extracts metadata and clean text chunks
* Saves to `data/processed/aaos.jsonl`
* Creates `faiss_index.index` and `metadata.pkl`

#### 2. Asking Legal Questions

```bash
python scripts/ask_rag.py
```

* Interactive CLI: type questions, get answers with full metadata citations
* Type `exit` to quit

---

### 🪪 License

This project is licensed under the MIT License. See LICENSE for details.
