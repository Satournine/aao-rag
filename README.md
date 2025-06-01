🧾 AAO-RAG: Retrieval-Augmented Legal Assistant for EB-1 Visa Petitions

AAO-RAG is a Retrieval-Augmented Generation (RAG) pipeline designed to help users explore USCIS AAO (Administrative Appeals Office) decisions for EB-1 Extraordinary Ability visa petitions. The system combines PDF parsing, vector embedding, semantic search, and local language model generation with citation-aware answers.

⸻

📦 Features
	•	Parses AAO decision PDFs and extracts clean legal text with metadata
	•	Embeds legal content using BAAI/bge-small-en-v1.5
	•	Indexes text chunks with FAISS for fast semantic search
	•	Answers legal questions using mistralai/Mistral-7B-Instruct-v0.2
	•	Returns grounded, concise responses with citation metadata
	•	Fully local, no API calls needed

⸻

📁 Project Structure

├── data/                     # Raw PDFs, processed JSONL chunks
├── models/                  # Place for quantized LLM GGUF files
├── notebooks/               # Development notebooks (e.g., Ask_RAG.ipynb)
├── scripts/                 # All processing scripts
│   ├── process_pdfs.py      # Parse and chunk PDF documents
│   ├── embed_index.py       # Embed and create FAISS index
│   └── ask_rag.py           # Query engine CLI interface
├── faiss_index.index        # Saved FAISS vector index
├── metadata.pkl             # Corresponding metadata
├── requirements.txt         # Python dependencies
└── README.md                # You're here


⸻

🚀 Getting Started

🔧 Environment Setup
	1.	Clone the repo:

git clone https://github.com/yourusername/aao-rag.git
cd aao-rag

	2.	(Optional) Create a virtual environment:

python3 -m venv casenv
source casenv/bin/activate

	3.	Install dependencies:

pip install -r requirements.txt


⸻

🏗️ Pipeline Steps

1. PDF Processing & Embedding + Indexing

python scripts/main.py

Parses all PDFs under data/raw/, extracts metadata and chunks, saves to data/processed/aaos.jsonl
Creates a FAISS index (faiss_index.index) and metadata (metadata.pkl).

2. Asking Questions

python scripts/ask_rag.py

Interactive CLI: type questions, get answers with citations. Type exit to quit.


🪪 License

This project is licensed under the MIT License. See LICENSE for details.

⸻
