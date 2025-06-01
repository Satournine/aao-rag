import os
import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configuration
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DATA_PATH = "data/processed/aaos.jsonl"  # Your JSONL with {"text": ..., "source": ...}
INDEX_OUT = "data/faiss/faiss_index.pkl"
META_OUT = "data/faiss/metadata.pkl"

# Load model
model = SentenceTransformer(EMBED_MODEL)

# Load and encode documents
documents = []
metadatas = []

with open(DATA_PATH, "r") as f:
    for line in tqdm(f, desc="Loading documents"):
        record = json.loads(line)
        documents.append(record["text"])
        metadatas.append({
        "case_id": record.get("case_id", "N/A"),
        "date": record.get("date", "N/A"),
        "chunk_id": record.get("chunk_id", "N/A"),
        "source": record.get("source", "N/A")
    })

embeddings = model.encode(documents, show_progress_bar=True, convert_to_numpy=True)
# ... (your code up to index.add(np.array(embeddings)))
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# DO THIS INSTEAD OF PICKLE:
faiss.write_index(index, "data/faiss/faiss_index.index")

# Save metadata (pickle is OK here)
with open(META_OUT, "wb") as f:
    pickle.dump({
        "documents": documents,
        "metadatas": metadatas
    }, f)

print("Saved FAISS index to: data/faiss/faiss_index.index")
print(f"Saved metadata to: {META_OUT}")