import pickle
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

# Config
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased").to("mps")

FAISS_DIR = "data/faiss"
QUERY = "How does AAO evaluate judging criteria in EB1 cases?"
TOP_K = 5

def encode_legalbert(texts):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to("mps")
            output = model(**inputs).last_hidden_state[:, 0, :]  # CLS token
            embeddings.append(output.cpu().numpy()[0])
    return np.array(embeddings)

# Load FAISS index
print("[*] Loading FAISS index and metadata...")
with open(f"{FAISS_DIR}/faiss_index.pkl", "rb") as f:
    index = pickle.load(f)
with open(f"{FAISS_DIR}/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Encode query
print("[*] Encoding query with Legal-BERT...")
vec = encode_legalbert([QUERY])
faiss.normalize_L2(vec)

# Search
print("[*] Searching FAISS index...")
D, I = index.search(vec, TOP_K)

print("\n[*] Top", TOP_K, "results:")
for rank, (score, idx) in enumerate(zip(D[0], I[0]), 1):
    print(f"\n#{rank}: (score: {score:.4f})")
    print(metadata["documents"][idx][:500])
    print("Metadata:", metadata["metadatas"][idx])