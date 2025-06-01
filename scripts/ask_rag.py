
import faiss
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F


FAISS_DIR = "data/faiss"#FAISS DIR
index = faiss.read_index(f"{FAISS_DIR}/faiss_index.index")

with open(f"{FAISS_DIR}/metadata.pkl", "rb") as f:
    metadata_store = pickle.load(f)

texts = metadata_store["documents"]
metadatas = metadata_store["metadatas"]

from sentence_transformers import SentenceTransformer

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
embedder = SentenceTransformer(EMBED_MODEL, device='cuda' if torch.cuda.is_available() else 'cpu')

def encode(texts):
    return embedder.encode(texts, convert_to_numpy=True)


LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

TOP_K = 5

def retrieve(query, top_k=TOP_K):
    query_embedding = encode([query])
    D, I = index.search(query_embedding, top_k)
    retrieved_chunks = []
    for idx in I[0]:
        try:
            retrieved_chunks.append({
                "text": texts[idx],
                "metadata": metadatas[idx]
            })
        except IndexError:
            continue
    return retrieved_chunks

def get_citation(meta, chunk_number=None):
    parts = []
    if "case_id" in meta:
        parts.append(f"Case: {meta['case_id']}")
    if "date" in meta:
        parts.append(f"Date: {meta['date']}")
    if "source" in meta:
        parts.append(f"Source: {meta['source']}")
    if "chunk_id" in meta:
        parts.append(f"Chunk: {meta['chunk_id']}")
    if not parts and chunk_number is not None:
        parts.append(f"Chunk {chunk_number}")
    return "[" + " | ".join(parts) + "]"

def build_prompt(query, retrieved_chunks, max_context_tokens=4096):
    cutoff_chunks = []
    total_tokens = 0
    for i, chunk in enumerate(retrieved_chunks):
        meta = chunk.get("metadata", {})
        case_id = meta.get("case_id", "N/A")
        date = meta.get("date", "N/A")
        chunk_id = meta.get("chunk_id", f"Chunk{i+1}")
        citation = f"[Chunk {i+1} | Case: {case_id} | Date: {date} | ChunkID: {chunk_id}]"

        chunk_text = f"{chunk['text']}\n{citation}\n\n"
        estimated_tokens = len(chunk_text.split())
        if total_tokens + estimated_tokens > max_context_tokens:
            break
        total_tokens += estimated_tokens
        cutoff_chunks.append(chunk_text)

    full_context = "".join(cutoff_chunks)


    prompt = f"""You are an expert legal assistant.

- Answer ONLY the specific user question below.
- Summarize and synthesize facts from the legal context.
- DO NOT repeat or restate the same fact, rule, or regulation, even if found in multiple places.
- DO NOT copy full sentences; paraphrase in your own words.
- ONLY include information directly relevant to the question.
- ALWAYS cite facts with full metadata citation in this exact format:

  [Case: CASE_ID | Date: DATE]

- Your answer should be clear, concise, and avoid unnecessary repetition.

LEGAL CONTEXT:
{full_context}

USER QUESTION:
{query}

### Answer:
"""
    return prompt

CTX_LENGTH = 4096

def ask(query):
    chunks = retrieve(query)
    prompt = build_prompt(query, chunks)

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=CTX_LENGTH).to(model.device)
    if inputs["input_ids"].shape[-1] > CTX_LENGTH:
        inputs["input_ids"] = inputs["input_ids"][:, -CTX_LENGTH:]
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"][:, -CTX_LENGTH:]

    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask", None),
        max_new_tokens=300,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id
    )
    full_output = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = full_output.split("### Answer:")[-1].strip()
    return answer

print("🔍 Ask your legal question (type 'exit' to quit):")
while True:
    try:
        query = input("\nYour question: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Exiting. Goodbye!")
            break
        answer = ask(query)
        print("\nAnswer:\n", answer)
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")
        break

"""Your question: How do recent AAO decisions evaluate an applicant’s Participation as a Judge service criteria?



Answer:
 Recent AAO decisions have evaluated an applicant's Participation as a Judge service criterion by requiring evidence that the applicant was not only selected to serve as a judge but also actually participated in the judging process. The judging duties must have been directly related to the applicant's field. The evidence provided must satisfy all of these elements to meet the plain language requirements of this criterion. If the applicant fails to submit sufficient evidence, the petition may be denied. [Case: Immigrant_Petition_for_Alien_Worker_(Extraordinary_Ability)_-_FEB052025_03B2203_(PDF,_711.56_KB) | Date: February 5, 2025 | ChunkID: Immigrant_Petition_for_Alien_Worker_(Extraordinary_Ability)_-_FEB052025_03B2203_(PDF,_711.56_KB)_chunk4]

Your question:         What characteristics of national or international awards persuade the AAO that they constitute ‘sustained acclaim’?



Answer:
 National or international awards that demonstrate sustained acclaim for an individual in their field of endeavor, as required for an extraordinary ability petition, typically possess the following characteristics:

1. Recognition from reputable organizations: Awards granted by reputable organizations, particularly those with a strong reputation in the field, are more likely to be considered as evidence of sustained acclaim.

2. Competitive selection process: Awards that are the result of a rigorous and competitive selection process, where the individual's achievements are evaluated against those of their peers, are more persuasive in demonstrating sustained acclaim.

3. Prestige and impact: Awards that carry significant prestige within the field and have a broad impact on the individual's career or the industry as a whole are more likely to be considered as evidence of sustained acclaim.

4. Consistency and frequency: Multiple awards, particularly those received over an extended period, can provide stronger evidence of sustained acclaim compared to a single award.

5. International recognition: Awards that have an international scope and reach, as opposed to being limited to a specific country or region, are more likely to be considered as evidence of sustained acclaim due to their broader impact and recognition.

[Metadata Citation: CASE: Immigrant_Petition_for_Alien_Worker_(Extraordinary_Ability)_-_FEB052025_0

"""



