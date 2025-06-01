from process_pdfs import process_all_pdfs
from embed_index import embed_and_index

if __name__ == "__main__":
    print("[*] Starting AAO PDF processing...")
    process_all_pdfs()
    print("[✓] All PDFs processed into aaos.jsonl")

    print("[*] Embedding and indexing chunks...")
    embed_and_index()
    print("[✓] Chroma DB saved.")

    # Preview first 3 entries
    import json

    print("\n--- Preview of Parsed Records ---")
    with open("data/processed/aaos.jsonl", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            record = json.loads(line)
            print(f"Record {i+1}:")
            print(f"  Case ID: {record.get('case_id')}")
            print(f"  Date: {record.get('date')}")
            print(f"  Occupation: {record.get('occupation')}")
            print(f"  Text Snippet: {record.get('text')[:200]}...\n")