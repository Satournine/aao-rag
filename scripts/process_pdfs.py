import os
import re
import json
import fitz  # PyMuPDF
import uuid
from tqdm import tqdm

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

def extract_date(text):
    match = re.search(r"Date:\s*(FEB\.?|February)\s+(\d{1,2}),?\s+2025", text, re.IGNORECASE)
    if match:
        return f"February {int(match.group(2))}, 2025"
    return "February 2025"

def extract_case_id(text, fallback_name):
    match = re.search(r"\b(FEB\d{2}2025_\d{2}[A-Z]\d{4})\b", text)
    return match.group(1) if match else os.path.splitext(fallback_name)[0]

def extract_a_number(text):
    match = re.search(r"In Re:\s*(\d+)", text)
    return match.group(1) if match else None

def extract_petition_type(text):
    match = re.search(r"Form\s+I-140.*?Extraordinary Ability.*", text)
    return match.group(0).strip() if match else None

def extract_occupation(text):
    match = re.search(r"The Petitioner, an? (.+?),", text)
    return match.group(1) if match else None

def extract_decision_sections(text):
    return re.findall(r"(?<=\n)([IVXLC]+\.\s[A-Z][^\n]+)", text)

def extract_outcomes(text):
    return re.findall(r"(ORDER:[^\n]+)", text)

def extract_criteria(text, keyword):
    pattern = {
        "claimed": r"(?:claim|assert|maintain)[^.\n]+?(\(\w+\)(?:,? ?\(\w+\))*)",
        "met": r"(?:met|satisfied)[^.\n]+?criteria[^.\n]*?(\(\w+\)(?:,? ?\(\w+\))*)",
        "failed": r"(?:did not show|not established|has not met|did not meet)[^.\n]*?(\(\w+\)(?:,? ?\(\w+\))*)"
    }.get(keyword)
    if not pattern:
        return []
    matches = re.findall(pattern, text, re.IGNORECASE)
    flat = [m for match in matches for m in re.findall(r"\(\w+\)", match)]
    return list(set(flat))

def chunk_text(text, max_words=300):
    paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 40]
    chunks = []
    buf = ""
    for para in paragraphs:
        if len(buf.split()) + len(para.split()) < max_words:
            buf += " " + para
        else:
            chunks.append(buf.strip())
            buf = para
    if buf:
        chunks.append(buf.strip())
    return chunks

def process_all_pdfs(input_dir="data/raw", output_path="data/processed/aaos.jsonl"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    records = []
    for fname in tqdm(os.listdir(input_dir)):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(input_dir, fname)
        try:
            text = extract_text_from_pdf(path)
            case_id = extract_case_id(text, fname)
            date = extract_date(text)
            a_number = extract_a_number(text)
            petition_type = extract_petition_type(text)
            occupation = extract_occupation(text)
            decision_sections = extract_decision_sections(text)
            outcomes = extract_outcomes(text)
            criteria_claimed = extract_criteria(text, "claimed")
            criteria_met = extract_criteria(text, "met")
            criteria_failed = extract_criteria(text, "failed")
            chunks = chunk_text(text)

            for i, chunk in enumerate(chunks):
                records.append({
                    "id": str(uuid.uuid4()),
                    "case_id": case_id,
                    "date": date,
                    "a_number": a_number,
                    "petition_type": petition_type,
                    "occupation": occupation,
                    "decision_type": decision_sections,
                    "outcome": outcomes,
                    "criteria_claimed": criteria_claimed,
                    "criteria_met": criteria_met,
                    "criteria_failed": criteria_failed,
                    "chunk_id": f"{case_id}_chunk{i}",
                    "text": chunk
                })
        except Exception as e:
            print(f"[!] Failed to process {fname}: {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            json.dump(record, f)
            f.write("\n")

    print(f"[+] Saved {len(records)} records to {output_path}")