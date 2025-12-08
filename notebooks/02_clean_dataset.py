# %% Load raw dataset
import json
from pathlib import Path

RAW_TRAIN = Path("../data/raw/train.jsonl")

def load_jsonl(path):
    return [json.loads(line) for line in path.open("r", encoding="utf-8")]

train_raw = load_jsonl(RAW_TRAIN)
len(train_raw)

# %% Text cleaning functions
import re
def clean_text(text: str):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)         # remove URLs
    text = re.sub(r"<.*?>", "", text)                    # remove HTML tags
    text = re.sub(r"\s+", " ", text).strip()             # normalize whitespace
    return text

# %% Label cleaning functions
def clean_spans(spans, text_length):
    cleaned = []
    for s in spans:
        if 0 <= s["start"] < s["end"] <= text_length:
            cleaned.append(s)
        # else: drop invalid span
    return cleaned
def clean_techniques(techs, text_length):
    cleaned = []
    for t in techs:
        if 0 <= t["start"] < t["end"] <= text_length:
            cleaned.append(t)
    return cleaned

# %% Build cleaned dataset
PROCESSED_TRAIN = Path("../data/processed/train.jsonl")
PROCESSED_TRAIN.parent.mkdir(parents=True, exist_ok=True)

cleaned_records = []

for item in train_raw:
    raw_text = item["text"]
    cleaned_text = clean_text(raw_text)

    # offsets are kept from raw text, so use original length
    text_length = len(raw_text)

    si_clean = clean_spans(item["span_identification"], text_length)
    tc_clean = clean_techniques(item["technique_classification"], text_length)

    cleaned_records.append({
        "id": item["id"],
        "text": cleaned_text,
        "span_identification": si_clean,
        "technique_classification": tc_clean
    })

with PROCESSED_TRAIN.open("w", encoding="utf-8") as f:
    for rec in cleaned_records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

len(cleaned_records), "saved!"
