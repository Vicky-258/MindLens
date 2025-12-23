# %%

import json

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)
            
# %%
with open("../configs/labels.json", "r") as f:
    label2id = json.load(f)

# %%
data = list(load_jsonl("../data/processed/span_ner.jsonl"))
total_tokens = sum(len(sample["labels"]) for sample in data)
print("Total tokens:", total_tokens)

# %%
O_ID = 0

prop_tokens = 0
non_prop_tokens = 0

for sample in data:
    for lid in sample["labels"]:
        if lid == O_ID:
            non_prop_tokens += 1
        else:
            prop_tokens += 1

print("Propaganda tokens:", prop_tokens)
print("Non-propaganda tokens:", non_prop_tokens)
print("Propaganda ratio:", prop_tokens / (prop_tokens + non_prop_tokens))

# %%
from collections import Counter

label_counter = Counter()

# Phase 1: accumulate
for sample in data:
    label_counter.update(sample["labels"])

# Phase 2: prepare mapping once
id2label = {v: k for k, v in label2id.items()}

# Phase 3: print once
for lid, count in label_counter.most_common():
    print(f"{id2label[lid]:40} {count}")

# %%

technique_counter = Counter()

for lid, count in label_counter.items():
    if lid == O_ID:
        continue
    label = id2label[lid]
    technique = label[2:]  # remove B- / I-
    technique_counter[technique] += count

for tech, count in technique_counter.most_common():
    print(f"{tech:45} {count}")

# %%

RARE_THRESHOLD = 100  # you can adjust

rare = {
    tech: cnt
    for tech, cnt in technique_counter.items()
    if cnt < RARE_THRESHOLD
}

print("Rare techniques:")
for tech, cnt in rare.items():
    print(f"{tech:45} {cnt}")
# %%
from transformers import DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

import random

sample = random.choice(data)
tokens = tokenizer.convert_ids_to_tokens(sample["input_ids"])
labels = [id2label[l] for l in sample["labels"]]

for t, l in zip(tokens, labels):
    if l != "O":
        print(f"{t:15} {l}")
        
# %%

import json
from collections import Counter

label_counts = Counter()

with open("../data/processed/span_ner.jsonl") as f:
    for line in f:
        sample = json.loads(line)
        label_counts.update(sample["labels"])

# map id → label name
id2label = {v: k for k, v in label2id.items()}

for lid, count in label_counts.most_common():
    print(f"{id2label[lid]:40s} {count}")
