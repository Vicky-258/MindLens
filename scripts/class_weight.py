import json
import torch
import numpy as np
from collections import Counter
from pathlib import Path

# load label map
with open("configs/labels.json") as f:
    label2id = json.load(f)

num_labels = len(label2id)

# count labels
label_counts = Counter()
with open("data/processed/span_ner.jsonl") as f:
    for line in f:
        sample = json.loads(line)
        label_counts.update(sample["labels"])

counts = np.array(
    [label_counts.get(i, 0) for i in range(num_labels)],
    dtype=np.float64
)

counts[counts == 0] = 1.0

alpha = 0.5
raw_weights = 1.0 / (counts ** alpha)
weights = raw_weights / raw_weights.mean()

class_weights = torch.tensor(weights, dtype=torch.float)

# 🔒 SAVE
Path("configs").mkdir(exist_ok=True)
torch.save(
    {
        "alpha": alpha,
        "class_weights": class_weights,
        "label_counts": dict(label_counts),
    },
    "configs/class_weights.pt"
)

print("✅ Saved class weights to configs/class_weights.pt")
