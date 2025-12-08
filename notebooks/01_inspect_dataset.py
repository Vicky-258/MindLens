# %% Load dataset
import json
from pathlib import Path

train_path = Path("../data/raw/train.jsonl")

def load_jsonl(path):
    return [json.loads(line) for line in path.open("r", encoding="utf-8")]

train = load_jsonl(train_path)
len(train)
# %% Peek
import pprint
pprint.pprint(train[0])
# %% Technique distribution
from collections import Counter

techs = []
for item in train:
    for span in item["technique_classification"]:
        techs.append(span["technique"])

Counter(techs).most_common(20)
# %% span lengths
lengths = []
for item in train:
    for span in item["span_identification"]:
        lengths.append(span["end"] - span["start"])

min(lengths), max(lengths), sum(lengths)/len(lengths)
# %% article lengths
article_lengths = [len(sample["text"]) for sample in train]
min(article_lengths), max(article_lengths), sum(article_lengths)/len(article_lengths)
