from transformers import DistilBertTokenizerFast
from Span2Bio import spans_to_bio

import json

with open("configs/labels.json", "r") as f:
    label2id = json.load(f)

tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

text = "These corrupt elites are destroying our country."

encoding = tokenizer(
    text,
    return_offsets_mapping=True,
    return_attention_mask=False,
    add_special_tokens=True
)

spans = [
    (6, 20, "Name_Calling,Labeling")
]

out = spans_to_bio(text, spans, tokenizer, label2id)

tokens = tokenizer.convert_ids_to_tokens(out["input_ids"])

for t, l, o in zip(tokens, out["labels"], out["offsets"]):
    print(f"{t:12} {l:35} {o}")