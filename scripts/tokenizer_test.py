from transformers import DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

text = """These corrupt elites are destroying our country."""

encoding = tokenizer(
    text,
    return_offsets_mapping=True,
    return_attention_mask=False,
    add_special_tokens=True
)

print(encoding["input_ids"])
print(tokenizer.convert_ids_to_tokens(encoding["input_ids"]))

for token, (start, end) in zip(
    tokenizer.convert_ids_to_tokens(encoding["input_ids"]),
    encoding["offset_mapping"]
):
    print(f"{token:15} -> ({start}, {end}) -> '{text[start:end]}'")
