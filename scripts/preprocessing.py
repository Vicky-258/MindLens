from transformers import DistilBertTokenizerFast

def align_spans_to_bio(
    text: str,
    spans: list,  
    tokenizer: DistilBertTokenizerFast,
    label2id: dict
):
    """
    Convert character-level spans to token-level BIO labels.
    """

    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=True
    )

    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    offsets = encoding["offset_mapping"]

    labels = ["O"] * len(tokens)

    for span_start, span_end, span_label in spans:
        first_token = True

        for i, (tok_start, tok_end) in enumerate(offsets):

            # Rule 1: skip special tokens
            if tok_start == tok_end == 0:
                continue

            # Rule 2: token fully inside span
            if tok_start >= span_start and tok_end <= span_end:
                if first_token:
                    labels[i] = f"B-{span_label}"
                    first_token = False
                else:
                    labels[i] = f"I-{span_label}"

    label_ids = [label2id[label] for label in labels]

    return {
        "tokens": tokens,
        "labels": labels,
        "label_ids": label_ids
    }
