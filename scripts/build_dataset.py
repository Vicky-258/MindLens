import json
from pathlib import Path
from Span2Bio import spans_to_bio
from pathlib import Path
from collections import defaultdict
from transformers import DistilBertTokenizerFast

with open("configs/labels.json", "r") as f:
    label2id = json.load(f)

tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

def load_task2_spans(label_path):
    """
    Returns:
    {
      article_id: [(start, end, label), ...]
    }
    """
    spans_by_article = defaultdict(list)

    with open(label_path, encoding="utf-8") as f:
        for line in f:
            article_id, label, start, end = line.strip().split("\t")
            spans_by_article[article_id].append(
                (int(start), int(end), label)
            )

    return spans_by_article

def load_article_text(article_id, articles_dir):
    path = Path(articles_dir) / f"article{article_id}.txt"
    with open(path, encoding="utf-8") as f:
        return f.read()

def build_span_ner_dataset(
    articles_dir,
    labels_path,
    tokenizer,
    label2id,
    output_path
):
    spans_by_article = load_task2_spans(labels_path)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_samples = 0

    with output_path.open("w", encoding="utf-8") as f:
        for article_id, spans in spans_by_article.items():

            text = load_article_text(article_id, articles_dir)

            result = spans_to_bio(
                text=text,
                spans=spans,
                tokenizer=tokenizer,
                label2id=label2id
            )

            input_ids = result["input_ids"]
            label_ids = result["label_ids"]

            # --- Sanity checks ---
            assert len(input_ids) == len(label_ids), \
                f"Length mismatch in article {article_id}"

            assert all(
                0 <= lid < len(label2id) for lid in label_ids
            ), f"Invalid label id in article {article_id}"

            record = {
                "article_id": article_id,
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": label_ids
            }

            f.write(json.dumps(record) + "\n")
            num_samples += 1

    print(f"✅ Saved {num_samples} samples to {output_path}")

build_span_ner_dataset(
    articles_dir="tmp_semeval/datasets/train-articles",
    labels_path="tmp_semeval/datasets/train-task2-TC.labels",
    tokenizer=tokenizer,
    label2id=label2id,
    output_path="data/processed/span_ner.jsonl"
)
