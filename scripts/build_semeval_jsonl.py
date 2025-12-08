import json
from pathlib import Path

BASE = Path("tmp_semeval/datasets")
ARTICLES = BASE / "train-articles"
SI_DIR = BASE / "train-labels-task1-span-identification"
TC_DIR = BASE / "train-labels-task2-technique-classification"

OUT = Path("data/raw/train.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)


def load_article_text(article_id: str):
    path = ARTICLES / f"article{article_id}.txt"
    return path.read_text(encoding="utf-8")


def load_task1_labels(article_id: str):
    """Span Identification labels."""
    si_file = SI_DIR / f"article{article_id}.task1-SI.labels"
    spans = []
    if not si_file.exists():
        return spans  # no spans for this article

    for line in si_file.read_text().splitlines():
        parts = line.strip().split()
        _, start, end = parts  # article_id is first
        spans.append({
            "start": int(start),
            "end": int(end)
        })
    return spans


def load_task2_labels(article_id: str):
    """Technique Classification labels."""
    tc_file = TC_DIR / f"article{article_id}.task2-TC.labels"
    techniques = []
    if not tc_file.exists():
        return techniques

    for line in tc_file.read_text().splitlines():
        parts = line.strip().split()
        _, technique, start, end = parts
        techniques.append({
            "technique": technique,
            "start": int(start),
            "end": int(end)
        })
    return techniques


def get_all_article_ids():
    ids = []
    for f in ARTICLES.iterdir():
        if f.name.startswith("article") and f.suffix == ".txt":
            article_id = f.stem.replace("article", "")
            ids.append(article_id)
    return sorted(ids)


def build_jsonl():
    ids = get_all_article_ids()
    print(f"Found {len(ids)} train articles")

    with OUT.open("w", encoding="utf-8") as outfile:
        for aid in ids:
            text = load_article_text(aid)
            si = load_task1_labels(aid)
            tc = load_task2_labels(aid)

            record = {
                "id": aid,
                "text": text,
                "span_identification": si,
                "technique_classification": tc
            }

            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote: {OUT}")


if __name__ == "__main__":
    build_jsonl()
