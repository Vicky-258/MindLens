import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.optim import AdamW

def load_jsonl(path):
    return [json.loads(line) for line in path.open()]


class PropagandaDataset(Dataset):
    def __init__(self, encodings, labels):
        self.enc = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def main():
    print("📥 Loading dataset...")
    TRAIN = Path(__file__).resolve().parent.parent / "data/processed/train.jsonl"
    data = load_jsonl(TRAIN)

    X = [d["text"] for d in data]
    y = [1 if len(d["technique_classification"]) > 0 else 0 for d in data]

    print(f"Dataset loaded: {len(X)} samples, {sum(y)} positives")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Tokenizer
    print("🔤 Loading tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    print("🔧 Tokenizing data...")
    train_enc = tokenizer(X_train, truncation=True, padding=True, max_length=256)
    test_enc = tokenizer(X_test, truncation=True, padding=True, max_length=256)

    # Torch datasets
    train_ds = PropagandaDataset(train_enc, y_train)
    test_ds = PropagandaDataset(test_enc, y_test)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16)

    # Model
    print("🤖 Loading DistilBERT...")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optim = AdamW(model.parameters(), lr=5e-5)

    print("🚀 Starting training...")
    model.train()
    step = 0
    for batch in train_loader:
        step += 1
        optim.zero_grad()

        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss
        loss.backward()
        optim.step()

        if step % 20 == 0:
            print(f"  → Step {step}, Loss: {loss.item():.4f}")

    print("🎉 Training complete!")

    # Evaluation
    print("📊 Evaluating...")
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch_labels = batch["labels"].tolist()
            labels.extend(batch_labels)

            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    cm = confusion_matrix(labels, preds)

    print("\n📈 Results:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")

    # Save results
    out = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
    }

    save_path = Path(__file__).resolve().parent.parent / "data/processed/distilbert_results.json"
    save_path.write_text(json.dumps(out, indent=2))

    print(f"\n💾 Results saved to: {save_path}")


if __name__ == "__main__":
    main()
