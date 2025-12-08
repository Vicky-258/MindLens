# %% Load cleaned dataset
import json
from pathlib import Path

TRAIN = Path("../data/processed/train.jsonl")

def load_jsonl(path):
    return [json.loads(line) for line in path.open()]

data = load_jsonl(TRAIN)
len(data)
# %% Build X (text) and y (binary labels)
X = [d["text"] for d in data]
y = [1 if len(d["technique_classification"]) > 0 else 0 for d in data]

sum(y), len(y)
# %% Train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# %% TF-IDF vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),     # unigrams + bigrams
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
# %% Train logistic regression
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=2000)
clf.fit(X_train_vec, y_train)
# %% Predictions & metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_pred = clf.predict(X_test_vec)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

acc, prec, rec, f1
# %% Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
# %% Pretty confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Propaganda","Propaganda"],
            yticklabels=["No Propaganda","Propaganda"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
# %% Save confusion matrix
import numpy as np
import json

cm_path = Path("../data/processed/confusion_matrix.json")
cm_path.write_text(json.dumps(cm.tolist()))
