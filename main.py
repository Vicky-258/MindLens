import torch
import json

ckpt = torch.load("configs/class_weights.pt")

class_weights = ckpt["class_weights"]
label_counts = ckpt["label_counts"]
alpha = ckpt["alpha"]

print("alpha:", alpha)
print("num labels:", len(class_weights))
print("mean weight:", class_weights.mean().item())
print("min weight:", class_weights.min().item())
print("max weight:", class_weights.max().item())

with open("configs/labels.json") as f:
    label2id = json.load(f)

id2label = {v: k for k, v in label2id.items()}

pairs = []
for i, w in enumerate(class_weights):
    pairs.append((id2label[i], label_counts.get(i, 0), w.item()))

# sort by weight (descending)
pairs.sort(key=lambda x: x[2], reverse=True)

print("\n🔥 Top 10 highest-weight labels:")
for lbl, cnt, w in pairs[:10]:
    print(f"{lbl:40s} count={cnt:6d} weight={w:.3f}")

print("\n🧊 Lowest-weight labels:")
for lbl, cnt, w in pairs[-5:]:
    print(f"{lbl:40s} count={cnt:6d} weight={w:.3f}")
