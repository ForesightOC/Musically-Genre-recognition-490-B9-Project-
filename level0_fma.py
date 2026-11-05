"""Level 0 baseline for FMA-small genre classification.

Logistic regression on the precomputed MFCC summary statistics in features.csv.
Picks C on the validation split, then evaluates the chosen model once on test.

Needs only the metadata (no audio):
    https://os.unil.cloud.switch.ch/fma/fma_metadata.zip  (~342 MiB)
Unzip it so that fma_metadata/tracks.csv and fma_metadata/features.csv exist
next to this script.

Run:
    pip install numpy pandas scikit-learn
    python level0_fma.py
Writes results/level0_fma_metrics.json and results/level0_fma_confusion.csv.
"""
import json
import os

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

META = "fma_metadata"
OUT = "results"
SEED = 0

# --- load metadata (both files have multi-row headers) ---
tracks = pd.read_csv(f"{META}/tracks.csv", index_col=0, header=[0, 1])
features = pd.read_csv(f"{META}/features.csv", index_col=0, header=[0, 1, 2])

small = tracks[tracks[("set", "subset")] == "small"]
genre = small[("track", "genre_top")]
split = small[("set", "split")]          # "training", "validation", "test"
artist = small[("artist", "name")]
X_all = features.loc[small.index, "mfcc"]  # 20 MFCCs x 7 statistics = 140 features

print(f"FMA-small: {len(small)} tracks, {X_all.shape[1]} MFCC features")
print(pd.crosstab(genre, split), "\n")

# --- check: does any artist appear in more than one split? ---
sets = {s: set(artist[split == s]) for s in ["training", "validation", "test"]}
overlap = {
    "train_val": len(sets["training"] & sets["validation"]),
    "train_test": len(sets["training"] & sets["test"]),
    "val_test": len(sets["validation"] & sets["test"]),
}
print("Artists shared between splits:", overlap, "\n")


def part(name):
    mask = split == name
    return X_all[mask], genre[mask]


X_tr, y_tr = part("training")
X_va, y_va = part("validation")
X_te, y_te = part("test")

# --- choose C on validation only ---
best_c, best_acc = None, -1.0
for c in [0.001, 0.01, 0.1, 1.0, 10.0]:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, max_iter=5000, random_state=SEED),
    )
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_va, model.predict(X_va))
    print(f"C={c:<6} validation accuracy {acc:.4f}")
    if acc > best_acc:
        best_c, best_acc = c, acc
print(f"\nChosen C = {best_c}\n")

# --- final model: trained on training only, evaluated on val and test once ---
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=best_c, max_iter=5000, random_state=SEED),
)
model.fit(X_tr, y_tr)
pred_va, pred_te = model.predict(X_va), model.predict(X_te)

labels = sorted(genre.unique())
metrics = {
    "task": "FMA-small genre classification",
    "model": "Level 0: StandardScaler + LogisticRegression on MFCC summary features",
    "C": best_c,
    "n_train": int(len(y_tr)),
    "n_val": int(len(y_va)),
    "n_test": int(len(y_te)),
    "artists_shared_between_splits": overlap,
    "val_accuracy": round(accuracy_score(y_va, pred_va), 4),
    "val_macro_f1": round(f1_score(y_va, pred_va, average="macro"), 4),
    "test_accuracy": round(accuracy_score(y_te, pred_te), 4),
    "test_macro_f1": round(f1_score(y_te, pred_te, average="macro"), 4),
    "chance_accuracy": round(1 / len(labels), 4),
}

os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/level0_fma_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
cm = pd.DataFrame(
    confusion_matrix(y_te, pred_te, labels=labels),
    index=[f"true_{l}" for l in labels],
    columns=[f"pred_{l}" for l in labels],
)
cm.to_csv(f"{OUT}/level0_fma_confusion.csv")

print(json.dumps(metrics, indent=2))
print("\nTest confusion matrix:\n", cm)
