"""Data check for FMA-small genre classification.

Prints examples per genre and split and a few checks on the data, saves them to
results/data_check.json, and writes samples.md with three training tracks per genre.
Uses the metadata only, so it cannot see corrupted audio: that check waits for the clips.

Run:
    python data_check.py
"""
import ast

import pandas as pd

from mustcite import save_results

META = "fma_metadata"
SEED = 0
SPLITS = ["training", "validation", "test"]
CLIP_SECONDS = 30  # FMA-small clips are 30 s, so a shorter full track cannot give a full clip

tracks = pd.read_csv(f"{META}/tracks.csv", index_col=0, header=[0, 1])
features = pd.read_csv(f"{META}/features.csv", index_col=0, header=[0, 1, 2])
genres = pd.read_csv(f"{META}/genres.csv", index_col=0)

small = tracks[tracks[("set", "subset")] == "small"]
genre = small[("track", "genre_top")]
split = small[("set", "split")]

# --- examples per genre and split ---
table = pd.crosstab(genre, split)[SPLITS]
print(table, "\n")
print("Every genre in every split:", bool((table > 0).all(axis=None)))

# --- do artists or albums cross splits? (related tracks should stay together) ---
artist = small[("artist", "id")]
album = small[("album", "id")]
album = album[album >= 0]  # -1 means no album


def shared(ids):
    sets = {s: set(ids[split.reindex(ids.index) == s]) for s in SPLITS}
    return {
        "train_val": len(sets["training"] & sets["validation"]),
        "train_test": len(sets["training"] & sets["test"]),
        "val_test": len(sets["validation"] & sets["test"]),
    }


artists_shared = shared(artist)
albums_shared = shared(album)
print("Artists shared between splits:", artists_shared)
print("Albums shared between splits:", albums_shared)

# --- are features and labels complete? ---
missing_features = small.index.difference(features.index)
X = features.reindex(small.index)["mfcc"]
short = small[small[("track", "duration")] < CLIP_SECONDS]
print("Tracks without features:", len(missing_features))
print("Tracks with a missing MFCC value:", int(X.isna().any(axis=1).sum()))
print("Tracks without a genre label:", int(genre.isna().sum()))
print(f"Tracks shorter than {CLIP_SECONDS} s in the metadata:", len(short))
print(short[[("track", "title"), ("track", "duration"), ("set", "split")]], "\n")

save_results(
    "data_check",
    {
        "examples_per_genre_and_split": {g: row.to_dict() for g, row in table.iterrows()},
        "total_per_split": table.sum().to_dict(),
        "artists_per_split": {s: int(artist[split == s].nunique()) for s in SPLITS},
        "artists_shared_between_splits": artists_shared,
        "albums_shared_between_splits": albums_shared,
        "tracks_without_features": len(missing_features),
        "tracks_with_missing_mfcc": int(X.isna().any(axis=1).sum()),
        "tracks_without_genre": int(genre.isna().sum()),
        "tracks_shorter_than_clip": {str(i): int(d) for i, d in short[("track", "duration")].items()},
    },
    {"subset": "small", "clip_seconds": CLIP_SECONDS},
)


# --- samples.md: three training tracks per genre, so that reading them does not touch test ---
def clean(value):
    return str(value).replace("|", "/").replace("\n", " ")


def genre_names(value):
    return ", ".join(genres.loc[g, "title"] for g in ast.literal_eval(value))


lines = ["# Samples", "", "Three random training tracks per genre, written by `data_check.py`.", ""]
for name in table.index:
    lines += [f"## {name}", "", "| Track | Title | Artist | Album | All genre tags |", "|---:|---|---|---|---|"]
    picked = small[(genre == name) & (split == "training")].sample(3, random_state=SEED)
    for track_id, row in picked.iterrows():
        cells = [
            track_id,
            clean(row[("track", "title")]),
            clean(row[("artist", "name")]),
            clean(row[("album", "title")]),
            clean(genre_names(row[("track", "genres")])),
        ]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    lines.append("")

with open("samples.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("Wrote samples.md")
