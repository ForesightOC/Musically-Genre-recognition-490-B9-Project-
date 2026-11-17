"""Level 3: which artists make up each genre in each split?

The professor pointed out that Folk is 79/100 correct on validation but 18/100 on test
with Level 0. This script looks only at the metadata (no model, no predictions): for every
genre and split it counts the artists and how many tracks the biggest artists contribute.
If a split's Folk is a few artists with an unusual sound, that would explain the drop.

Saves results/level3_artists.json.

Run:
    python level3_artists.py
"""
import pandas as pd

from mustcite import save_results

META = "fma_metadata"
SPLITS = ["training", "validation", "test"]
TOP = 5  # how many of the biggest artists to list per genre and split
FOCUS = "Folk"  # genre to print in detail

tracks = pd.read_csv(f"{META}/tracks.csv", index_col=0, header=[0, 1])
small = tracks[tracks[("set", "subset")] == "small"]

df = pd.DataFrame({
    "genre": small[("track", "genre_top")],
    "split": small[("set", "split")],
    "artist": small[("artist", "name")],
})

summary = {}
for (g, s), group in df.groupby(["genre", "split"]):
    counts = group["artist"].value_counts()  # tracks per artist, largest first
    summary.setdefault(g, {})[s] = {
        "tracks": len(group),
        "artists": int(counts.size),
        # share of the genre's tracks from its single biggest artist, and from the top 5
        "top1_share": float(counts.iloc[0] / len(group)),
        "top5_share": float(counts.iloc[:TOP].sum() / len(group)),
        "top_artists": {name: int(n) for name, n in counts.iloc[:TOP].items()},
    }

# compact table: number of artists and top-5 share, per genre and split
table = pd.DataFrame({
    (s, col): {g: summary[g][s][col] for g in summary}
    for s in SPLITS for col in ["artists", "top5_share"]
})
print(table.round(2), "\n")

for s in SPLITS:
    print(f"{FOCUS}, {s}: {summary[FOCUS][s]['top_artists']}")

save_results(
    "level3_artists",
    {"per_genre_and_split": summary},
    {"subset": "small", "top": TOP, "source": "tracks.csv metadata only, no predictions"},
)
