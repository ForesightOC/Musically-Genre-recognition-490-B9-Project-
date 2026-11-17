"""Decode every FMA-small clip once and cache its log-mel spectrogram.

Run on Colab from the repository folder, after downloading and unzipping fma_small.zip
and fma_metadata.zip to local disk (/content), with Google Drive mounted:

    python level1_cache_mels.py --audio /content/fma_small --meta /content/fma_metadata \
                                --out /content/drive/MyDrive/fma_cache

Writes to --out:
    mels.npy    float16 array, shape (n_ok, 128, 1292): one log-mel spectrogram per clip
    index.csv   one row per array row: row, track_id, genre, split
Writes results/cache_report.json (with save_results): how many clips decoded, how many
failed and why. This is the corrupted-audio check that the metadata could not do.

Clips that fail to decode, or are shorter than --min-sec seconds, are skipped and
listed in the report. Nothing here uses the validation or test labels for any choice.
"""
import argparse
import os
import shutil
import time
from multiprocessing import Pool

import librosa
import numpy as np
import pandas as pd

from mustcite import save_results

SR = 22050          # sample rate after resampling (Hz)
N_FFT = 2048        # window: about 93 ms
HOP = 512           # hop: about 23 ms, so 30 s -> 1292 frames
N_MELS = 128
CLIP_SEC = 30.0
T = 1 + int(CLIP_SEC * SR) // HOP   # 1292 frames for a full 30 s clip
EPS = 1e-6
PAD_VALUE = np.log(EPS)             # log of "silence", used to pad short clips

p = argparse.ArgumentParser()
p.add_argument("--audio", default="/content/fma_small")
p.add_argument("--meta", default="/content/fma_metadata")
p.add_argument("--out", default="/content/drive/MyDrive/fma_cache")
p.add_argument("--tmp", default="/content/mels_tmp.npy")
p.add_argument("--min-sec", type=float, default=10.0,
               help="clips shorter than this are skipped (Levels 1-2 use 10 s crops)")
args = p.parse_args()


def audio_path(track_id):
    tid = f"{track_id:06d}"
    return os.path.join(args.audio, tid[:3], tid + ".mp3")


def decode(track_id):
    """Return (track_id, log-mel array or None, failure reason or None)."""
    try:
        y, _ = librosa.load(audio_path(track_id), sr=SR, mono=True, duration=CLIP_SEC)
    except Exception as e:  # corrupted or unreadable mp3
        return track_id, None, f"decode error: {type(e).__name__}"
    seconds = len(y) / SR
    if seconds < args.min_sec:
        return track_id, None, f"too short: {seconds:.1f} s"
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    logmel = np.log(mel + EPS)
    if logmel.shape[1] < T:   # pad a slightly short clip at the end
        logmel = np.pad(logmel, ((0, 0), (0, T - logmel.shape[1])),
                        constant_values=PAD_VALUE)
    return track_id, logmel[:, :T].astype(np.float16), None


if __name__ == "__main__":
    start = time.time()

    tracks = pd.read_csv(f"{args.meta}/tracks.csv", index_col=0, header=[0, 1])
    small = tracks[tracks[("set", "subset")] == "small"].sort_index()
    ids = small.index.to_numpy()
    n = len(ids)
    print(f"{n} clips to decode; full array would be "
          f"{n * N_MELS * T * 2 / 1e9:.1f} GB as float16")

    # 1. Decode in parallel, writing into a temporary array on local disk.
    tmp = np.lib.format.open_memmap(args.tmp, mode="w+", dtype=np.float16,
                                    shape=(n, N_MELS, T))
    ok = np.zeros(n, dtype=bool)
    failures = []
    with Pool(os.cpu_count()) as pool:
        # imap keeps the input order, so result i belongs to ids[i]
        for i, (tid, logmel, reason) in enumerate(pool.imap(decode, ids, chunksize=8)):
            if logmel is None:
                failures.append({"track_id": int(tid), "reason": reason})
            else:
                tmp[i] = logmel
                ok[i] = True
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{n} done, {len(failures)} failed, "
                      f"{(time.time() - start) / 60:.1f} min")
    tmp.flush()

    # 2. Keep only the clips that decoded, and copy them to Drive.
    os.makedirs(args.out, exist_ok=True)
    final_local = "/content/mels.npy"
    final = np.lib.format.open_memmap(final_local, mode="w+", dtype=np.float16,
                                      shape=(int(ok.sum()), N_MELS, T))
    rows = np.flatnonzero(ok)
    for start_row in range(0, len(rows), 500):
        chunk = rows[start_row:start_row + 500]
        final[start_row:start_row + len(chunk)] = tmp[chunk]
    final.flush()
    del tmp, final
    os.remove(args.tmp)
    print("Copying to Drive...")
    shutil.copy(final_local, os.path.join(args.out, "mels.npy"))

    kept = small.loc[ids[ok]]
    index = pd.DataFrame({
        "row": np.arange(len(kept)),
        "track_id": kept.index,
        "genre": kept[("track", "genre_top")].to_numpy(),
        "split": kept[("set", "split")].to_numpy(),
    })
    index.to_csv(os.path.join(args.out, "index.csv"), index=False)

    # 3. Report: this is the corrupted-audio check for the README.
    failed_meta = small.loc[sorted(f["track_id"] for f in failures)]
    report = {
        "n_clips": int(n),
        "n_cached": int(ok.sum()),
        "n_failed": len(failures),
        "n_decode_errors": sum(f["reason"].startswith("decode") for f in failures),
        "n_too_short": sum(f["reason"].startswith("too short") for f in failures),
        "failed_per_split": failed_meta[("set", "split")].value_counts().to_dict(),
        "failed_per_genre": failed_meta[("track", "genre_top")].value_counts().to_dict(),
        "cached_per_split": index["split"].value_counts().to_dict(),
        "failures": failures,
        "minutes": round((time.time() - start) / 60, 1),
    }
    config = {"sr": SR, "n_fft": N_FFT, "hop": HOP, "n_mels": N_MELS, "frames": T,
              "min_sec": args.min_sec, "log": "ln(mel + 1e-6)", "dtype": "float16"}
    path = save_results("cache_report", report, config)
    shutil.copy(path, os.path.join(args.out, "cache_report.json"))  # keep a copy next to the cache
