# Music Genre Recognition on FMA-small

**Author:** Osabo Christopher · CSCI 490-B9 Deep Learning, NIU, Fall 2026

**AI tool use:** Claude Code (Claude Opus 5.5) set up Git and GitHub, changed how `level0_fma.py` saves its results, wrote `data_check.py`, adapted `AGENTS.md`, and filled in the README's data and Level 0 tables from the files in `results/`. <add who wrote the first version of `level0_fma.py`, and any other tools you used and for what>

## 1. Task

The input is a 30-second music clip, and the output is one of 8 top-level genres: Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop or Rock. The Level 0 baseline does not read the audio directly: its input is the 140 MFCC summary statistics that FMA provides for each track.

**Example:** track 112528, "Fix yourself" by Action Will Be Taken. Label: Rock.

## 2. Motivation

<Who would use this, and what makes it hard? Three to five sentences.>

## 3. Data

The data is FMA-small from the Free Music Archive dataset (Defferrard et al., ISMIR 2017, https://github.com/mdeff/fma): 8,000 clips of 30 seconds, balanced across 8 genres, about 7.2 GB of audio. The dataset's metadata is licensed CC BY 4.0, and each audio track keeps its own license from its artist.

I use the official split in `tracks.csv`, so results can be compared directly with the published baselines. No artist appears in more than one split, which `data_check.py` confirms by artist id, so a model cannot score well by recognizing an artist it has already heard.

Output of `data_check.py` (saved in `results/data_check.json`):

| Class | Train | Validation | Test |
|---|---:|---:|---:|
| Electronic | 800 | 100 | 100 |
| Experimental | 800 | 100 | 100 |
| Folk | 800 | 100 | 100 |
| Hip-Hop | 800 | 100 | 100 |
| Instrumental | 800 | 100 | 100 |
| International | 800 | 100 | 100 |
| Pop | 800 | 100 | 100 |
| Rock | 800 | 100 | 100 |
| **Total** | 6,400 | 800 | 800 |

Other checks from the same file: no track is missing its features, MFCC values or genre label, and no track is shorter than 30 seconds in the metadata. Albums do cross splits (58 train/validation, 49 train/test, 21 validation/test). Because no artist crosses splits, these must be albums with several artists, such as compilations. The metadata cannot show corrupted audio, so I will check the clips themselves when I build the log-mel spectrograms.

Three examples of each class are in [samples.md](samples.md). <What did you notice when you read them? Write this after reading the file yourself.>

## 4. Method

**Level 0 result** (from `results/level0_fma_val.json` and `results/level0_fma_test.json`; guessing gives 12.5%):

| Model | Validation accuracy | Validation macro-F1 | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|
| Level 0: logistic regression on MFCC summary features, C = 0.01 | 0.5025 | 0.4911 | 0.4163 | 0.4113 |

**Plan for the next levels.** For each, the number you expect:

| Level | Model | Expected accuracy on validation |
|---|---|---:|
| 1 | <a neural model you train> | <number> |
| 2 | <a pretrained Transformer, fine-tuned> | <number> |
| 3 | The question below | – |

**Level 3 question:** <your question about the gap between validation and test accuracy, and the evidence that would answer it>

## 5. Evaluation

I report accuracy, macro-F1 and the confusion matrix. Models, hyperparameters and epochs are chosen on the validation split only, and each final model is evaluated once on the test split.

<Say why these metrics fit this task, and what counts as success.>

## 6. Plan

| Week | What I will do |
|---|---|
| 11/2 – 11/8 | <...> |
| 11/9 – 11/15 | <...> |
| 11/16 – 11/22 | <...> (progress report 11/19: Level 1) |
| 11/23 – 11/29 | <...> |
| 11/30 – 12/3 | <...> (presentation: Level 2) |

## Results

| Level | Model | Validation accuracy | Test accuracy | Results file |
|---|---|---:|---:|---|
| 0 | Logistic regression on MFCC summary features | 0.5025 | 0.4163 | `results/level0_fma_val.json`, `results/level0_fma_test.json` |

## How to run

Download https://os.unil.cloud.switch.ch/fma/fma_metadata.zip (about 342 MiB) and unzip it next to the scripts, so that `fma_metadata/tracks.csv` exists. Then:

```
pip install -r requirements.txt
python level0_fma.py
python data_check.py
```

`mustcite.py` and the `data/` folder come from the course starter repository. This project only uses `save_results` from `mustcite.py`, and `level0_task_*.py` are the starter's unused examples.

## Data source

Defferrard, Benzi, Vandergheynst and Bresson, "FMA: A Dataset for Music Analysis", ISMIR 2017. https://github.com/mdeff/fma
