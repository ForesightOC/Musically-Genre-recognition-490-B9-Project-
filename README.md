# Osabo Christopher – CSCI 490 Course Project: Music Genre Classification (FMA-small)

CSCI 490-B9 Deep Learning, NIU, Fall 2026

**AI tool use:** I used Claude (Anthropic) to choose the dataset, to write the first version of `level0_fma.py`, and to draft parts of this README, which I edited. Claude Code (Claude Opus 5.5) in VS Code set up Git and GitHub, changed how `level0_fma.py` saves its results, wrote `data_check.py`, adapted `AGENTS.md`, filled in the README's data and Level 0 tables from the files in `results/`, and ran `level0_fma.py` and `data_check.py` to produce the committed results. Every number here comes from a file in `results/`. I can explain every line of the code.

## 1. Task

The input is a 30-second music clip, and the output is one of 8 top-level genres: Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop or Rock. The Level 0 baseline does not read the audio directly: its input is the 140 MFCC summary statistics that FMA provides for each track.

**Example:** track 112528, "Fix yourself" by Action Will Be Taken. Label: Rock.

## 2. Motivation

Music platforms that accept uploads, such as the Free Music Archive, rely on artists to tag their own tracks, and those tags are often missing or inconsistent. A model that predicts genre from the audio could suggest or check tags at upload time, and help recommend new tracks that have no listening history yet. The task is hard because genre is defined by people, not by the sound alone: Experimental, International and Pop are broad categories that overlap with the others. The model also has to work on artists it has never heard, which is why my split keeps each artist in one set. My Level 0 baseline shows the difficulty: 50.3% on validation but 41.6% on test, against 12.5% for guessing.

## 3. Data

The data is FMA-small from the Free Music Archive dataset (Defferrard et al., ISMIR 2017, https://github.com/mdeff/fma): 8,000 clips of 30 seconds, balanced across 8 genres, 7.2 GiB of audio. The dataset's metadata is licensed CC BY 4.0, and each audio track keeps its own license from its artist.

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
| 1 | CNN trained on log-mel spectrograms | 52–55% |
| 2 | Audio Spectrogram Transformer (AST), fine-tuned | 60–65% |
| 3 | The question below | – |

**Level 1:** a CNN I train myself on log-mel spectrograms (128 mel bins, 10-second crops, computed once with torchaudio and cached). I expect about 52–55% validation accuracy: a little better than Level 0, because the CNN sees how the sound changes over time instead of 140 summary numbers, but limited by the small training set (6,400 clips).

**Level 2:** the Audio Spectrogram Transformer (AST), `MIT/ast-finetuned-audioset-10-10-0.4593` from Hugging Face, pretrained on AudioSet and fine-tuned on my training split. I expect about 60–65% validation accuracy, because AudioSet pretraining already includes music and instruments. Published FMA-small results go much higher, but many use different splits, so I don't expect to match them.

**Level 3 question:** Why is test accuracy about 9 points below validation (50.3% vs 41.6%), and does the gap shrink with a better model?

Evidence that would answer it:

- **Is it noise?** Bootstrap the 800 validation and 800 test predictions (resample 1,000 times) to get a confidence interval for the gap. If the interval includes about 0, the gap is mostly chance.
- **Which genres drop?** Compare per-genre accuracy and the confusion matrices on validation and test. If the drop is concentrated in Experimental, International and Pop, the gap comes from broad labels, not from the model.
- **Are the splits different?** Compare validation and test metadata in `tracks.csv`: number of artists, tracks per artist, upload date. For example, a test set dominated by a few unusual artists would explain a drop.
- **Does it persist?** Measure the same gap for Level 1 and Level 2. If it stays about the same, it's in the data; if it shrinks, the baseline features were the limit.

## 5. Evaluation

The classes are balanced (100 clips per genre in validation and test), so accuracy is meaningful, and I compare it with chance at 12.5%. I also report macro-F1, which weights every genre equally and would reveal a model that ignores one genre, and the confusion matrix, because which genres get confused is the core of my Level 3 question. I choose all models, hyperparameters and epochs on validation and evaluate each final model once on test. Success: Level 2 beats Level 0 on validation by at least 10 points, and I can answer the Level 3 question with the evidence above, whichever way it comes out.


## 6. Plan

| Week | What I will do |
|---|---|
| 11/6 – 11/12 | Decode the audio once and cache 10-second log-mel spectrograms; write the PyTorch Dataset and training loop; get a first CNN run on Colab |
| 11/13 – 11/19 | Tune the CNN on validation (learning rate, dropout, epochs); evaluate once on test; progress report 11/19 with Level 1 in the results table |
| 11/20 – 11/26 | Fine-tune AST on Colab within the 30-minute limit; start the Level 3 analysis (bootstrap interval, per-genre accuracy for Levels 0 and 1) (Thanksgiving week, lighter) |
| 11/27 – 12/1 | Evaluate AST once on test; finish the Level 3 comparison across all three levels; make the slides; presentation 12/1 |
| 12/2 – 12/3 | Backup presentation date (12/3); start the IEEE final report |

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
