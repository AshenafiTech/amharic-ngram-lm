# NLP Assignment 1 — N-gram Language Models for Amharic (GPAC)

N-gram language models (n = 1–4) built on the **GPAC** Amharic corpus
(Gereme et al., *Information* 2021, [doi:10.3390/info12010020](https://doi.org/10.3390/info12010020)),
with intrinsic and extrinsic evaluation and an interactive next-word suggestion app.

| Deliverable | Where |
|---|---|
| Tasks 1–4, with all code, outputs, figures and discussion | [`Amharic_Ngram_LM.ipynb`](Amharic_Ngram_LM.ipynb) |
| Task 4 web app (live suggestions as you type) | `python app/server.py` → http://localhost:8000 |
| Task 4 terminal app | `python app/cli.py` |
| Figures (PNG) | `outputs/figures/` |
| Reusable code | `amharic_lm/` |
| Full documentation (setup, methods, results, code reference) | [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md) |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/build_artifacts.py      # downloads GPAC (1.06 GB), encodes it, trains all models (~2 min)
jupyter notebook Amharic_Ngram_LM.ipynb
python app/server.py                   # Task 4 web app on http://localhost:8000
```

`build_artifacts.py` is optional: the notebook and the app build anything that is missing on first use.
Peak memory is about 6 GB, and the cached artifacts take about 6 GB of disk (`artifacts/`, git-ignored and
regenerable). On Google Colab, upload the folder (without `artifacts/` and `data/`) and run the notebook.

## Project layout

```
amharic_lm/
  preprocess.py   sentence splitting, tokenisation, homophone + number normalisation
  corpus.py       corpus -> int32 id arrays, vocabulary, de-duplication, block train/dev/test split
  ngram.py        n-gram tables as sorted numpy arrays; MLE / Laplace / add-k / interpolated Kneser-Ney;
                  sentence scoring, perplexity, generation, next-word candidates
  stopwords.py    curated Amharic stop-word list
  analysis.py     Task 1 helpers (statistics, top n-grams, sentence reports, generation analysis)
  evaluation.py   Task 2 perplexity; Task 3 next-word prediction, keystroke savings,
                  spelling correction, word-order test
  predictor.py    Task 4 suggestion engine (shared by the web app, the CLI and the notebook)
  viz.py          plotting style and Ethiopic word clouds
  pipeline.py     cached build steps; paths.py
app/server.py, app/static/index.html   web app (Python standard library + numpy only)
app/cli.py                             terminal app
scripts/build_artifacts.py             builds every cached artifact
fonts/NotoSansEthiopic.ttf             Ethiopic font for word clouds and the app (SIL OFL)
```

## Results at a glance

| n | KN test perplexity | Laplace perplexity | Next-word acc@5 | Keystroke savings | Spelling correction | Word-order (swap) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 10,609 | 10,642 | 13.6% | 36.5% | 81.5% | 50.0% |
| 2 | 1,264 | 41,202 | 26.5% | 46.4% | 90.3% | 91.0% |
| 3 | 789 | 263,659 | 33.1% | 48.4% | 90.5% | 91.4% |
| 4 | **704.5** | 411,381 | **34.6%** | **48.8%** | **90.7%** | 91.3% |

MLE perplexity is infinite for n ≥ 2: 24% / 58% / 72% of test tokens are unseen n-grams. Corpus after
cleaning: 61.9M tokens, 4.09M sentences, 1.81M types (52% hapax). The app uses the trigram context by default:
from n = 3 to n = 4, perplexity improves by 11% but the extrinsic gains are under 1.5 points.

## Corpus quirks found (worth mentioning in the presentation)

* The dump deleted `.` `,` `-`, so `ዓ.ም` became `ዓ ም` and `8.4` became `8 4`. These are repaired before tokenisation.
* The letter **ኃ** never occurs: ኃላፊነት appears as ላፊነት, and ኃይለማርያም as ይለማርያም.
* 24% of sentences repeat an earlier one (re-published articles, boilerplate such as ESAT datelines).

## Design notes

* **Scale.** The clean corpus has 61.9M tokens and 4.1M sentences, which give 52M distinct 4-grams. A Python
  dict of tuples would not fit in memory, so each order is stored as sorted numpy arrays. A context of up to
  three words is packed into one int64 (21 bits per word, vocabulary < 2²¹), and every lookup is a vectorised
  binary search. Building orders 1–4 takes about 40 s; scoring all 3.3M test events takes about 1 s.
* **Preprocessing.** Sentence split on `። ፡፡ ? !`. Tokens are Ethiopic letter/digit runs, with `/` allowed
  inside abbreviations. Homophone letters are merged (ሐ/ኀ→ሀ, ሠ→ሰ, ዐ→አ, ፀ→ጸ, ሃ→ሀ, ኣ/ዓ→አ, ዉ→ው), digit runs
  become `<NUM>`, and repeated sentences with 5+ tokens are removed (19% of tokens were duplicates). For display,
  each word keeps its most frequent original spelling.
* **Smoothing.** Interpolated Kneser-Ney uses one discount per order, D = n₁/(n₁+2n₂), and continuation counts
  for lower orders, except for n-grams starting with `<s>` (the KenLM convention). Unit tests confirmed that
  the distributions sum to 1 for every order, both for seen and unseen contexts.
* **Evaluation split.** 90/5/5 split by blocks of 200 consecutive sentences, which approximates a
  document-level split. Words seen once in training become `<UNK>`.
