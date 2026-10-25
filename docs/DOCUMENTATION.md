# Amharic N-gram Language Models — Project Documentation

NLP Assignment 1 (10%) · Addis Ababa University · Instructor: Fantahun B. (PhD)

This document explains how the project is organised, how to run every part of it, how each assignment
question is answered, and how the code works. For a one-page overview see [`README.md`](../README.md).

**Contents**

1. [Overview](#1-overview)
2. [Getting started](#2-getting-started)
3. [Working in VS Code](#3-working-in-vs-code)
4. [Project structure and data flow](#4-project-structure-and-data-flow)
5. [Notebook guide (assignment question → notebook section)](#5-notebook-guide)
6. [Methods](#6-methods)
7. [Results](#7-results)
8. [Task 4: the next-word suggestion app](#8-task-4-the-next-word-suggestion-app)
9. [Code reference](#9-code-reference)
10. [Cached artifacts](#10-cached-artifacts)
11. [Troubleshooting](#11-troubleshooting)
12. [Known limitations](#12-known-limitations)
13. [References](#13-references)

---

## 1. Overview

The project builds unigram, bigram, trigram and 4-gram language models on **GPAC**, a general-purpose Amharic
corpus of about 77 million raw tokens (Gereme et al., *Information* 12(1):20, 2021). It covers all four questions
in the assignment:

| Assignment question | Weight | Where it is answered |
|---|---:|---|
| **1. N-gram language model:** statistics, n-grams, probabilities, stop words, word clouds, sentence probability, generation | 6% | Notebook, Task 1 (sections 1.1–1.8) |
| **2. Intrinsic evaluation:** perplexity for n = 1–4 | 3% | Notebook, Task 2 |
| **3. Extrinsic evaluation:** any convenient task | 5% | Notebook, Task 3 (three tasks) |
| **4. Predictive text / next-word suggestion system** | 6% | `app/server.py` (web), `app/cli.py` (terminal), notebook Task 4 |

Everything is in plain Python with **numpy** for the counting and **pandas / matplotlib / wordcloud** for the
tables and figures. There is no NLTK or KenLM dependency: the n-gram counting, smoothing and generation are
implemented in the `amharic_lm/` package so that every step can be explained in the presentation.

---

## 2. Getting started

### Requirements

* Python 3.10 or newer (developed on 3.12)
* About **6 GB of free RAM** at peak (building the 4-gram tables)
* About **7 GB of free disk**: 1.06 GB for the corpus and 5.9 GB for cached models

### Setup

```bash
cd NLP
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The `.venv` folder in this copy of the project already has every package installed.

### Build the cached data (optional)

```bash
python scripts/build_artifacts.py
```

This downloads GPAC from Google Drive (file id `1jEOAORylTBAiNKes_JerD6gFF7cnLFyr`) into `data/GPAC.txt`,
then encodes, cleans and splits it and trains the two models. It takes about 2 minutes after the download.
The step is optional: the notebook and the apps build anything that is missing the first time they need it.

### Run

| What | Command |
|---|---|
| Notebook (all four tasks) | `jupyter notebook Amharic_Ngram_LM.ipynb`, or open it in VS Code |
| Web app (Task 4) | `python app/server.py`, then open <http://localhost:8000> |
| Terminal app (Task 4) | `python app/cli.py` |

A full notebook run takes about 6 minutes on a laptop when the artifacts are already cached. Results are
deterministic: every random step uses a fixed seed, so a re-run reproduces the numbers in this document exactly.

---

## 3. Working in VS Code

The project includes a `.vscode/` folder, so it is ready to use as soon as you open the folder:

| File | What it does |
|---|---|
| `.vscode/settings.json` | Selects `.venv/bin/python` as the interpreter and Jupyter kernel. Hides `artifacts/`, `data/` and `.venv/` from search and the file watcher (they are several GB and would slow VS Code down). |
| `.vscode/launch.json` | Run configurations: **Web app (Task 4)**, **Terminal app (Task 4)**, **Build artifacts (download + train)**. |
| `.vscode/extensions.json` | Recommends the Python and Jupyter extensions. VS Code offers to install them on first open. |

**Common actions**

* **Run the notebook:** open `Amharic_Ngram_LM.ipynb`. If VS Code asks for a kernel, choose
  *Python Environments → .venv*. Use *Run All* in the toolbar, or run cells one by one with Shift+Enter.
  The notebook is saved with all outputs, so you can read it without running anything.
* **Start the web app:** open *Run and Debug* (⇧⌘D / Ctrl+Shift+D), pick **Web app (Task 4)** and press F5.
  When the terminal prints `serving on http://127.0.0.1:8000`, open that address. Stop it with ⇧F5.
* **Read this document as a formatted page:** with the file open, press ⇧⌘V (Ctrl+Shift+V) for a preview
  tab, or ⌘K V (Ctrl+K V) for a side-by-side preview. Formulas render in the preview.
* **Jump to code:** every module has a docstring at the top that explains its job; hover a function name to
  see its documentation, and ⌘-click (Ctrl-click) to go to its definition.

---

## 4. Project structure and data flow

```
NLP/
├── Amharic_Ngram_LM.ipynb      the main deliverable: Tasks 1–4 with outputs and discussion
├── README.md                   one-page overview
├── docs/DOCUMENTATION.md       this file
├── requirements.txt
├── amharic_lm/                 the Python package used by the notebook and both apps
│   ├── preprocess.py           repair, sentence splitting, tokenisation, normalisation
│   ├── corpus.py               text → integer arrays, vocabulary, de-duplication, train/dev/test split
│   ├── ngram.py                n-gram tables and the language model (MLE, Laplace, add-k, Kneser-Ney)
│   ├── stopwords.py            curated Amharic stop-word list
│   ├── analysis.py             Task 1 helpers (statistics, top n-grams, sentence reports, generation)
│   ├── evaluation.py           Task 2 perplexity and Task 3 extrinsic tasks
│   ├── predictor.py            Task 4 suggestion engine
│   ├── viz.py                  chart style and Ethiopic word clouds
│   ├── pipeline.py             cached build steps (download → encode → clean → split → train)
│   └── paths.py                every file location in one place
├── app/
│   ├── server.py               Task 4 web server (Python standard library only)
│   ├── static/index.html       Task 4 web interface
│   └── cli.py                  Task 4 terminal interface
├── scripts/build_artifacts.py  builds all cached data in one go
├── fonts/NotoSansEthiopic.ttf  Ethiopic font for word clouds, charts and the app (SIL Open Font License)
├── outputs/figures/            every chart as PNG (ready for slides)
├── data/                       GPAC.txt (downloaded, git-ignored)
└── artifacts/                  cached corpora and models (generated, git-ignored)
```

**Data flow**

```
data/GPAC.txt  (1.06 GB, one line)
   │  pipeline.raw_corpus()      repair → tokenise → normalise → integer ids
   ▼
artifacts/corpus_raw/            76.7M tokens, 5.09M sentences
   │  pipeline.clean_corpus()    remove repeated sentences (≥ 5 tokens)
   ▼
artifacts/corpus_clean/          61.9M tokens, 4.09M sentences, 1.81M word types
   ├── pipeline.full_model()     orders 1–4 on all clean data          → Task 1, Task 4 app
   └── pipeline.splits()         90 / 5 / 5 split in blocks of 200 sentences
          └── pipeline.train_model()   orders 1–4 on the 90% split,
                                       words seen once → <UNK>         → Task 2, Task 3
```

Two models are used on purpose. Task 1 and the app want the most data possible, so the **full model** uses
the whole clean corpus. Tasks 2 and 3 must be evaluated on text the model has never seen, so the **train
model** is trained on 90% only and tested on a held-out 5% (the other 5% is a development set for tuning).

---

## 5. Notebook guide

`Amharic_Ngram_LM.ipynb` follows the assignment in order. Each section starts with a short explanation, then
code and outputs, and ends with a discussion of the results.

| Section | Assignment asks | What the notebook shows | Figure |
|---|---|---|---|
| **1.1** Corpus statistics and preprocessing | Statistics, preprocessing | Raw character inventory; the preprocessing steps with before/after examples; corpus size, vocabulary, hapax rate, Zipf and Heaps curves, top words | `1_1_corpus_statistics.png` |
| **1.2** Creating n-grams | n-grams for n = 1–4, sample prints | Plain-Python n-grams of one sentence with `<s>`/`</s>` padding; counts of distinct n-grams and singletons per order | `1_2_ngram_counts.png` |
| **1.3** N-gram probabilities | Probabilities, top 10 for each n | Joint probability P(w₁…wₙ) and conditional probability P(wₙ \| w₁…wₙ₋₁) of the 10 most frequent n-grams for each n | — |
| **1.4** Conditional probability | P(word \| previous word) | P(word \| previous word) for chosen pairs (e.g. አዲስ → አበባ, ጠቅላይ → ሚኒስትር) and the full next-word distribution after several words | — |
| **1.5** Stop-word removal | Remove stop words, recompute, top 10 | The stop-word list, its coverage (18.6% of tokens), and the top 10 n-grams for each n after removal | — |
| **1.6** Word clouds | Clouds before and after stop-word removal | Eight word clouds (n = 1–4, before and after) | `1_6_wordclouds.png` |
| **1.7** Sentence probability | P("ኢትዮጵያ በአፍሪካ ቀንድ ትገኛለች።") and more | Step-by-step probabilities for the assignment sentence and others, with MLE and Kneser-Ney, for each n | — |
| **1.8** Random generation | Generate sentences, explain how n matters | Sentences sampled from each n; fluency, length and how many are copied verbatim from the corpus | — |
| **Task 2** Intrinsic evaluation | Perplexity for n = 1–4 | Perplexity on held-out text with four estimators; coverage of unseen n-grams; add-k tuning | `2_perplexity.png` |
| **Task 3** Extrinsic evaluation | Any convenient task | 3A next-word prediction and keystroke savings; 3B spelling correction; 3C word-order judgement; intrinsic vs extrinsic | `3_extrinsic.png` |
| **Task 4** Suggestion system | Read input → last 1–2 words → candidates → probabilities → top 5 → choose | The suggestion engine on example inputs with an explanation of every column; how to launch the apps | `app_screenshot.png` |
| **Summary** | — | Key results and what we learned | — |

---

## 6. Methods

### 6.1 Preprocessing (`amharic_lm/preprocess.py`)

GPAC is distributed as one 1 GB line of text. The pipeline applies these steps, in order:

1. **Repair.** The corpus creators deleted `.`, `,` and `-`. Abbreviations and numbers therefore fell apart:
   `ዓ.ም` became `ዓ ም`, `እ.ኤ.አ` became `እ ኤ አ`, and `8.4` became `8 4`. A run of single letters is joined back
   with `/` (`ዓ ም` → `ዓ/ም`), the standard Amharic abbreviation form, and a run of digit groups is joined
   (`8 4` → `84`).
2. **Sentence splitting** on `።`, `፡፡`, `::`, `?`, `!`, `፧` and `፨`.
3. **Tokenisation.** A token is a run of Ethiopic letters or digits, which may contain `/` (e.g. `ት/ቤት`).
   Other punctuation (`፣ ፤ ፥ ፦ ( ) %` …) separates tokens and is dropped.
4. **Homophone normalisation.** Letters that sound the same are merged so one word is not counted under
   several spellings: ሐ, ኀ → ሀ; ሠ → ሰ; ዐ → አ; ፀ → ጸ; ሃ, ሓ, ኃ → ሀ; ኣ, ዓ → አ; ዉ → ው. This merged
   189,702 spelling variants (9.5% of raw types).
5. **Numbers.** Every run of digits becomes `<NUM>`, so `2010` → `<NUM>` and `ለ19ኛ` → `ለ<NUM>ኛ`.
6. **De-duplication.** 24% of sentences repeat an earlier one (re-published news, boilerplate). Repeats with at
   least 5 tokens are removed: 1.0M sentences, 19% of all tokens. Short repeats such as "አመሰግናለሁ" are kept
   because they are genuinely frequent.

Each word keeps its **most frequent original spelling** for display, so tables and the app show natural text
even though counting uses the normalised form.

### 6.2 Storing n-grams at scale (`amharic_lm/ngram.py`)

The clean corpus has 23.7M distinct bigrams, 44.4M trigrams and 52.1M 4-grams. Python dictionaries of
tuples would need tens of gigabytes, so each order is stored as **sorted numpy arrays**:

* Each word is an integer id below 2²¹ (the vocabulary has 1.81M types).
* A context of up to three words is **packed into one 64-bit integer**, 21 bits per word.
* For each order: `ctx_keys` (sorted unique contexts), `ctx_start` (where each context's n-grams begin),
  `keys` (`context_index × V + word`, sorted) and `counts`.
* Any lookup is a binary search (`np.searchsorted`) and works on millions of queries at once. All words that
  follow a context form one contiguous slice, which is exactly what next-word suggestion needs.

Building all four orders takes about 40 seconds; scoring the 3.3M test events takes about a second.

Every sentence is padded with n−1 start symbols `<s>` and one end symbol `</s>`. The model predicts `</s>`
(so it learns where sentences end) but never predicts `<s>`.

### 6.3 Probability estimators

Here *h* is the context (the previous n−1 words), *c(·)* is a count and |V| is the vocabulary size.

**Maximum likelihood (MLE)**

$$P_{\text{MLE}}(w \mid h) = \frac{c(h, w)}{c(h)}$$

It gives probability 0 to any n-gram not seen in training, so a single unseen n-gram makes a whole sentence
impossible and perplexity infinite.

**Laplace (add-one) and add-k**

$$P_{\text{add-}k}(w \mid h) = \frac{c(h, w) + k}{c(h) + k\,|V|}$$

Laplace uses k = 1. Add-k tunes k on the development set (the best values were 3×10⁻⁴ for bigrams, 3×10⁻⁵ for
trigrams and 3×10⁻⁶ for 4-grams). With |V| = 810,853 even a small k moves most of the probability mass to
unseen events.

**Interpolated Kneser-Ney (KN)**

$$P_{\text{KN}}(w \mid h) = \frac{\max\big(c(h, w) - D,\ 0\big)}{c(h)} + \frac{D \cdot N_{1+}(h\,\bullet)}{c(h)}\; P_{\text{KN}}(w \mid h')$$

where *h′* is *h* without its oldest word, N₁₊(h •) is the number of different words seen after *h*, and the
discount D = n₁ / (n₁ + 2n₂) is computed per order from the number of n-grams seen once (n₁) and twice (n₂).
For the lower orders, raw counts are replaced by **continuation counts** (how many different words precede an
n-gram), so a word that is frequent but appears after only a few different words (the textbook example is
*Francisco*, almost always after *San*) gets a low weight when the model backs off. N-grams
that begin with `<s>` keep raw counts, as in KenLM. If a context was never seen, the model backs off to the
shorter context. We verified numerically that every distribution sums to 1.

### 6.4 Sentence probability and generation

* **Sentence probability** (1.7) is the product of P(wᵢ | context) over all words plus `</s>`. The notebook
  reports it in log₁₀ and shows each step, so you can see exactly which n-gram makes an MLE probability 0.
* **Generation** (1.8) samples from the MLE distribution: start from `<s>`, draw the next word in proportion
  to its count after the current context, and stop at `</s>` or 25 words. Each generated sentence is also
  checked against the corpus to see whether it is a verbatim copy.

### 6.5 Evaluation setup (Tasks 2 and 3)

* The clean corpus is cut into blocks of 200 consecutive sentences, which are shuffled (seed 42) and assigned
  90% / 5% / 5% to train / dev / test. Blocks keep neighbouring sentences of an article together, which
  approximates a document-level split and avoids leaking nearly identical sentences into the test set.
* Words seen only once in training become `<UNK>` (vocabulary 810,853). The test set has a 2.60% OOV rate.
* **Perplexity** is exp of the average negative log probability over all 3.28M test tokens, including `</s>`.

### 6.6 Extrinsic tasks (Task 3)

| Task | How it works | Metric |
|---|---|---|
| **3A Next-word prediction** | At 5,000 random positions in test sentences, hide the next word and ask for the top 5 suggestions. | Accuracy @1, @3, @5 and mean reciprocal rank |
| **3A Keystroke savings** | Simulate typing 200 test sentences letter by letter with a 5-word suggestion bar filtered by the letters typed so far; tapping a correct suggestion costs one keystroke. | Keystroke savings rate (KSR) = 1 − keystrokes with / without suggestions |
| **3B Spelling correction** | Replace a word with variants that differ only in vowel order (e.g. እያደረገ / እያደረግ / እያደረጉ), the most common Amharic typing error; the model must pick the original by scoring each whole sentence. | Accuracy (random guessing: 29.4%) |
| **3C Word order** | Compare a real test sentence with the same sentence after swapping two adjacent words, or after a full shuffle. | % of pairs where the real sentence scores higher |

---

## 7. Results

### Corpus

| Statistic | Value |
|---|---:|
| Raw tokens / sentences | 76,729,727 / 5,088,228 |
| Clean tokens / sentences (after de-duplication) | 61,882,505 / 4,085,911 |
| Word types after normalisation | 1,810,800 |
| Words seen once (hapax) | 945,786 (52.2% of types) |
| Mean sentence length | 15.1 tokens |
| Stop-word tokens | 18.6% of the corpus |

| n | Distinct n-grams | Seen only once |
|---|---:|---:|
| 1 | 1,810,801 | 52.2% |
| 2 | 23,686,097 | 74.9% |
| 3 | 44,431,120 | 86.2% |
| 4 | 52,134,952 | 90.5% |

### Task 2: test-set perplexity (lower is better)

| Estimator | n = 1 | n = 2 | n = 3 | n = 4 |
|---|---:|---:|---:|---:|
| MLE (unsmoothed) | 10,609 | ∞ | ∞ | ∞ |
| Laplace (add-1) | 10,642 | 41,202 | 263,659 | 411,381 |
| Add-k (k tuned on dev) | 10,609 | 3,014 | 20,987 | 66,596 |
| **Interpolated Kneser-Ney** | 10,609 | 1,264 | 789.1 | **704.5** |

MLE is infinite because 24% (bigram), 58% (trigram) and 72% (4-gram) of test tokens complete an n-gram never
seen in training. Laplace gets *worse* as n grows: longer contexts are sparser, so the added counts dominate.

### Task 3: extrinsic evaluation (train model, Kneser-Ney)

| n | Next-word acc@1 | acc@5 | MRR@5 | Keystroke savings | Spelling correction | Adjacent swap detected | Full shuffle detected |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.8% | 13.6% | 0.113 | 36.5% | 81.5% | 50.0% | 50.0% |
| 2 | 17.3% | 26.5% | 0.207 | 46.4% | 90.3% | 91.0% | 99.4% |
| 3 | 22.2% | 33.1% | 0.263 | 48.4% | 90.5% | 91.4% | 99.5% |
| 4 | **24.3%** | **34.6%** | **0.282** | **48.8%** | **90.7%** | 91.3% | 99.4% |

**Takeaways for the presentation**

* Context helps most from n = 1 to n = 2; each further step adds less. From n = 3 to n = 4 perplexity still
  improves by 11%, but the extrinsic gains are under 1.5 points.
* Better perplexity mostly means better task results, but not always: with Laplace smoothing, spelling
  correction and word-order accuracy *fall* as n grows, matching its exploding perplexity.
* The assignment sentence "ኢትዮጵያ በአፍሪካ ቀንድ ትገኛለች" gets probability 0 from every unsmoothed model with
  n ≥ 2, because "ቀንድ ትገኛለች" never occurs in the corpus. Kneser-Ney gives it 5.2×10⁻¹² (bigram) to
  1.9×10⁻¹¹ (4-gram), and the per-word perplexity of the sentence falls from 2,252 (unigram) to 140 (4-gram).
* Generated text gets more fluent as n grows, but 41% of 4-gram sentences are exact copies of corpus
  sentences: the model memorises rather than generalises.

---

## 8. Task 4: the next-word suggestion app

### How a suggestion is produced (`amharic_lm/predictor.py`)

The assignment's required pipeline maps onto the code as follows:

| Required step | Implementation |
|---|---|
| Read the input | Keep only the sentence being written: the text after the last `።`, `?` or `!`. |
| Tokenise | Same repair, tokenisation and normalisation as the training corpus, so typing ሐ or ሀ gives the same result. |
| Half-typed word? | If the text does not end with a space, the last token is treated as a **prefix**: only words that start with it are suggested. |
| Extract the last one or two words | Trigram context (last two words) by default; the interface can switch to bigram (one word) or 4-gram (three words). At the start of a sentence the context is padded with `<s>`. |
| Retrieve candidates | The 300 most frequent words seen after the full context, after the shorter context, and overall, merged. |
| Estimate probabilities | Score every candidate with interpolated Kneser-Ney P(w \| context). |
| Return the top 5 | Sorted by probability. Number tokens and `<UNK>` are skipped; the end of sentence is shown as **።**. |
| Choose and continue | The chosen word replaces the half-typed word (or is appended), followed by a space, and new suggestions appear. |

The app uses the **full model** (all 61.9M tokens). Each request takes about 10–15 ms.

### Web app

```bash
python app/server.py              # options: --port 8000 --host 127.0.0.1
```

Open <http://localhost:8000>. Loading the model takes a few seconds (the arrays are memory-mapped from disk).

* Type Amharic text; suggestions update as you type.
* Click a suggestion, press **Tab** for the first one, or **Alt+1 … Alt+5** (or Ctrl+1 … 5) to pick by number.
* Choose the context length: bigram, trigram or 4-gram.
* The table under the suggestions shows each word's probability, which context it was seen with and its count.
* Start with text already filled in: `http://localhost:8000/?q=ኢትዮጵያ%20ውስጥ%20የቡና%20`

**HTTP API** (all responses are JSON):

| Endpoint | Parameters | Returns |
|---|---|---|
| `GET /api/suggest` | `text`, `order` (1–4, default 3), `k` (1–10, default 5) | The parsed context, the half-typed prefix and the suggestions |
| `GET /api/apply` | `text`, `word` | `{"text": …}` with the chosen word inserted |
| `GET /api/info` | — | Model size: tokens, sentences, vocabulary, n-grams per order |

Example: `GET /api/suggest?text=ኢትዮጵያ ውስጥ የቡና ` returns (abridged)

```json
{
  "history": ["ኢትዮጵያ", "ውስጥ", "የቡና"],
  "prefix": "",
  "context": ["ውስጥ", "የቡና"],
  "order": 3,
  "suggestions": [
    {"word": "ፍጆታ", "p": 0.1184, "matched": "trigram", "count": 4, "p_mle": 0.1481},
    {"word": "ዋጋ",  "p": 0.0950, "matched": "trigram", "count": 3, "p_mle": 0.1111},
    ...
  ],
  "ms": 13.0
}
```

`matched` is the longest context the word was actually seen with; `count` and `p_mle` are its raw count and
unsmoothed probability in that context.

### Terminal app

```bash
python app/cli.py --order 3
```

| Input | Effect |
|---|---|
| A word, then Enter | Add the word to the sentence |
| `1` … `5` | Accept that suggestion |
| A partial word ending in `+`, e.g. `አብ+` | Show completions of the partial word |
| `:2`, `:3`, `:4` | Switch to bigram, trigram or 4-gram context |
| `:c` | Clear the sentence |
| `:q` | Quit |

---

## 9. Code reference

All code is in the `amharic_lm` package. Run Python from the project root (or set `PYTHONPATH` to it) so the
package can be imported.

### Quick examples

```python
from amharic_lm import pipeline
from amharic_lm.preprocess import tokenize
from amharic_lm.predictor import NextWordPredictor

model = pipeline.full_model(mmap=True)           # loads the cached model (builds it if missing)

# probability of a sentence, step by step
ids = model.encode(tokenize("ኢትዮጵያ በአፍሪካ ቀንድ ትገኛለች"))
ctx, words, p = model.sentence_logprob(ids, order=3, method="kn")
print(model.words(words), p)

# top-5 next words
predictor = NextWordPredictor(model, order=3)
for s in predictor.suggest("ኢትዮጵያ ውስጥ የቡና ")["suggestions"]:
    print(s["word"], round(s["p"], 4))

# a random 4-gram sentence
import numpy as np
print(" ".join(model.words(model.generate(4, np.random.default_rng(0)))))
```

```python
# perplexity on the held-out test set
from amharic_lm import pipeline
train, dev, test = pipeline.split_corpora()
tm = pipeline.train_model(mmap=True)
ppl, _ = tm.perplexity(test, order=4, method="kn")      # 704.5
```

### `preprocess.py` — text processing

| Function | Purpose |
|---|---|
| `repair(text)` | Re-join split abbreviations (`ዓ ም` → `ዓ/ም`) and numbers (`8 4` → `84`) |
| `normalize_chars(text)` | Merge homophone letters only |
| `normalize_token(token)` | Homophones plus digit runs → `<NUM>` |
| `split_sentences(text)` | Split on Amharic and Latin sentence terminators |
| `tokenize(text, normalize=True)` | Repair and tokenise; returns normalised tokens by default |
| `tokenize_sentences(text)` | List of token lists, one per sentence |

### `corpus.py` — encoded corpus

| Name | Purpose |
|---|---|
| `Vocab` | `itos` (id → normalised word), `counts`, `display` (most frequent original spelling); `encode`, `decode`, `stoi` |
| `EncodedCorpus` | `ids` (int32 word ids of all sentences) and `offsets` (sentence boundaries); `subset`, `filter_tokens`, `map_ids`, `sentence(i)` |
| `encode_text(text)` | Raw text → `EncodedCorpus`, in 20M-character chunks; ids sorted by frequency |
| `deduplicate(corpus, min_len=5)` | Drop repeated sentences of at least `min_len` tokens |
| `block_split(n, fractions, block=200, seed=42)` | Sentence indices for train / dev / test |

Ids 0, 1 and 2 are reserved for `<s>`, `</s>` and `<UNK>`.

### `ngram.py` — language model

| Name | Purpose |
|---|---|
| `NGramTable` | One order. `build`, `continuations(context)` (words seen after a context, with counts), `count`, `top(k)` (most frequent n-grams), `save` / `load` |
| `NGramModel.train(corpus, max_order=4, min_count=1)` | Build orders 1…max_order and the Kneser-Ney continuation counts |
| `NGramModel.prob(ctx2d, words, order, method, k)` | Batched P(w \| context); `method` is `"mle"`, `"laplace"`, `"addk"` or `"kn"` |
| `NGramModel.perplexity(corpus, order, method, k)` | Perplexity and per-token log probabilities |
| `NGramModel.sentence_logprob(ids, order, method)` | Per-step probabilities of one sentence |
| `NGramModel.score_sentences(list_of_ids, order, method)` | Total log₁₀ probability of many sentences |
| `NGramModel.generate(order, rng, max_len=25, prefix=())` | Sample a sentence from the MLE distribution |
| `NGramModel.suggest(history, k=5, order=3, prefix="")` | Top-k next words with probability, matched order and count |
| `NGramModel.encode(tokens)` / `words(ids)` | Normalised tokens ↔ model ids / display strings |

### `analysis.py` — Task 1 helpers

`corpus_summary`, `char_stats`, `top_words`, `zipf_fit`, `heaps_curve` (1.1); `ngrams`, `table_summary` (1.2);
`top_ngrams` (1.3, 1.5); `ngram_frequencies` (1.6); `sentence_report`, `sentence_probability_table` (1.7);
`generation_report` (1.8).

### `evaluation.py` — Tasks 2 and 3

| Function | Task |
|---|---|
| `perplexity_table(model, dev, test)` | Perplexity of all estimators for n = 1–4 (tunes add-k on dev) |
| `ngram_coverage(model, test)` | Share of test events whose n-gram / context was seen |
| `next_word_prediction(model, test, n_events=5000)` | 3A accuracy@k and MRR |
| `keystroke_savings(model, test, n_sentences=150)` | 3A keystroke savings (the notebook uses 200) |
| `spelling_correction(model, test, n_trials=3000, method="kn")` | 3B vowel-order confusion correction |
| `word_order_test(model, test, n_pairs=5000, method="kn")` | 3C adjacent swap and full shuffle |

### Other modules

* `stopwords.py`: `STOPWORDS` (241 words), `is_stopword(token)` (also flags single letters and number tokens),
  `stopword_mask(itos)`.
* `predictor.py`: `NextWordPredictor(model, order=3)` with `parse(text)`, `suggest(text, k=5, order=None)` and
  `apply(text, word)`.
* `pipeline.py`: `download_corpus`, `raw_corpus`, `clean_corpus`, `splits`, `split_corpora`, `full_model`,
  `train_model`. Each step loads its cached result if it exists and builds it otherwise.
* `viz.py`: `apply_style()` (chart style and Ethiopic font), `wordcloud(freqs, ax, title)`, `save(fig, name)`.
* `paths.py`: every file location (`CORPUS_TXT`, `MODEL_FULL`, `MODEL_TRAIN`, `FIGURES`, …).

---

## 10. Cached artifacts

| Path | Contents | Size |
|---|---|---:|
| `data/GPAC.txt` | Downloaded corpus | 1.0 GB |
| `artifacts/corpus_raw/` | Encoded corpus before de-duplication | — |
| `artifacts/corpus_clean/` | Encoded, de-duplicated corpus | — |
| `artifacts/splits.npz` | Train / dev / test sentence indices | — |
| `artifacts/models/full/` | Orders 1–4 on all clean data (Task 1, Task 4) | — |
| `artifacts/models/train/` | Orders 1–4 on the 90% split with `<UNK>` (Tasks 2–3) | — |
| **`artifacts/` total** | | **5.9 GB** |

Both folders are git-ignored and can be deleted at any time; they are rebuilt automatically (about 2 minutes,
plus the download for `data/`). **Do not share them with the group**: share the code and the notebook, and let
each person rebuild. If you change preprocessing, delete `artifacts/` so the models are rebuilt with the
new rules.

---

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'amharic_lm'` | Run from the project root, or `export PYTHONPATH=$PWD`. The VS Code run configurations set this for you. |
| Notebook uses the wrong Python / packages missing | Select the `.venv` kernel (top right of the notebook). |
| Process killed or very slow while training | Building needs about 6 GB of RAM. Close other programs; on Colab use a high-RAM runtime. |
| Download from Google Drive fails | Download GPAC manually from the paper's link and save it as `data/GPAC.txt`. |
| Amharic shows as boxes in charts | `fonts/NotoSansEthiopic.ttf` is missing; restore it. |
| `OSError: Address already in use` when starting the web app | Port 8000 is taken: `python app/server.py --port 8001`. |
| VS Code is slow or uses a lot of CPU | Make sure `.vscode/settings.json` is present: it stops VS Code from indexing the 7 GB of data. |

---

## 12. Known limitations

* **Missing letter in the corpus.** The letter ኃ never occurs in GPAC (it was apparently lost when the corpus
  was built), so words such as ኃላፊነት appear as ላፊነት. Those truncated forms are what the model learns.
* **Stop words.** The 241-word list was curated for this project and should be reviewed by a native speaker.
* **Repair heuristic.** Joining single letters occasionally creates a wrong token such as `ዓ/ም/በ`. This is rare.
* **No morphological analysis.** Amharic attaches prepositions, articles and possessives to words
  (የቡና, በአፍሪካ, ልጆቿን), which is why the vocabulary is so large (1.8M types) and why half of all words are seen
  only once. A subword or morpheme-level model would reduce sparsity.
* **Generation uses MLE**, so 4-gram samples often copy training sentences.

---

## 13. References

* Gereme, F., Zhu, W., Ayall, T., & Alemu, D. (2021). Combating Fake News in "Low-Resource" Languages: Amharic
  Fake News Detection Accompanied by Resource Crafting. *Information*, 12(1), 20.
  <https://doi.org/10.3390/info12010020> (source of the GPAC corpus)
* Jurafsky, D., & Martin, J. H. *Speech and Language Processing* (3rd ed. draft), Chapter 3: N-gram Language
  Models.
* Chen, S. F., & Goodman, J. (1999). An empirical study of smoothing techniques for language modeling.
  *Computer Speech & Language*, 13(4), 359–394.
* Heafield, K. (2011). KenLM: Faster and smaller language model queries. *Proceedings of the Sixth Workshop on
  Statistical Machine Translation*.
