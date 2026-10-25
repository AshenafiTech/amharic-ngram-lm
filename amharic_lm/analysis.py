"""Task 1 helpers: corpus statistics, n-gram listings, sentence scoring and
generation analysis."""
from __future__ import annotations

import unicodedata
from collections import Counter

import numpy as np
import pandas as pd

from .corpus import BOS, EOS, UNK, EncodedCorpus, sentence_hashes
from .ngram import NGramModel, NGramTable, windows
from .preprocess import tokenize

# --------------------------------------------------------------------------
# 1.1 corpus statistics
# --------------------------------------------------------------------------
def char_category(ch: str) -> str:
    o = ord(ch)
    if 0x1200 <= o <= 0x135A or 0x1380 <= o <= 0x138F or 0x2D80 <= o <= 0x2DDE:
        return "Ethiopic letter"
    if 0x1361 <= o <= 0x1368:
        return "Ethiopic punctuation"
    if 0x1369 <= o <= 0x137C:
        return "Ethiopic numeral"
    if ch.isspace():
        return "whitespace"
    if ch.isdigit():
        return "ASCII digit"
    if ch.isascii() and ch.isalpha():
        return "Latin letter"
    return "other symbol"


def char_stats(text: str):
    """Character inventory of the raw text -> (category table, symbol table)."""
    counts = Counter(text)
    cats = Counter()
    for ch, n in counts.items():
        cats[char_category(ch)] += n
    total = sum(cats.values())
    cat_df = (pd.DataFrame([(k, v, 100 * v / total) for k, v in cats.items()],
                           columns=["category", "characters", "%"])
              .sort_values("characters", ascending=False, ignore_index=True))
    sym = [(ch, f"U+{ord(ch):04X}", unicodedata.name(ch, "?").title(), n)
           for ch, n in counts.items() if char_category(ch) in ("Ethiopic punctuation", "other symbol")]
    sym_df = (pd.DataFrame(sym, columns=["symbol", "code point", "name", "count"])
              .sort_values("count", ascending=False, ignore_index=True))
    return cat_df, sym_df, len(counts)


def corpus_summary(corpus: EncodedCorpus) -> dict:
    lens = corpus.sentence_lengths()
    freq = np.bincount(corpus.ids, minlength=len(corpus.vocab))
    freq[[BOS, EOS, UNK]] = 0
    types = int((freq > 0).sum())
    word_len = np.array([len(w) for w in corpus.vocab.itos])
    present = freq > 0
    return {
        "sentences": corpus.n_sentences,
        "tokens": corpus.n_tokens,
        "types (vocabulary)": types,
        "type/token ratio": types / corpus.n_tokens,
        "hapax legomena (freq = 1)": int((freq == 1).sum()),
        "hapax, % of types": 100 * (freq == 1).sum() / types,
        "dis legomena (freq = 2)": int((freq == 2).sum()),
        "mean sentence length (tokens)": float(lens.mean()),
        "median sentence length": float(np.median(lens)),
        "longest sentence": int(lens.max()),
        "mean word length (chars, per token)": float((word_len * freq).sum() / freq.sum()),
        "mean word length (chars, per type)": float(word_len[present].mean()),
        "top-100 words cover % of tokens": 100 * np.sort(freq)[::-1][:100].sum() / freq.sum(),
    }


def top_words(corpus: EncodedCorpus, k: int = 25) -> pd.DataFrame:
    freq = np.bincount(corpus.ids, minlength=len(corpus.vocab))
    freq[[BOS, EOS, UNK]] = 0
    order = np.argsort(-freq)[:k]
    total = freq.sum()
    df = pd.DataFrame({"rank": np.arange(1, k + 1),
                       "word": [corpus.vocab.display[i] for i in order],
                       "count": freq[order],
                       "relative freq %": 100 * freq[order] / total})
    df["cumulative %"] = df["relative freq %"].cumsum()
    return df


def zipf_fit(freq: np.ndarray, rank_range=(10, 10_000)):
    """Least-squares slope of log(freq) vs log(rank)."""
    f = np.sort(freq[freq > 0])[::-1]
    r = np.arange(1, len(f) + 1)
    lo, hi = rank_range
    s, b = np.polyfit(np.log10(r[lo:hi]), np.log10(f[lo:hi]), 1)
    return f, r, s, b


def heaps_curve(ids: np.ndarray, points: int = 60):
    """Vocabulary size after the first N tokens, for log-spaced N."""
    _, first = np.unique(ids, return_index=True)
    first = np.sort(first)
    Ns = np.unique(np.logspace(3, np.log10(len(ids)), points).astype(np.int64))
    Vs = np.searchsorted(first, Ns)
    beta, logK = np.polyfit(np.log10(Ns), np.log10(Vs), 1)
    return Ns, Vs, 10 ** logK, beta


# --------------------------------------------------------------------------
# 1.2 / 1.3 n-grams
# --------------------------------------------------------------------------
def ngrams(tokens: list[str], n: int, pad: bool = True) -> list[tuple]:
    """Plain-python n-gram extraction, for illustration."""
    seq = (["<s>"] * (n - 1) + tokens + ["</s>"]) if pad else tokens
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def within_sentence_ngrams(corpus: EncodedCorpus, n: int) -> int:
    """Number of n-gram tokens inside sentences (no <s>/</s> involved)."""
    return int(np.maximum(corpus.sentence_lengths() - n + 1, 0).sum())


def table_summary(tables: dict[int, NGramTable], corpus: EncodedCorpus) -> pd.DataFrame:
    rows = []
    for n, t in tables.items():
        rows.append({"n": n,
                     "n-gram tokens (with <s>,</s>)": int(t.ctx_total.sum()),
                     "distinct n-grams": len(t),
                     "distinct contexts": t.n_contexts if n > 1 else 1,
                     "singletons": int((t.counts == 1).sum()),
                     "singletons %": 100 * float((t.counts == 1).mean()),
                     "max count": int(t.counts.max())})
    return pd.DataFrame(rows)


def ngram_string(model_vocab, toks) -> str:
    return " ".join(model_vocab.display[t] if model_vocab.display else model_vocab.itos[t] for t in toks)


def top_ngrams(tables: dict[int, NGramTable], n: int, corpus: EncodedCorpus, k: int = 10,
               exclude=(BOS, EOS, UNK)) -> pd.DataFrame:
    """k most frequent n-grams with joint and conditional MLE probabilities.

    P(w1..wn)       = C(w1..wn) / (number of n-gram tokens inside sentences)
    P(wn | w1..wn-1) = C(w1..wn) / C(w1..wn-1 •)
    """
    t = tables[n]
    idx = t.top(k, exclude)
    toks = t.ngram_tokens(idx)
    cnt = t.counts[idx].astype(np.int64)
    N = within_sentence_ngrams(corpus, n)
    ctx_tot = t.ctx_total[t.keys[idx] // t.V]
    return pd.DataFrame({
        "rank": np.arange(1, len(idx) + 1),
        f"{n}-gram": [ngram_string(corpus.vocab, r) for r in toks],
        "count": cnt,
        "P(n-gram)": cnt / N,
        "P(w_n | history)": cnt / ctx_tot,
    })


def ngram_frequencies(tables: dict[int, NGramTable], n: int, vocab, k: int = 150,
                      exclude=(BOS, EOS, UNK)) -> dict[str, int]:
    t = tables[n]
    idx = t.top(k, exclude)
    return {ngram_string(vocab, r): int(c) for r, c in zip(t.ngram_tokens(idx), t.counts[idx])}


# --------------------------------------------------------------------------
# 1.7 sentence probability
# --------------------------------------------------------------------------
def sentence_report(model: NGramModel, text: str, order: int, method: str = "mle"):
    tokens = tokenize(text)
    ids = model.encode(tokens)
    ctx, words, p = model.sentence_logprob(ids, order, method)
    disp = model.vocab.display
    rows = [{"history": " ".join(disp[c] for c in row) if order > 1 else "-",
             "word": disp[w], "P(word | history)": pr} for row, w, pr in zip(ctx, words, p)]
    with np.errstate(divide="ignore"):
        log10p = float(np.log10(p).sum())
    return pd.DataFrame(rows), {"log10 P": log10p, "P": 10 ** log10p,
                                "perplexity": 10 ** (-log10p / len(p)) if np.isfinite(log10p) else np.inf}


def sentence_probability_table(model: NGramModel, sentences, orders=(1, 2, 3, 4),
                               methods=("mle", "kn")) -> pd.DataFrame:
    rows = []
    for s in sentences:
        for m in methods:
            row = {"sentence": s, "estimator": m.upper()}
            for n in orders:
                _, summ = sentence_report(model, s, n, m)
                row[f"{n}-gram P"] = summ["P"]
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 1.8 generation
# --------------------------------------------------------------------------
def hash_id_lists(sentences: list) -> np.ndarray:
    flat = np.concatenate([np.asarray(s, dtype=np.int32) for s in sentences])
    off = np.concatenate([[0], np.cumsum([len(s) for s in sentences])]).astype(np.int64)
    return sentence_hashes(EncodedCorpus(flat, off, None))


def generation_report(model: NGramModel, corpus_hashes: np.ndarray, orders=(1, 2, 3, 4),
                      n_samples: int = 300, seed: int = 0, max_len: int = 30) -> pd.DataFrame:
    """Quantify what happens as n grows: length, how much of the output is
    made of real corpus 4-grams, and how often a whole training sentence is
    reproduced verbatim."""
    rows = []
    t4 = model.tables[max(model.tables)]
    for n in orders:
        rng = np.random.default_rng(seed + n)
        sents = [model.generate(n, rng, max_len) for _ in range(n_samples)]
        sents = [s for s in sents if s]
        flat = np.concatenate([np.asarray(s, dtype=np.int32) for s in sents])
        off = np.concatenate([[0], np.cumsum([len(s) for s in sents])]).astype(np.int64)
        ctx, words = windows(flat, off, t4.n)
        c, _ = t4.count(ctx, words)
        h = hash_id_lists(sents)
        rows.append({"n": n, "samples": len(sents),
                     "mean length": float(np.mean([len(s) for s in sents])),
                     "hit max length %": 100 * float(np.mean([len(s) >= max_len for s in sents])),
                     "4-grams seen in corpus %": 100 * float((c > 0).mean()),
                     "verbatim corpus sentence %": 100 * float(np.isin(h, corpus_hashes).mean())})
    return pd.DataFrame(rows)
