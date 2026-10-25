"""Intrinsic (perplexity) and extrinsic (task-based) evaluation."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .corpus import BOS, EOS, UNK, EncodedCorpus
from .ngram import NGramModel

# ==========================================================================
# Task 2 -- intrinsic evaluation: perplexity on held-out text
# ==========================================================================
def oov_rate(model: NGramModel, corpus: EncodedCorpus) -> float:
    mapped = model.id_map[corpus.ids]
    return float((mapped == UNK).mean())


def tune_addk(model: NGramModel, dev: EncodedCorpus, order: int,
              grid=(1.0, 0.1, 0.03, 0.01, 0.003, 0.001, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6)):
    """Pick k for add-k smoothing by dev-set perplexity."""
    scores = {k: model.perplexity(dev, order, "addk", k)[0] for k in grid}
    best = min(scores, key=scores.get)
    return best, scores


def perplexity_table(model: NGramModel, dev: EncodedCorpus, test: EncodedCorpus,
                     orders=(1, 2, 3, 4), verbose: bool = True) -> pd.DataFrame:
    rows = []
    for n in orders:
        t0 = time.time()
        # MLE: unseen events get probability 0 -> infinite perplexity
        _, lp = model.perplexity(test, n, "mle")
        finite = np.isfinite(lp)
        rows.append({"n": n, "method": "MLE (unsmoothed)",
                     "perplexity": float(np.exp(-lp.mean())) if finite.all() else np.inf,
                     "zero_prob_%": 100 * (1 - finite.mean()),
                     "ppl_on_seen_events": float(np.exp(-lp[finite].mean())), "k": None})
        pp, _ = model.perplexity(test, n, "laplace")
        rows.append({"n": n, "method": "Laplace (add-1)", "perplexity": pp, "zero_prob_%": 0.0,
                     "ppl_on_seen_events": None, "k": 1.0})
        k, _ = tune_addk(model, dev, n)
        pp, _ = model.perplexity(test, n, "addk", k)
        rows.append({"n": n, "method": "Add-k (k tuned on dev)", "perplexity": pp,
                     "zero_prob_%": 0.0, "ppl_on_seen_events": None, "k": k})
        pp, _ = model.perplexity(test, n, "kn")
        rows.append({"n": n, "method": "Interpolated Kneser-Ney", "perplexity": pp,
                     "zero_prob_%": 0.0, "ppl_on_seen_events": None, "k": None})
        if verbose:
            print(f"  n={n} done ({time.time() - t0:.1f}s)")
    df = pd.DataFrame(rows)
    df["cross_entropy_bits"] = np.log2(df["perplexity"])
    return df


def ngram_coverage(model: NGramModel, test: EncodedCorpus, orders=(1, 2, 3, 4)) -> pd.DataFrame:
    """Share of test n-gram events whose n-gram / context was seen in training."""
    from .ngram import windows
    mapped = test.map_ids(model.id_map)
    rows = []
    for n in orders:
        ctx, words = windows(mapped.ids, mapped.offsets, n)
        c, tot = model.tables[n].count(ctx, words)
        rows.append({"n": n, "test_events": len(words),
                     "ngram_seen_%": 100 * float((c > 0).mean()),
                     "context_seen_%": 100 * float((tot > 0).mean())})
    return pd.DataFrame(rows)


# ==========================================================================
# Task 3 -- extrinsic evaluation
# ==========================================================================
def _sample_sentences(corpus: EncodedCorpus, n: int, rng, min_len=5, max_len=30):
    lens = corpus.sentence_lengths()
    ok = np.flatnonzero((lens >= min_len) & (lens <= max_len))
    return rng.choice(ok, size=min(n, len(ok)), replace=False)


# --------------------------------------------------------------------------
# 3A. next-word prediction (the core of a predictive keyboard)
# --------------------------------------------------------------------------
def next_word_prediction(model: NGramModel, test: EncodedCorpus, orders=(1, 2, 3, 4),
                         n_events: int = 5000, k: int = 5, seed: int = 0) -> pd.DataFrame:
    """Hide the next word, ask the model for its top-k suggestions.
    Reports accuracy@1/@3/@5 and mean reciprocal rank (MRR@k).  One random
    position per sampled test sentence; the end of the sentence (</s>, shown
    as "።" in the app) is a legitimate target, as in the LM itself."""
    rng = np.random.default_rng(seed)
    sents = _sample_sentences(test, n_events, rng, min_len=2, max_len=60)
    events = []
    for s in sents:
        ids = list(model.id_map[test.ids[test.offsets[s]:test.offsets[s + 1]]]) + [EOS]
        i = int(rng.integers(len(ids)))
        events.append((ids[:i], int(ids[i])))
    rows = []
    for n in orders:
        t0 = time.time()
        ranks = []
        for hist, target in events:
            ids = [s["id"] for s in model.suggest(hist, k=k, order=n, details=False)]
            ranks.append(ids.index(target) + 1 if target in ids else 0)
        ranks = np.array(ranks)
        rows.append({"n": n, "events": len(ranks),
                     "acc@1_%": 100 * np.mean(ranks == 1),
                     "acc@3_%": 100 * np.mean((ranks >= 1) & (ranks <= 3)),
                     "acc@5_%": 100 * np.mean(ranks >= 1),
                     "MRR@5": float(np.mean(np.where(ranks > 0, 1 / np.maximum(ranks, 1), 0))),
                     "seconds": round(time.time() - t0, 1)})
    df = pd.DataFrame(rows)
    df.attrs["target_is_unk_%"] = 100 * np.mean([t == UNK for _, t in events])
    return df


def keystroke_savings(model: NGramModel, test: EncodedCorpus, orders=(1, 2, 3, 4),
                      n_sentences: int = 150, k: int = 5, seed: int = 1) -> pd.DataFrame:
    """Simulate a user typing test sentences with a top-k suggestion bar.

    Before every character the user looks at the k suggestions (filtered by
    the letters typed so far); if the intended word is there they tap it
    (1 keystroke, which also inserts the space).  Otherwise they type the next
    letter.  Unknown words must be typed in full.  Keystrokes are counted in
    Ethiopic characters (fidel)."""
    rng = np.random.default_rng(seed)
    sents = _sample_sentences(test, n_sentences, rng, min_len=4, max_len=25)
    itos = model.vocab.itos
    rows = []
    for n in orders:
        t0 = time.time()
        baseline = used = 0
        for s in sents:
            ids = model.id_map[test.ids[test.offsets[s]:test.offsets[s + 1]]]
            raw = test.ids[test.offsets[s]:test.offsets[s + 1]]
            for i, (target, raw_id) in enumerate(zip(ids, raw)):
                word = itos[raw_id]
                cost = len(word) + 1                      # letters + space
                baseline += cost
                if target != UNK:
                    hist = list(ids[:i])
                    for typed in range(len(word)):
                        sug = model.suggest(hist, k=k, order=n, prefix=word[:typed], details=False)
                        if any(x["id"] == target for x in sug):
                            cost = typed + 1
                            break
                used += cost
        rows.append({"n": n, "keystrokes_without": baseline, "keystrokes_with": used,
                     "KSR_%": 100 * (1 - used / baseline), "seconds": round(time.time() - t0, 1)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3B. context-sensitive spelling correction (vowel-order confusions)
# --------------------------------------------------------------------------
def _fidel_variants(word: str):
    """Words obtained by changing the vowel order of ONE syllable
    (e.g. ሰበረ -> ሰበሩ / ሰበራ / ሰባረ ...).  This is the most common
    real-word typing error in Ethiopic script."""
    out = set()
    for i, ch in enumerate(word):
        o = ord(ch)
        if not (0x1200 <= o <= 0x1357):
            continue
        base = o - (o - 0x1200) % 8
        for order in range(7):
            alt = base + order
            if alt != o:
                out.add(word[:i] + chr(alt) + word[i + 1:])
    return out


def spelling_correction(model: NGramModel, test: EncodedCorpus, orders=(1, 2, 3, 4),
                        n_trials: int = 3000, min_freq: int = 20, max_cands: int = 8,
                        seed: int = 2, method: str = "kn") -> pd.DataFrame:
    """In a held-out sentence replace one word by a confusable variant; the
    LM must pick the original word among {original + variants} by scoring
    each full candidate sentence.  Unigram = 'choose the most frequent word'."""
    rng = np.random.default_rng(seed)
    stoi, itos = model.vocab.stoi, model.vocab.itos
    freq = model.vocab.counts
    in_model = model.id_map != UNK

    def ok(i):
        return in_model[i] and freq[i] >= min_freq and "<NUM>" not in itos[i]

    trials = []
    for s in _sample_sentences(test, n_trials * 3, rng):
        ids = list(model.id_map[test.ids[test.offsets[s]:test.offsets[s + 1]]])
        positions = rng.permutation(len(ids))
        for pos in positions:
            w = ids[pos]
            if w in (UNK,) or not ok(w):
                continue
            cands = [stoi[v] for v in sorted(_fidel_variants(itos[w])) if v in stoi and ok(stoi[v])]
            if cands:
                cands = list(rng.permutation(cands)[:max_cands])
                trials.append((ids, pos, [w] + [int(c) for c in cands]))
                break
        if len(trials) >= n_trials:
            break

    rows = []
    for n in orders:
        sents, owners = [], []
        for t, (ids, pos, cands) in enumerate(trials):
            for c in cands:
                sents.append(ids[:pos] + [c] + ids[pos + 1:])
                owners.append(t)
        scores = model.score_sentences(sents, n, method)
        owners = np.array(owners)
        # candidates of one trial are contiguous; index 0 is the original word
        starts = np.flatnonzero(np.r_[True, owners[1:] != owners[:-1]])
        best = np.array([scores[a:b].argmax() for a, b in zip(starts, np.r_[starts[1:], len(scores)])])
        correct = float(np.mean(best == 0))
        rows.append({"n": n, "trials": len(trials), "accuracy_%": 100 * correct})
    df = pd.DataFrame(rows)
    df.attrs["avg_candidates"] = float(np.mean([len(c) for _, _, c in trials]))
    df.attrs["random_baseline_%"] = float(100 * np.mean([1 / len(c) for _, _, c in trials]))
    df.attrs["examples"] = [(ids, pos, cands) for ids, pos, cands in trials[:5]]
    return df


# --------------------------------------------------------------------------
# 3C. word-order discrimination (fluency judgement / n-best re-ranking)
# --------------------------------------------------------------------------
def word_order_test(model: NGramModel, test: EncodedCorpus, orders=(1, 2, 3, 4),
                    n_pairs: int = 5000, seed: int = 3, method: str = "kn") -> pd.DataFrame:
    """Does the LM give a real sentence a higher probability than (a) the same
    sentence with two adjacent words swapped, (b) a random shuffle of it?
    Ties count as half correct (a unigram model always ties)."""
    rng = np.random.default_rng(seed)
    orig, swap, shuf = [], [], []
    for s in _sample_sentences(test, n_pairs, rng):
        ids = list(model.id_map[test.ids[test.offsets[s]:test.offsets[s + 1]]])
        i = int(rng.integers(len(ids) - 1))
        if ids[i] == ids[i + 1]:
            continue
        sw = ids.copy()
        sw[i], sw[i + 1] = sw[i + 1], sw[i]
        sh = list(rng.permutation(ids))
        if sh == ids:
            continue
        orig.append(ids); swap.append(sw); shuf.append(sh)
    rows = []
    for n in orders:
        so = model.score_sentences(orig, n, method)
        res = {"n": n, "pairs": len(orig)}
        for name, corrupted in (("adjacent_swap", swap), ("full_shuffle", shuf)):
            sc = model.score_sentences(corrupted, n, method)
            diff = so - sc
            res[f"{name}_acc_%"] = 100 * float(np.mean(np.where(np.isclose(diff, 0, atol=1e-9), 0.5, diff > 0)))
        rows.append(res)
    return pd.DataFrame(rows)
