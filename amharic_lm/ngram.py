"""Count-based n-gram language models backed by sorted numpy arrays.

Storage
-------
Python dicts of tuples do not scale to ~80M tokens, so every order n has an
:class:`NGramTable` made of flat sorted arrays:

* a context (the n-1 previous words) is packed into one int64,
  21 bits per word (vocabulary < 2**21 = 2,097,152 types), oldest word in the
  highest bits;
* ``ctx_keys``  sorted unique packed contexts               int64[C]
* ``ctx_start`` n-grams of context c live in [ctx_start[c], ctx_start[c+1])
* ``keys``      ``context_index * V + word``, sorted          int64[G]
* ``counts``    frequency of every n-gram                    int32[G]

Any lookup is a binary search (``np.searchsorted``), so millions of n-grams
can be scored at once, and the continuations of a context are a contiguous
slice -- exactly what the next-word suggester needs.

Sentences are padded with n-1 ``<s>`` and one ``</s>``.  ``<s>`` is never
predicted; ``</s>`` is.

Estimators
----------
* ``mle``      c(h, w) / c(h)
* ``laplace``  (c(h, w) + 1) / (c(h) + |V|)
* ``addk``     (c(h, w) + k) / (c(h) + k|V|)
* ``kn``       interpolated Kneser-Ney (one absolute discount per order,
               D = n1 / (n1 + 2 n2)); lower orders use continuation counts
               N1+(. h w), except n-grams that start with <s>, which keep
               their raw counts (same convention as KenLM).
"""
from __future__ import annotations

import bisect
import json
import time
from pathlib import Path

import numpy as np

from .corpus import BOS, EOS, UNK, EncodedCorpus, Vocab

BITS = 21
MAX_VOCAB = 1 << BITS
MASK = MAX_VOCAB - 1
METHODS = ("mle", "laplace", "addk", "kn")


# --------------------------------------------------------------------------
# packing helpers
# --------------------------------------------------------------------------
def pack(ctx) -> np.ndarray:
    """int array [M, k] (oldest word first) -> int64 [M]."""
    ctx = np.asarray(ctx, dtype=np.int64)
    if ctx.ndim == 1:
        ctx = ctx[None, :]
    key = np.zeros(ctx.shape[0], dtype=np.int64)
    for j in range(ctx.shape[1]):
        key = (key << BITS) | ctx[:, j]
    return key


def unpack(keys: np.ndarray, k: int) -> np.ndarray:
    keys = np.asarray(keys, dtype=np.int64)
    out = np.empty((len(keys), k), dtype=np.int64)
    for j in range(k):
        out[:, j] = (keys >> (BITS * (k - 1 - j))) & MASK
    return out


def padded_stream(ids: np.ndarray, offsets: np.ndarray, n: int):
    """Concatenate sentences as  <s>*(n-1) w1 .. wm </s>  and return the stream
    plus the positions of every predicted token (words and </s>)."""
    S, N, pad = len(offsets) - 1, len(ids), n - 1
    stream = np.full(N + S * (pad + 1), BOS, dtype=np.int32)
    sid = np.repeat(np.arange(S, dtype=np.int64), np.diff(offsets))
    stream[np.arange(N, dtype=np.int64) + sid * (pad + 1) + pad] = ids
    del sid
    stream[np.asarray(offsets[1:]) + np.arange(S, dtype=np.int64) * (pad + 1) + pad] = EOS
    pos = np.flatnonzero(stream != BOS)
    return stream, pos


def windows(ids: np.ndarray, offsets: np.ndarray, n: int):
    """All (context, word) prediction events of an n-gram model."""
    stream, pos = padded_stream(ids, offsets, n)
    ctx = np.empty((len(pos), max(n - 1, 0)), dtype=np.int32)
    for j in range(n - 1):
        ctx[:, j] = stream[pos - (n - 1) + j]
    return ctx, stream[pos]


def sentence_windows(sent_ids, n: int):
    """Prediction events for a single sentence given as a list of ids."""
    seq = [BOS] * (n - 1) + list(sent_ids) + [EOS]
    ctx = np.array([seq[i:i + n - 1] for i in range(len(seq) - n + 1)], dtype=np.int64)
    ctx = ctx.reshape(len(seq) - n + 1, n - 1)
    words = np.array(seq[n - 1:], dtype=np.int64)
    return ctx, words


def _discount(counts: np.ndarray) -> float:
    n1 = int((counts == 1).sum())
    n2 = int((counts == 2).sum())
    if n1 == 0:
        return 0.5
    return float(min(max(n1 / (n1 + 2 * n2), 0.05), 0.95))


# --------------------------------------------------------------------------
# one order
# --------------------------------------------------------------------------
class NGramTable:
    ARRAYS = ("ctx_keys", "ctx_start", "keys", "counts", "kn_counts")

    def __init__(self, n, V, ctx_keys, ctx_start, keys, counts, kn_counts=None):
        self.n, self.V = n, V
        self.ctx_keys, self.ctx_start = ctx_keys, ctx_start
        self.keys, self.counts = keys, counts
        self.ctx_types = np.diff(ctx_start)
        self.ctx_total = np.add.reduceat(counts.astype(np.int64), ctx_start[:-1])
        self.disc = _discount(counts)
        self.kn_counts = None
        if kn_counts is not None:
            self.set_kn_counts(kn_counts)

    # ------------------------------------------------------------- building
    @classmethod
    def build(cls, ids, offsets, n: int, V: int) -> "NGramTable":
        stream, pos = padded_stream(ids, offsets, n)
        words = stream[pos].astype(np.int64)
        if n == 1:
            keys, counts = np.unique(words, return_counts=True)
            ctx_keys = np.zeros(1, dtype=np.int64)
        else:
            ctx = np.zeros(len(pos), dtype=np.int64)
            for j in range(n - 1, 0, -1):
                ctx = (ctx << BITS) | stream[pos - j]
            del stream, pos
            ctx_keys, inv = np.unique(ctx, return_inverse=True)
            del ctx
            keys, counts = np.unique(inv.astype(np.int64) * V + words, return_counts=True)
            del inv
        ctx_start = np.searchsorted(keys // V, np.arange(len(ctx_keys) + 1)).astype(np.int64)
        return cls(n, V, ctx_keys, ctx_start, keys, counts.astype(np.int32))

    def set_kn_counts(self, kn_counts):
        self.kn_counts = kn_counts
        self.kn_ctx_total = np.add.reduceat(kn_counts.astype(np.int64), self.ctx_start[:-1])
        self.kn_disc = _discount(kn_counts)

    # ------------------------------------------------------------- lookups
    def __len__(self):
        return len(self.keys)

    @property
    def n_contexts(self) -> int:
        return len(self.ctx_keys)

    def lookup_ctx(self, packed):
        packed = np.asarray(packed, dtype=np.int64)
        if self.n == 1:
            return np.zeros(len(packed), dtype=np.int64), np.ones(len(packed), dtype=bool)
        i = np.searchsorted(self.ctx_keys, packed)
        i = np.minimum(i, len(self.ctx_keys) - 1)
        return i, self.ctx_keys[i] == packed

    def lookup(self, cidx, found_ctx, words):
        q = np.asarray(cidx, dtype=np.int64) * self.V + np.asarray(words, dtype=np.int64)
        j = np.searchsorted(self.keys, q)
        j = np.minimum(j, len(self.keys) - 1)
        return j, found_ctx & (self.keys[j] == q)

    def count(self, ctx2d, words):
        """Raw counts c(h, w) and c(h) for a batch of events."""
        ci, fc = self.lookup_ctx(pack(ctx2d[:, ctx2d.shape[1] - (self.n - 1):]) if self.n > 1
                                 else np.zeros(len(words)))
        j, f = self.lookup(ci, fc, words)
        return (np.where(f, self.counts[j], 0).astype(np.int64),
                np.where(fc, self.ctx_total[ci], 0).astype(np.int64))

    def continuations(self, context_ids):
        """(words, counts) observed after a context (list of n-1 ids)."""
        if self.n == 1:
            return self.keys % self.V, self.counts
        ci, fc = self.lookup_ctx(pack([context_ids]))
        if not fc[0]:
            return np.empty(0, np.int64), np.empty(0, np.int32)
        s, e = self.ctx_start[ci[0]], self.ctx_start[ci[0] + 1]
        return self.keys[s:e] % self.V, self.counts[s:e]

    def ngram_tokens(self, idx) -> np.ndarray:
        """Decode n-gram rows ``idx`` into an int array [len(idx), n]."""
        idx = np.asarray(idx)
        k = self.keys[idx]
        words = (k % self.V)[:, None]
        if self.n == 1:
            return words
        return np.hstack([unpack(self.ctx_keys[k // self.V], self.n - 1), words])

    def top(self, k: int = 10, exclude_ids=(BOS, EOS, UNK)):
        """Indices of the k most frequent n-grams not containing exclude_ids."""
        counts = np.asarray(self.counts)
        pool = max(50 * k, 20_000)
        if len(counts) > pool:     # partial sort first; fall back to a full sort
            cand = np.argpartition(-counts, pool)[:pool]
            order = cand[np.lexsort((cand, -counts[cand]))]
        else:
            order = np.lexsort((np.arange(len(counts)), -counts))
        exclude = np.asarray(list(exclude_ids))
        out = []
        for start in range(0, len(order), 4096):
            chunk = order[start:start + 4096]
            toks = self.ngram_tokens(chunk)
            ok = ~np.isin(toks, exclude).any(axis=1)
            out.extend(chunk[ok][: k - len(out)])
            if len(out) >= k:
                break
        if len(out) < k and len(counts) > pool:
            full = np.lexsort((np.arange(len(counts)), -counts))
            toks = self.ngram_tokens(full)
            return full[~np.isin(toks, exclude).any(axis=1)][:k]
        return np.array(out)

    # ------------------------------------------------------------- io
    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        for name in self.ARRAYS:
            arr = getattr(self, name)
            if arr is not None:
                np.save(directory / f"{name}.npy", arr)
        (directory / "meta.json").write_text(json.dumps({"n": self.n, "V": self.V}))

    @classmethod
    def load(cls, directory: Path, mmap: bool = False):
        mode = "r" if mmap else None
        meta = json.loads((directory / "meta.json").read_text())
        arrs = {}
        for name in cls.ARRAYS:
            p = directory / f"{name}.npy"
            arrs[name] = np.load(p, mmap_mode=mode) if p.exists() else None
        return cls(meta["n"], meta["V"], **arrs)


# --------------------------------------------------------------------------
# the language model
# --------------------------------------------------------------------------
class NGramModel:
    """N-gram LM of orders 1..max_order sharing one vocabulary."""

    def __init__(self, tables: dict[int, NGramTable], vocab: Vocab, id_map: np.ndarray,
                 vocab_size: int, info: dict | None = None):
        self.tables = tables
        self.vocab = vocab
        self.id_map = id_map            # corpus id -> model id (rare words -> <UNK>)
        self.vocab_size = vocab_size    # |V| used by Laplace / KN floor
        self.info = info or {}
        self._sorted_words = None

    @property
    def max_order(self) -> int:
        return max(self.tables)

    # ------------------------------------------------------------- training
    @classmethod
    def train(cls, corpus: EncodedCorpus, max_order: int = 4, min_count: int = 1,
              verbose: bool = True) -> "NGramModel":
        V = len(corpus.vocab)
        if V >= MAX_VOCAB:
            raise ValueError(f"vocabulary of {V} types does not fit in {BITS} bits")
        freq = np.bincount(corpus.ids, minlength=V)
        id_map = np.arange(V, dtype=np.int32)
        if min_count > 1:
            id_map[freq < min_count] = UNK
            id_map[[BOS, EOS]] = [BOS, EOS]
            ids = id_map[corpus.ids]
        else:
            ids = corpus.ids
        tables = {}
        for n in range(1, max_order + 1):
            t0 = time.time()
            tables[n] = NGramTable.build(ids, corpus.offsets, n, V)
            if verbose:
                print(f"  order {n}: {len(tables[n]):>12,} distinct n-grams, "
                      f"{tables[n].n_contexts:>11,} contexts  ({time.time() - t0:.1f}s)")
        vocab_size = len(tables[1]) + int(not (tables[1].keys == UNK).any())
        info = {"min_count": min_count, "train_tokens": int(len(ids)),
                "train_sentences": int(corpus.n_sentences)}
        model = cls(tables, corpus.vocab, id_map, vocab_size, info)
        model._compute_kn_counts(verbose)
        return model

    def _compute_kn_counts(self, verbose=True):
        """Continuation counts N1+(. w_2..w_k) for every order below the max."""
        t0 = time.time()
        for k in range(1, self.max_order):
            lo, hi = self.tables[k], self.tables[k + 1]
            V = hi.V
            words = hi.keys % V
            if k == 1:
                j, found = lo.lookup(np.zeros(len(words), np.int64), True, words)
            else:
                suffix = hi.ctx_keys[hi.keys // V] & ((1 << (BITS * (k - 1))) - 1)
                ci, fc = lo.lookup_ctx(suffix)
                j, found = lo.lookup(ci, fc, words)
            assert found.all(), "suffix n-gram missing from lower order"
            cont = np.bincount(j, minlength=len(lo)).astype(np.int32)
            if k >= 2:   # n-grams starting with <s> cannot be extended: keep raw counts
                first = lo.ctx_keys[lo.keys // V] >> (BITS * (k - 2))
                bos = first == BOS
                cont[bos] = lo.counts[bos]
            lo.set_kn_counts(cont)
        if verbose:
            print(f"  Kneser-Ney continuation counts ({time.time() - t0:.1f}s)")

    # ------------------------------------------------------------- io
    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for n, t in self.tables.items():
            t.save(directory / f"order{n}")
        np.save(directory / "id_map.npy", self.id_map)
        self.vocab.save(directory / "vocab.tsv")
        meta = {"vocab_size": self.vocab_size, "max_order": self.max_order, **self.info}
        (directory / "model.json").write_text(json.dumps(meta, indent=1))

    @classmethod
    def load(cls, directory, mmap: bool = False, max_order: int | None = None):
        directory = Path(directory)
        meta = json.loads((directory / "model.json").read_text())
        top = min(meta["max_order"], max_order or meta["max_order"])
        tables = {n: NGramTable.load(directory / f"order{n}", mmap) for n in range(1, top + 1)}
        vocab = Vocab.load(directory / "vocab.tsv")
        id_map = np.load(directory / "id_map.npy")
        return cls(tables, vocab, id_map, meta["vocab_size"], meta)

    @classmethod
    def load_or_train(cls, directory, corpus_fn, max_order=4, min_count=1, verbose=True):
        directory = Path(directory)
        if (directory / "model.json").exists():
            if verbose:
                print(f"loading cached model from {directory}")
            return cls.load(directory)
        model = cls.train(corpus_fn(), max_order, min_count, verbose)
        model.save(directory)
        return model

    # ------------------------------------------------------------- encoding
    def encode(self, tokens) -> np.ndarray:
        """Normalised tokens -> model ids (unknown words -> <UNK>)."""
        return self.id_map[self.vocab.encode(tokens)]

    def words(self, ids, display=True) -> list[str]:
        return self.vocab.decode(ids, display)

    # ------------------------------------------------------------- probabilities
    def prob(self, ctx2d, words, order: int, method: str = "kn", k: float = 1.0) -> np.ndarray:
        """P(word | context) for a batch.  ``ctx2d`` holds at least order-1
        previous ids per row (oldest first)."""
        if order > self.max_order:
            raise ValueError(f"model has orders 1..{self.max_order}")
        ctx2d = np.asarray(ctx2d, dtype=np.int64)
        words = np.asarray(words, dtype=np.int64)
        if method in ("mle", "laplace", "addk"):
            c, tot = self.tables[order].count(ctx2d, words)
            if method == "mle":
                return np.divide(c, tot, out=np.zeros(len(c)), where=tot > 0)
            kk = 1.0 if method == "laplace" else k
            return (c + kk) / (tot + kk * self.vocab_size)
        if method != "kn":
            raise ValueError(method)

        # ---- interpolated Kneser-Ney, built bottom-up
        t1 = self.tables[1]
        raw = order == 1
        cnt = t1.counts if raw else t1.kn_counts
        tot = float(t1.ctx_total[0] if raw else t1.kn_ctx_total[0])
        D = t1.disc if raw else t1.kn_disc
        j, f = t1.lookup(np.zeros(len(words), np.int64), True, words)
        c = np.where(f, cnt[j], 0)
        p = np.maximum(c - D, 0) / tot + D * len(t1) / tot / self.vocab_size
        for m in range(2, order + 1):
            t = self.tables[m]
            raw = m == order
            cnt = t.counts if raw else t.kn_counts
            ctot = t.ctx_total if raw else t.kn_ctx_total
            D = t.disc if raw else t.kn_disc
            ci, fc = t.lookup_ctx(pack(ctx2d[:, ctx2d.shape[1] - (m - 1):]))
            j, f = t.lookup(ci, fc, words)
            c = np.where(f, cnt[j], 0)
            denom = np.where(fc, ctot[ci], 1)
            types = np.where(fc, t.ctx_types[ci], 0)
            p = np.where(fc, (np.maximum(c - D, 0) + D * types * p) / denom, p)
        return p

    def logprobs(self, corpus: EncodedCorpus, order: int, method: str = "kn",
                 k: float = 1.0, batch: int = 2_000_000) -> np.ndarray:
        """Natural-log probability of every predicted token of ``corpus``."""
        mapped = corpus.map_ids(self.id_map)
        ctx, words = windows(mapped.ids, mapped.offsets, order)
        out = np.empty(len(words))
        for s in range(0, len(words), batch):
            with np.errstate(divide="ignore"):
                out[s:s + batch] = np.log(self.prob(ctx[s:s + batch], words[s:s + batch],
                                                    order, method, k))
        return out

    def perplexity(self, corpus: EncodedCorpus, order: int, method: str = "kn", k: float = 1.0):
        lp = self.logprobs(corpus, order, method, k)
        return float(np.exp(-lp.mean())), lp

    def sentence_logprob(self, sent_ids, order: int, method: str = "kn", k: float = 1.0):
        """Per-step probabilities of one sentence (ids already mapped)."""
        ctx, words = sentence_windows(sent_ids, order)
        return ctx, words, self.prob(ctx, words, order, method, k)

    def score_sentences(self, sentences: list, order: int, method: str = "kn") -> np.ndarray:
        """Total log10 probability of many sentences (lists of model ids)."""
        lens = np.array([len(s) + 1 for s in sentences])
        flat = np.concatenate([np.asarray(s, dtype=np.int32) for s in sentences])
        off = np.concatenate([[0], np.cumsum([len(s) for s in sentences])]).astype(np.int64)
        ctx, words = windows(flat, off, order)
        with np.errstate(divide="ignore"):
            lp = np.log10(self.prob(ctx, words, order, method))
        return np.add.reduceat(lp, np.concatenate([[0], np.cumsum(lens)[:-1]]))

    # ------------------------------------------------------------- generation
    def generate(self, order: int, rng: np.random.Generator, max_len: int = 25,
                 prefix=()) -> list[int]:
        """Sample a sentence from the MLE distribution of the given order."""
        t = self.tables[order]
        ctx = [BOS] * (order - 1) + list(prefix)
        out = list(prefix)
        while len(out) < max_len:
            words, counts = t.continuations(ctx[len(ctx) - (order - 1):] if order > 1 else [])
            if len(words) == 0:
                break
            cdf = np.cumsum(counts, dtype=np.int64)
            w = int(words[np.searchsorted(cdf, rng.integers(cdf[-1]), side="right")])
            if w == EOS:
                break
            out.append(w)
            ctx.append(w)
        return out

    # ------------------------------------------------------------- suggestions
    def _prefix_rank(self):
        if self._sorted_words is None:
            order = sorted(range(len(self.vocab.itos)), key=self.vocab.itos.__getitem__)
            rank = np.empty(len(order), dtype=np.int64)
            rank[order] = np.arange(len(order))
            self._sorted_words = ([self.vocab.itos[i] for i in order], rank)
        return self._sorted_words

    def prefix_range(self, prefix: str):
        words, _ = self._prefix_rank()
        lo = bisect.bisect_left(words, prefix)
        hi = bisect.bisect_left(words, prefix + "\U0010FFFF")
        return lo, hi

    def _unigrams_by_rank(self):
        """Unigram words / counts sorted alphabetically (prefix = one slice)."""
        if not hasattr(self, "_uni_lex"):
            _, rank = self._prefix_rank()
            t = self.tables[1]
            words = t.keys % t.V
            counts = t.kn_counts if t.kn_counts is not None else t.counts
            o = np.argsort(rank[words])
            self._uni_lex = (rank[words][o], words[o], np.asarray(counts)[o])
        return self._uni_lex

    def candidates(self, history, order: int, per_order: int = 300, prefix: str = ""):
        """Candidate next words: the most frequent continuations of every
        context suffix (order-1 words, then shorter), plus top unigrams."""
        hist = [BOS] * (order - 1) + list(history)
        rank = None
        if prefix:
            _, rank = self._prefix_rank()
            lo, hi = self.prefix_range(prefix)
        pools = []
        for m in range(order, 0, -1):
            t = self.tables[m]
            if m > 1:
                words, counts = t.continuations(hist[len(hist) - (m - 1):])
            elif prefix:
                r, w, c = self._unigrams_by_rank()
                a, b = np.searchsorted(r, lo), np.searchsorted(r, hi)
                words, counts = w[a:b], c[a:b]
            else:
                words, counts = t.keys % t.V, (t.kn_counts if t.kn_counts is not None else t.counts)
            if len(words) == 0:
                continue
            if prefix and m > 1:
                r = rank[words]
                keep = (r >= lo) & (r < hi)
                words, counts = words[keep], counts[keep]
            keep = ~np.isin(words, [BOS, UNK])
            words, counts = words[keep], counts[keep]
            if len(words) > per_order:
                top = np.argpartition(-counts, per_order)[:per_order]
                words = words[top]
            pools.append(words)
        if not pools:
            return np.empty(0, np.int64)
        return np.unique(np.concatenate(pools))

    def suggest(self, history, k: int = 5, order: int = 3, prefix: str = "",
                method: str = "kn", per_order: int = 300, details: bool = True):
        """Top-k next words after ``history`` (list of model ids), ranked by
        P(w | last order-1 words).  ``prefix`` restricts to words starting
        with the letters typed so far."""
        order = min(order, self.max_order)
        cand = self.candidates(history, order, per_order, prefix)
        if len(cand) == 0:
            return []
        hist = [BOS] * (order - 1) + list(history)
        ctx = np.tile(np.array(hist[len(hist) - (order - 1):] if order > 1 else [], np.int64),
                      (len(cand), 1)).reshape(len(cand), order - 1)
        p = self.prob(ctx, cand, order, method)
        best = np.argsort(-p, kind="stable")[:k]
        if not details:
            return [{"id": int(cand[i]), "p": float(p[i])} for i in best]
        out = []
        for i in best:
            w = int(cand[i])
            # which (longest) context was actually observed with this word?
            matched = 1
            for m in range(order, 1, -1):
                c, _ = self.tables[m].count(ctx[i:i + 1], np.array([w]))
                if c[0] > 0:
                    matched = m
                    break
            mle_c, mle_tot = self.tables[matched].count(ctx[i:i + 1], np.array([w]))
            out.append({"id": w, "word": self.vocab.display[w] if self.vocab.display else self.vocab.itos[w],
                        "norm": self.vocab.itos[w], "p": float(p[i]), "matched_order": matched,
                        "count": int(mle_c[0]), "p_mle": float(mle_c[0] / max(mle_tot[0], 1))})
        return out
