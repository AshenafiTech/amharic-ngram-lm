"""Encode the GPAC corpus into integer arrays (with on-disk caching).

An encoded corpus is two numpy arrays:

* ``ids``     int32[N]   word ids of every token, sentences concatenated
* ``offsets`` int64[S+1] sentence i is ``ids[offsets[i]:offsets[i+1]]``

plus a :class:`Vocab`.  Ids 0, 1, 2 are reserved for ``<s>``, ``</s>`` and
``<UNK>``; real words follow in order of decreasing frequency.
"""
from __future__ import annotations

import json
import os
import time
from array import array
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .preprocess import TOKEN_OR_EOS_RE, SENTENCE_END, normalize_token, repair

BOS, EOS, UNK = 0, 1, 2
SPECIALS = ["<s>", "</s>", "<UNK>"]


# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------
@dataclass
class Vocab:
    itos: list[str]                       # id -> normalised word
    counts: np.ndarray                    # id -> corpus frequency
    display: list[str] | None = None      # id -> most frequent raw spelling
    _stoi: dict[str, int] | None = field(default=None, repr=False)

    def __post_init__(self):
        # the "most frequent raw spelling" of e.g. በ<NUM> would be one
        # particular number (በ2010) -- keep the normalised form for those
        if self.display:
            self.display = [w if "<NUM>" in w else d for w, d in zip(self.itos, self.display)]

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def stoi(self) -> dict[str, int]:
        if self._stoi is None:
            self._stoi = {w: i for i, w in enumerate(self.itos)}
        return self._stoi

    def encode(self, tokens, unk: int = UNK) -> np.ndarray:
        s = self.stoi
        return np.array([s.get(t, unk) for t in tokens], dtype=np.int32)

    def decode(self, ids, display: bool = False) -> list[str]:
        table = self.display if (display and self.display) else self.itos
        return [table[i] for i in ids]

    def save(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for i, w in enumerate(self.itos):
                d = self.display[i] if self.display else w
                f.write(f"{w}\t{int(self.counts[i])}\t{d}\n")

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        itos, counts, display = [], [], []
        with open(path, encoding="utf-8") as f:
            for line in f:
                w, c, d = line.rstrip("\n").split("\t")
                itos.append(w)
                counts.append(int(c))
                display.append(d)
        return cls(itos, np.array(counts, dtype=np.int64), display)


# --------------------------------------------------------------------------
# encoded corpus
# --------------------------------------------------------------------------
@dataclass
class EncodedCorpus:
    ids: np.ndarray
    offsets: np.ndarray
    vocab: Vocab
    info: dict = field(default_factory=dict)

    @property
    def n_sentences(self) -> int:
        return len(self.offsets) - 1

    @property
    def n_tokens(self) -> int:
        return len(self.ids)

    def sentence_lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    def sentence(self, i: int, display: bool = False) -> list[str]:
        return self.vocab.decode(self.ids[self.offsets[i]:self.offsets[i + 1]], display)

    def subset(self, sent_idx: np.ndarray) -> "EncodedCorpus":
        """New corpus made of the given sentences (in the given order)."""
        sent_idx = np.asarray(sent_idx)
        starts, ends = self.offsets[sent_idx], self.offsets[sent_idx + 1]
        lens = ends - starts
        new_off = np.zeros(len(sent_idx) + 1, dtype=np.int64)
        np.cumsum(lens, out=new_off[1:])
        # gather token positions without a python loop
        pos = np.repeat(starts - new_off[:-1], lens) + np.arange(new_off[-1])
        return EncodedCorpus(self.ids[pos], new_off, self.vocab)

    def filter_tokens(self, drop_mask_by_id: np.ndarray) -> "EncodedCorpus":
        """Remove tokens whose id is flagged in ``drop_mask_by_id`` (e.g. stop
        words); sentences that become empty are dropped."""
        keep = ~drop_mask_by_id[self.ids]
        sid = np.repeat(np.arange(self.n_sentences), self.sentence_lengths())[keep]
        lens = np.bincount(sid, minlength=self.n_sentences)
        lens = lens[lens > 0]
        off = np.zeros(len(lens) + 1, dtype=np.int64)
        np.cumsum(lens, out=off[1:])
        return EncodedCorpus(self.ids[keep], off, self.vocab)

    def map_ids(self, id_map: np.ndarray) -> "EncodedCorpus":
        return EncodedCorpus(id_map[self.ids].astype(np.int32), self.offsets, self.vocab)

    # ------------------------------------------------------------- caching
    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "ids.npy", self.ids)
        np.save(directory / "offsets.npy", self.offsets)
        self.vocab.save(directory / "vocab.tsv")
        with open(directory / "info.json", "w", encoding="utf-8") as f:
            json.dump(self.info, f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, directory: Path, mmap: bool = False) -> "EncodedCorpus":
        directory = Path(directory)
        mode = "r" if mmap else None
        info = {}
        if (directory / "info.json").exists():
            info = json.loads((directory / "info.json").read_text(encoding="utf-8"))
        return cls(np.load(directory / "ids.npy", mmap_mode=mode),
                   np.load(directory / "offsets.npy", mmap_mode=mode),
                   Vocab.load(directory / "vocab.tsv"), info)


class _Indexer(dict):
    """dict that hands out consecutive ids to unseen keys."""

    def __init__(self, preset):
        super().__init__(preset)
        self.next_id = 0

    def __missing__(self, key):
        v = self[key] = self.next_id
        self.next_id += 1
        return v


def encode_text(text: str, chunk_chars: int = 20_000_000, verbose: bool = True) -> EncodedCorpus:
    """Sentence-split, tokenise and normalise ``text`` into an EncodedCorpus."""
    t0 = time.time()
    index = _Indexer({m: -1 for m in SENTENCE_END})   # terminators -> -1
    stream = array("i")
    start, L = 0, len(text)
    while start < L:
        end = min(start + chunk_chars, L)
        if end < L:                         # never cut through a token
            sp = text.find(" ", end)
            end = L if sp == -1 else sp
        stream.extend(map(index.__getitem__, TOKEN_OR_EOS_RE.findall(repair(text[start:end]))))
        start = end
        if verbose:
            print(f"\r  tokenised {end / L:6.1%}  raw types={index.next_id:,}  "
                  f"({time.time() - t0:.0f}s)", end="", flush=True)
    if verbose:
        print()

    rid = np.frombuffer(stream, dtype=np.int32)
    raw_types = [None] * index.next_id
    for tok, i in index.items():
        if i >= 0:
            raw_types[i] = tok
    del index

    is_eos = rid < 0
    raw_ids = rid[~is_eos]
    raw_counts = np.bincount(raw_ids, minlength=len(raw_types)).astype(np.int64)

    # ---- normalise raw types -> normalised types
    norm_index: dict[str, int] = {}
    rid2nid = np.empty(len(raw_types), dtype=np.int64)
    for i, tok in enumerate(raw_types):
        rid2nid[i] = norm_index.setdefault(normalize_token(tok), len(norm_index))
    norm_types = [None] * len(norm_index)
    for w, i in norm_index.items():
        norm_types[i] = w
    norm_counts = np.bincount(rid2nid, weights=raw_counts, minlength=len(norm_types)).astype(np.int64)

    # most frequent raw spelling of every normalised type (for display)
    order = np.lexsort((-raw_counts, rid2nid))
    first = np.ones(len(order), dtype=bool)
    first[1:] = rid2nid[order][1:] != rid2nid[order][:-1]
    best_raw = order[first]                     # indexed by normalised id

    # ---- re-number: specials first, then by decreasing frequency
    freq_order = np.argsort(-norm_counts, kind="stable")
    nid2final = np.empty(len(norm_types), dtype=np.int64)
    nid2final[freq_order] = np.arange(len(norm_types)) + len(SPECIALS)
    itos = SPECIALS + [norm_types[i] for i in freq_order]
    display = SPECIALS + [raw_types[best_raw[i]] for i in freq_order]

    ids = nid2final[rid2nid[raw_ids]].astype(np.int32)

    # ---- sentence offsets (empty sentences vanish automatically)
    sid = np.cumsum(is_eos)[~is_eos]
    cuts = np.flatnonzero(np.diff(sid)) + 1
    offsets = np.concatenate([[0], cuts, [len(ids)]]).astype(np.int64)

    counts = np.zeros(len(itos), dtype=np.int64)
    counts[len(SPECIALS):] = norm_counts[freq_order]
    counts[BOS] = counts[EOS] = len(offsets) - 1

    info = {
        "raw_types": len(raw_types),
        "normalised_types": len(norm_types),
        "raw_terminators": int(is_eos.sum()),
        "encode_seconds": round(time.time() - t0, 1),
    }
    return EncodedCorpus(ids, offsets, Vocab(itos, counts, display), info)


def load_or_encode(corpus_path: str | os.PathLike, cache_dir: str | os.PathLike,
                   verbose: bool = True) -> EncodedCorpus:
    cache_dir = Path(cache_dir)
    if (cache_dir / "ids.npy").exists():
        if verbose:
            print(f"loading cached encoded corpus from {cache_dir}")
        return EncodedCorpus.load(cache_dir)
    if verbose:
        print(f"reading {corpus_path} ...")
    with open(corpus_path, encoding="utf-8") as f:
        text = f.read()
    corpus = encode_text(text, verbose=verbose)
    corpus.info["raw_chars"] = len(text)
    corpus.info["raw_bytes"] = os.path.getsize(corpus_path)
    corpus.save(cache_dir)
    return corpus


# --------------------------------------------------------------------------
# duplicate sentences
# --------------------------------------------------------------------------
def sentence_hashes(corpus: EncodedCorpus) -> np.ndarray:
    """64-bit polynomial hash of every sentence (length mixed in)."""
    lens = corpus.sentence_lengths()
    sid = np.repeat(np.arange(corpus.n_sentences), lens)
    pos = np.arange(corpus.n_tokens) - np.asarray(corpus.offsets)[sid]
    with np.errstate(over="ignore"):
        powers = np.cumprod(np.full(int(lens.max()) + 1, 1_000_003, dtype=np.uint64))
        h = np.add.reduceat((corpus.ids.astype(np.uint64) + np.uint64(7)) * powers[pos],
                            np.asarray(corpus.offsets[:-1]))
        h ^= lens.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
    return h


def deduplicate(corpus: EncodedCorpus, min_len: int = 5) -> tuple[EncodedCorpus, dict]:
    """Drop repeated sentences with >= min_len tokens (keep first occurrence).
    Short sentences ("ለምን", "አዎ" ...) repeat naturally and are kept."""
    h = sentence_hashes(corpus)
    _, first = np.unique(h, return_index=True)
    is_first = np.zeros(corpus.n_sentences, dtype=bool)
    is_first[first] = True
    keep = is_first | (corpus.sentence_lengths() < min_len)
    out = corpus.subset(np.flatnonzero(keep))
    counts = np.bincount(out.ids, minlength=len(corpus.vocab)).astype(np.int64)
    counts[BOS] = counts[EOS] = out.n_sentences
    out.vocab = Vocab(corpus.vocab.itos, counts, corpus.vocab.display)
    out.info = dict(corpus.info)
    stats = {"duplicate_sentences_all": int(corpus.n_sentences - len(first)),
             "removed_sentences": int((~keep).sum()),
             "removed_tokens": int(corpus.n_tokens - out.n_tokens)}
    out.info.update(stats)
    return out, stats


# --------------------------------------------------------------------------
# train / dev / test split
# --------------------------------------------------------------------------
def block_split(n_sentences: int, fractions=(0.90, 0.05, 0.05), block: int = 200,
                seed: int = 42) -> tuple[np.ndarray, ...]:
    """Split sentence indices into train/dev/test by *blocks* of consecutive
    sentences.  The corpus is a concatenation of articles, so keeping
    neighbouring sentences together approximates a document-level split and
    limits leakage of near-duplicate sentences from the same article."""
    n_blocks = (n_sentences + block - 1) // block
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_blocks)
    cuts = np.cumsum(np.array(fractions[:-1]) * n_blocks).astype(int)
    parts = np.split(perm, cuts)
    out = []
    for blocks in parts:
        blocks = np.sort(blocks)
        idx = (blocks[:, None] * block + np.arange(block)[None, :]).ravel()
        out.append(idx[idx < n_sentences])
    return tuple(out)
