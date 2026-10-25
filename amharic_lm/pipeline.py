"""Cached build steps shared by the notebook, the build script and the app."""
from __future__ import annotations

import json

import numpy as np

from . import paths
from .corpus import EncodedCorpus, block_split, deduplicate, load_or_encode
from .ngram import NGramModel


def download_corpus(verbose: bool = True):
    if paths.CORPUS_TXT.exists():
        return paths.CORPUS_TXT
    import gdown
    paths.DATA.mkdir(parents=True, exist_ok=True)
    gdown.download(id=paths.GDRIVE_ID, output=str(paths.CORPUS_TXT), quiet=not verbose)
    return paths.CORPUS_TXT


def raw_corpus(verbose: bool = True) -> EncodedCorpus:
    if not (paths.CORPUS_RAW / "ids.npy").exists():
        download_corpus(verbose)
    return load_or_encode(paths.CORPUS_TXT, paths.CORPUS_RAW, verbose)


def clean_corpus(verbose: bool = True) -> EncodedCorpus:
    if (paths.CORPUS_CLEAN / "ids.npy").exists():
        return EncodedCorpus.load(paths.CORPUS_CLEAN)
    clean, stats = deduplicate(raw_corpus(verbose))
    if verbose:
        print("de-duplication:", stats)
    clean.save(paths.CORPUS_CLEAN)
    return clean


def splits(corpus: EncodedCorpus | None = None):
    """Sentence indices (into the clean corpus) of train / dev / test."""
    if paths.SPLITS.exists():
        z = np.load(paths.SPLITS)
        return z["train"], z["dev"], z["test"]
    corpus = corpus or clean_corpus()
    train, dev, test = block_split(corpus.n_sentences)
    paths.SPLITS.parent.mkdir(parents=True, exist_ok=True)
    np.savez(paths.SPLITS, train=train, dev=dev, test=test)
    return train, dev, test


def split_corpora(corpus: EncodedCorpus | None = None):
    corpus = corpus or clean_corpus()
    tr, dv, te = splits(corpus)
    return corpus.subset(tr), corpus.subset(dv), corpus.subset(te)


def full_model(verbose: bool = True, mmap: bool = False, max_order: int | None = None) -> NGramModel:
    """Orders 1-4 trained on the whole clean corpus (no <UNK>)."""
    if (paths.MODEL_FULL / "model.json").exists():
        return NGramModel.load(paths.MODEL_FULL, mmap=mmap, max_order=max_order)
    model = NGramModel.train(clean_corpus(verbose), 4, min_count=1, verbose=verbose)
    model.save(paths.MODEL_FULL)
    return model


def train_model(verbose: bool = True, mmap: bool = False) -> NGramModel:
    """Orders 1-4 trained on the 90% training split; words seen once -> <UNK>."""
    if (paths.MODEL_TRAIN / "model.json").exists():
        return NGramModel.load(paths.MODEL_TRAIN, mmap=mmap)
    train, _, _ = split_corpora()
    model = NGramModel.train(train, 4, min_count=2, verbose=verbose)
    model.save(paths.MODEL_TRAIN)
    return model


def save_json(obj, name: str):
    paths.TABLES.mkdir(parents=True, exist_ok=True)
    (paths.TABLES / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
