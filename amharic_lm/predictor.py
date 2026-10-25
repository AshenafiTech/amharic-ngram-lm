"""Task 4 -- Amharic predictive text / next-word suggestion engine.

Shared by the web app (app/server.py), the terminal app (app/cli.py) and the
notebook.  Pipeline for every request:

1. read the user's text and keep only the sentence being written
   (text after the last ። ? ! ...);
2. tokenise + normalise it exactly like the training corpus;
3. if the text does not end with a space, the last token is a word being
   typed -> it becomes a *prefix* filter instead of context;
4. take the final one or two words as context (order 3 = trigram; the order
   can be changed) and retrieve candidate next words from the n-gram tables,
   backing off to shorter contexts;
5. score every candidate with interpolated Kneser-Ney P(w | context);
6. return the top five; the user picks one and keeps composing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .corpus import EOS, UNK
from .ngram import NGramModel
from .preprocess import ETH_LETTERS, SENTENCE_SPLIT_RE, normalize_token, tokenize

ORDER_NAMES = {1: "unigram", 2: "bigram", 3: "trigram", 4: "4-gram"}
_PARTIAL_RE = re.compile(rf"[{ETH_LETTERS}/]$")
END_MARK = "።"


@dataclass
class Parsed:
    history: list[str]       # normalised words of the current sentence
    prefix: str              # partially typed word ('' if none)
    raw_prefix: str          # the partial word as typed


class NextWordPredictor:
    def __init__(self, model: NGramModel, order: int = 3):
        self.model = model
        self.order = order

    # ------------------------------------------------------------------ parse
    def parse(self, text: str) -> Parsed:
        current = SENTENCE_SPLIT_RE.split(text)[-1]
        tokens = tokenize(current, normalize=False)
        if tokens and _PARTIAL_RE.search(current):
            return Parsed([normalize_token(t) for t in tokens[:-1]],
                          normalize_token(tokens[-1]), tokens[-1])
        return Parsed([normalize_token(t) for t in tokens], "", "")

    # ---------------------------------------------------------------- suggest
    def suggest(self, text: str, k: int = 5, order: int | None = None) -> dict:
        order = order or self.order
        parsed = self.parse(text)
        ids = [int(i) for i in self.model.encode(parsed.history)]
        raw = self.model.suggest(ids, k=k + 5, order=order, prefix=parsed.prefix)
        sugg = []
        for s in raw:
            if "<NUM>" in s["norm"] or s["id"] == UNK:
                continue
            if s["id"] == EOS:
                if parsed.prefix:
                    continue
                s = {**s, "word": END_MARK, "norm": END_MARK}
            s["matched"] = ORDER_NAMES[s["matched_order"]]
            sugg.append(s)
            if len(sugg) == k:
                break
        n_ctx = order - 1
        context = ["<s>"] * max(0, n_ctx - len(parsed.history)) + parsed.history[-n_ctx:] if n_ctx else []
        return {
            "history": parsed.history,
            "prefix": parsed.raw_prefix,
            "context": context,
            "unknown_context": [w for w, i in zip(parsed.history[-n_ctx:] if n_ctx else [],
                                                  ids[-n_ctx:] if n_ctx else []) if i == UNK],
            "order": order,
            "suggestions": sugg,
        }

    # ------------------------------------------------------------------ apply
    def apply(self, text: str, word: str) -> str:
        """Insert a chosen suggestion (replacing a half-typed word)."""
        parsed = self.parse(text)
        if parsed.raw_prefix:
            text = text[: len(text) - len(parsed.raw_prefix)]
        elif text and not text[-1].isspace():
            text += " "
        if word == END_MARK:
            return text.rstrip() + END_MARK + " "
        return text + word + " "
