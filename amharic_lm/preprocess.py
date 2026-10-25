"""Amharic text normalisation, sentence splitting and tokenisation.

The GPAC dump is a single 1 GB line with no newlines.  It contains only
Ethiopic script, digits, spaces and a handful of punctuation marks, so the
pipeline is:

0. repair: the dump deleted "." "," "-", so abbreviations and numbers fell
   apart (ዓ.ም -> "ዓ ም", 8.4 -> "8 4"); they are glued back (ዓ/ም, 84);
1. sentence splitting on  ።  ፡፡  ::  ?  !  ፧  ፨
2. tokenisation: a token is a run of Ethiopic letters/digits, optionally joined
   by "/" (Amharic abbreviations such as ት/ቤት, ዓ/ም, ጠ/ሚ).  Every other symbol
   (፣ ፤ ፥ ፦ ( ) % ...) is treated as a separator and dropped.
3. token normalisation:
   * homophone letters are merged (ሐ/ኀ -> ሀ, ሠ -> ሰ, ዐ -> አ, ፀ -> ጸ, ዉ -> ው,
     and the 4th-order ሃ/ኣ that are pronounced like ሀ/አ);
   * every run of digits (Arabic or Ethiopic numerals) becomes <NUM>,
     e.g. "2010" -> "<NUM>", "ለ19ኛ" -> "ለ<NUM>ኛ".
"""
from __future__ import annotations

import re

# --------------------------------------------------------------------------
# character classes
# --------------------------------------------------------------------------
# Ethiopic letters (syllables) in the main, supplement, extended and
# extended-A blocks; punctuation (U+1360-1368) and numerals (U+1369-137C)
# are excluded on purpose.
ETH_LETTERS = "ሀ-ፚᎀ-ᎏⶀ-ⷞꬁ-ꬮ"
ETH_NUMERALS = "፩-፼"
DIGITS = "0-9" + ETH_NUMERALS

_TOKEN_BODY = f"[{ETH_LETTERS}{DIGITS}]+"
# a token, or a sentence terminator
TOKEN_OR_EOS_RE = re.compile(
    rf"{_TOKEN_BODY}(?:/{_TOKEN_BODY})*|።|፡፡|::|[?!፧፨]"
)
TOKEN_RE = re.compile(rf"{_TOKEN_BODY}(?:/{_TOKEN_BODY})*")
SENTENCE_END = frozenset({"።", "፡፡", "::", "?", "!", "፧", "፨"})
SENTENCE_SPLIT_RE = re.compile(r"።|፡፡|::|[?!፧፨]")
DIGIT_RUN_RE = re.compile(f"[{DIGITS}]+")

NUM_TOKEN = "<NUM>"

# The corpus dump deleted "." and "," and "-", which split abbreviations and
# numbers into pieces:  ዓ.ም. -> "ዓ ም",  እ.ኤ.አ -> "እ ኤ አ",  8.4 -> "8 4",
# 2004-2007 -> "2004 2007".  ``repair`` glues them back together.
_ABBREV_RE = re.compile(
    rf"(?<![{ETH_LETTERS}{DIGITS}/])[{ETH_LETTERS}](?: +[{ETH_LETTERS}])+(?![{ETH_LETTERS}{DIGITS}/])")
_NUM_SEQ_RE = re.compile(rf"(?<![{DIGITS}])[{DIGITS}]+(?: +[{DIGITS}]+)+(?![{DIGITS}])")
_SPACES_RE = re.compile(r" +")


def repair(text: str) -> str:
    """Re-join abbreviations (ዓ ም -> ዓ/ም) and split numbers (8 4 -> 84)."""
    text = _ABBREV_RE.sub(lambda m: _SPACES_RE.sub("/", m.group(0)), text)
    return _NUM_SEQ_RE.sub(lambda m: _SPACES_RE.sub("", m.group(0)), text)

# --------------------------------------------------------------------------
# homophone normalisation
# --------------------------------------------------------------------------
def _series(src: int, dst: int, orders: int = 7) -> dict[int, int]:
    """Map the 7 vowel orders of one consonant series onto another."""
    return {src + i: dst + i for i in range(orders)}


_CHAR_MAP: dict[int, int] = {}
_CHAR_MAP.update(_series(0x1210, 0x1200))      # ሐ ሑ ሒ ሓ ሔ ሕ ሖ -> ሀ ሁ ሂ ሃ ሄ ህ ሆ
_CHAR_MAP.update(_series(0x1280, 0x1200))      # ኀ ኁ ኂ ኃ ኄ ኅ ኆ -> ሀ ...
_CHAR_MAP.update(_series(0x1220, 0x1230, 8))   # ሠ ... ሧ -> ሰ ... ሷ
_CHAR_MAP.update(_series(0x12D0, 0x12A0))      # ዐ ዑ ዒ ዓ ዔ ዕ ዖ -> አ ኡ ኢ ኣ ኤ እ ኦ
_CHAR_MAP.update(_series(0x1340, 0x1338))      # ፀ ፁ ፂ ፃ ፄ ፅ ፆ -> ጸ ጹ ጺ ጻ ጼ ጽ ጾ
# 4th-order "a" forms that are pronounced exactly like the 1st order
for _src in (0x1203, 0x1213, 0x1283):          # ሃ ሓ ኃ -> ሀ
    _CHAR_MAP[_src] = 0x1200
for _src in (0x12A3, 0x12D3):                  # ኣ ዓ -> አ
    _CHAR_MAP[_src] = 0x12A0
_CHAR_MAP[0x12C9] = 0x12CD                     # ዉ -> ው
_CHAR_MAP[0x1217] = 0x1207                     # ሗ -> ሇ (labialised ha)

_TRANSLATION = str.maketrans(_CHAR_MAP)


def normalize_chars(text: str) -> str:
    """Merge Amharic homophone letters (does not touch anything else)."""
    return text.translate(_TRANSLATION)


def normalize_token(token: str) -> str:
    """Normalise one raw token: homophones + digit runs -> <NUM>."""
    token = token.translate(_TRANSLATION)
    if DIGIT_RUN_RE.search(token):
        token = DIGIT_RUN_RE.sub(NUM_TOKEN, token)
    return token


# --------------------------------------------------------------------------
# user-facing helpers (used by the notebook, evaluation and the app)
# --------------------------------------------------------------------------
def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


def tokenize(text: str, normalize: bool = True) -> list[str]:
    """Tokenise a piece of text (sentence terminators are dropped)."""
    toks = TOKEN_RE.findall(repair(text))
    return [normalize_token(t) for t in toks] if normalize else toks


def tokenize_sentences(text: str, normalize: bool = True) -> list[list[str]]:
    out = []
    for sent in split_sentences(text):
        toks = tokenize(sent, normalize)
        if toks:
            out.append(toks)
    return out
