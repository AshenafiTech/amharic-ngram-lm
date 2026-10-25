"""Amharic stop words.

A hand-curated list of high-frequency function words (copulas and auxiliaries,
pronouns, demonstratives, conjunctions, pre/postpositions, discourse adverbs,
quantifiers and reporting verbs of "saying"), checked against the most
frequent types of GPAC.  Spellings are given *after* normalisation (see
``preprocess.normalize_token``), e.g. ሆኖ not ሆኖ/ኾኖ, አሁን not ዓሁን.

In addition, ``is_stopword`` removes
* single-letter tokens -- bare prefixes (የ, በ, ከ, ለ) and fragments left by
  deleted punctuation in the corpus dump;
* number tokens (<NUM>, በ<NUM>, የ<NUM>ኛ ...).
"""
from __future__ import annotations

import numpy as np

from .preprocess import normalize_token

_RAW = """
ነው ናቸው ነበር ነበሩ ነኝ ነህ ነሽ ነን ናችሁ ናት ነዉ ነበረ ነበረች ነበርኩ ነበረው ነበራቸው
አለ አሉ አለች አለው አላቸው አለኝ አለን አለበት አለባቸው ያለ ያሉ ያለው ያሉት ያለውን ያላቸው ያለችው
የለም የላቸውም የለውም አይደለም አይደሉም አልነበረም አልነበሩም ኖሮ ኖረ
ሆነ ሆኖ ሆነው ሆኖም ሆና ይሆናል ይሆን ይሁን ሊሆን መሆን መሆኑ መሆኑን መሆናቸውን
የሆነ የሆነው የሆኑ የሆኑት በሆነ ከሆነ ቢሆን ቢሆንም እንዲሆን እንደሆነ ሲሆን ሳይሆን
በመሆኑ በመሆኑም ስለሆነ እንደነበር የነበረው የነበሩ የነበረ የነበረችው
ይችላል ይቻላል ይገባል ይገባዋል ነበረበት
እና እንዲሁም ወይም ወይ ግን ደግሞ ደግሞም እንጂ ስለዚህ ስለዚሁ ምክንያቱም ታዲያ ሆኖም
እንደ እንደዚህ እንደዚሁ እንዲህ እንዲሁ እንዲያ ልክ
ላይ ውስጥ ወደ ጋር ስለ እስከ ድረስ ከ የ በ ለ ውጭ ውጪ ስር በኩል መካከል ዘንድ በላይ በታች
በፊት በኋላ ጀምሮ ጭምር ጨምሮ ዙሪያ ይልቅ አንጻር
ይህ ይህን ይህንን ይህም ይሄ ያ ያን ያንን ያም እነዚህ እነዚያ እነዚህን ይኸው ይኼ
በዚህ በዚህም ከዚህ ከዚያ ከዚያም ለዚህ ለዚያ በዚያ እዚህ እዚያ እዚሁ እዛ
እኔ አንተ አንቺ እሱ እሷ እሳቸው እርሱ እርሷ እኛ እናንተ እነሱ እነርሱ እራሱ ራሱ ራሳቸው
የኔ የእኔ የእሱ የእሷ የእኛ የነሱ የእነሱ ለእኔ ለእኛ እኮ
ምን ማን የት መቼ እንዴት ለምን ስንት የትኛው ምንም ማንም
ሁሉ ሁሉም ሁሉንም በሙሉ ብዙ አንዳንድ ሌላ ሌሎች ሌላው ሌሎችም አንድ አንዱ ጥቂት በጣም እጅግ
ብቻ ገና አሁን አሁንም ያኔ ደግሞ እንኳ እንኳን ቢያንስ በተለይ በተለይም በተጨማሪ በተጨማሪም ያህል
ብሎ ብለው ብላ ብለዋል ብሏል ብለን ብዬ ይላል ይላሉ ይል ሲል ሲሉ በማለት ማለት የሚል የሚለው የሚለውን በሚል ተብሎ
እያለ እያሉ ማድረግ አድርጎ
ዓ/ም እ/ኤ/አ ዓ/ዓ ወዘተ ወ/ዘ/ተ
"""

STOPWORDS: frozenset[str] = frozenset(normalize_token(w) for w in _RAW.split())


def is_stopword(token: str) -> bool:
    return token in STOPWORDS or len(token) == 1 or "<NUM>" in token


def stopword_mask(itos: list[str]) -> np.ndarray:
    """Boolean array over the vocabulary: True where the id is a stop word.
    Special tokens (<s>, </s>, <UNK>) are not flagged."""
    mask = np.fromiter((is_stopword(w) for w in itos), dtype=bool, count=len(itos))
    mask[:3] = False
    return mask
