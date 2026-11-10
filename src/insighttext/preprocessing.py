"""Lightweight, dependency-free text preprocessing utilities.

This module intentionally avoids heavyweight NLP libraries (spaCy, NLTK with
downloaded corpora, etc.) so the rest of the package stays fast, offline,
and deterministic. Sentence splitting uses a regex heuristic rather than a
statistical model, which is sufficient for the extractive techniques this
project builds on top of it.
"""

from __future__ import annotations

import re

# A standard, general-purpose English stopword list. Kept as a module
# constant (rather than loaded from a file or a third-party corpus) so the
# package has no runtime dependency on external data.
STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all am an and any are aren't as at be
    because been before being below between both but by can't cannot could
    couldn't did didn't do does doesn't doing don't down during each few for
    from further had hadn't has hasn't have haven't having he he'd he'll
    he's her here here's hers herself him himself his how how's i i'd i'll
    i'm i've if in into is isn't it it's its itself let's me more most
    mustn't my myself no nor not of off on once only or other ought our
    ours ourselves out over own same shan't she she'd she'll she's should
    shouldn't so some such than that that's the their theirs them themselves
    then there there's these they they'd they'll they're they've this those
    through to too under until up very was wasn't we we'd we'll we're we've
    were weren't what what's when when's where where's which while who
    who's whom why why's with won't would wouldn't you you'd you'll you're
    you've your yours yourself yourselves
    """.split()
)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
_WORD_PATTERN = re.compile(r"[A-Za-z']+")


def split_sentences(text: str) -> list[str]:
    """Splits raw text into sentences.

    Uses a regex heuristic that breaks on whitespace following a sentence-
    ending punctuation mark (., !, or ?) when followed by a capitalized word
    or digit. This is not a full sentence boundary disambiguation model, but
    it is dependency-free and good enough for extractive summarization.

    Args:
        text: The raw input text, potentially spanning multiple paragraphs.

    Returns:
        A list of non-empty sentence strings, in their original order, with
        surrounding whitespace stripped.
    """
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    sentences = _SENTENCE_BOUNDARY.split(normalized)
    return [s.strip() for s in sentences if s.strip()]


def tokenize_words(text: str) -> list[str]:
    """Extracts lowercase word tokens from text.

    Args:
        text: The input text to tokenize.

    Returns:
        A list of lowercase alphabetic tokens (apostrophes are kept so that
        contractions like "don't" remain a single token).
    """
    return [match.group(0).lower() for match in _WORD_PATTERN.finditer(text)]


def remove_stopwords(words: list[str]) -> list[str]:
    """Filters stopwords out of a list of word tokens.

    Args:
        words: A list of lowercase word tokens, e.g. from `tokenize_words`.

    Returns:
        The input list with any token present in `STOPWORDS` removed,
        preserving the original order.
    """
    return [word for word in words if word not in STOPWORDS]
