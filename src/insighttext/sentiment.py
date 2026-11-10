"""Lexicon-based sentiment analysis.

Scores text by looking up each word against small, built-in positive and
negative word lexicons and tallying the hits, with a simple windowed
negation rule (a negation word flips the polarity of sentiment words that
follow it within a few tokens, e.g. "not good" counts as negative). This
is a rule-based approach rather than a trained classifier: transparent,
fully offline, and good enough for short, informal text such as reviews
or comments.
"""

from __future__ import annotations

from dataclasses import dataclass

from insighttext.preprocessing import tokenize_words

# How many tokens after a negation word still have their polarity flipped.
# "not a particularly good film" -> "not"(0) "a"(1) "particularly"(2)
# "good"(3, still within window) "film"(4).
NEGATION_WINDOW = 3

# A score strictly above this counts as "positive"; strictly below its
# negation counts as "negative"; everything in between is "neutral".
SENTIMENT_THRESHOLD = 0.05

POSITIVE_WORDS: frozenset[str] = frozenset(
    """
    good great excellent amazing wonderful fantastic awesome love loved
    loves lovely best happy beautiful brilliant perfect nice enjoy enjoyed
    enjoyable positive impressive impressed superb outstanding delightful
    pleasant satisfying recommend recommended favorite favourite terrific
    charming comfortable elegant exciting fun funny generous gorgeous
    helpful honest incredible remarkable smooth successful talented
    thrilled warm win winning worth friendly glad grateful proud
    """.split()
)

NEGATIVE_WORDS: frozenset[str] = frozenset(
    """
    bad terrible horrible awful worst hate hated hates poor disappointing
    disappointed disappoints sad boring bored ugly broken annoying angry
    cheap confusing difficult dirty dull fail failed failure fake flawed
    frustrating inferior lousy mediocre messy mistake negative offensive
    painful pathetic pointless ridiculous rude sloppy slow stupid tragic
    trouble unacceptable uncomfortable unfortunate unhappy useless waste
    weak wrong disgusting
    """.split()
)

NEGATION_WORDS: frozenset[str] = frozenset(
    """
    not no never none nobody nothing nowhere neither without lack lacking
    hardly barely rarely cannot can't don't didn't doesn't isn't wasn't
    aren't weren't won't wouldn't couldn't shouldn't hasn't haven't hadn't
    """.split()
)


@dataclass(frozen=True)
class SentimentResult:
    """The outcome of a sentiment analysis.

    Attributes:
        label: One of "positive", "negative", or "neutral".
        score: A polarity score in [-1.0, 1.0]; positive values lean
            positive, negative values lean negative, 0.0 means either no
            sentiment-bearing words were found or they canceled out.
        positive_words: Distinct lexicon words that contributed positively
            to the score (after negation handling), sorted alphabetically.
        negative_words: Distinct lexicon words that contributed negatively
            to the score (after negation handling), sorted alphabetically.
    """

    label: str
    score: float
    positive_words: tuple[str, ...]
    negative_words: tuple[str, ...]


def analyze_sentiment(text: str) -> SentimentResult:
    """Scores the sentiment of `text`.

    Args:
        text: The text to analyze.

    Returns:
        A `SentimentResult` describing the overall sentiment. Text with no
        recognized sentiment words, or whose positive and negative hits
        exactly cancel out, is labeled "neutral" with a score of 0.0.
    """
    words = tokenize_words(text)

    positive_hits: list[str] = []
    negative_hits: list[str] = []
    negated_remaining = 0

    for word in words:
        if word in NEGATION_WORDS:
            negated_remaining = NEGATION_WINDOW
            continue

        is_positive = word in POSITIVE_WORDS
        is_negative = word in NEGATIVE_WORDS
        if is_positive or is_negative:
            flipped = negated_remaining > 0
            target_is_positive = is_negative if flipped else is_positive
            (positive_hits if target_is_positive else negative_hits).append(word)

        if negated_remaining > 0:
            negated_remaining -= 1

    total_hits = len(positive_hits) + len(negative_hits)
    if total_hits == 0:
        return SentimentResult(label="neutral", score=0.0, positive_words=(), negative_words=())

    score = round((len(positive_hits) - len(negative_hits)) / total_hits, 4)
    if score > SENTIMENT_THRESHOLD:
        label = "positive"
    elif score < -SENTIMENT_THRESHOLD:
        label = "negative"
    else:
        label = "neutral"

    return SentimentResult(
        label=label,
        score=score,
        positive_words=tuple(sorted(set(positive_hits))),
        negative_words=tuple(sorted(set(negative_hits))),
    )
