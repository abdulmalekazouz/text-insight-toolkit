"""TF-IDF keyword extraction.

Treats each sentence in a document as a pseudo-document and ranks terms by
their *summed* TF-IDF weight across those sentences, with document-length
normalization disabled (`norm=None`). Summing (rather than, say, taking
the max) rewards terms that recur across several sentences -- a signal
that the term is a recurring topic rather than a one-off word -- while
still discounting terms that appear in every sentence and therefore carry
little discriminative weight.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

from insighttext.preprocessing import STOPWORDS, split_sentences


def extract_keywords(text: str, num_keywords: int = 5) -> list[str]:
    """Extracts the top-ranked keywords from `text`.

    Args:
        text: The document to extract keywords from.
        num_keywords: The maximum number of keywords to return. Must be a
            positive integer.

    Returns:
        A list of lowercase keyword strings, ordered from highest to
        lowest TF-IDF score, with at most `num_keywords` entries. Returns
        an empty list if `text` has no content or consists entirely of
        stopwords/punctuation.

    Raises:
        ValueError: If `num_keywords` is not a positive integer.
    """
    if num_keywords <= 0:
        raise ValueError("num_keywords must be a positive integer")

    sentences = split_sentences(text)
    if not sentences:
        return []

    vectorizer = TfidfVectorizer(
        stop_words=list(STOPWORDS), token_pattern=r"[A-Za-z']+", norm=None
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        # Raised by scikit-learn when the vocabulary is empty, e.g. the
        # text is entirely stopwords or punctuation.
        return []

    terms = vectorizer.get_feature_names_out()
    total_scores = tfidf_matrix.sum(axis=0).A1

    ranked = sorted(zip(terms, total_scores), key=lambda pair: pair[1], reverse=True)
    return [term for term, score in ranked[:num_keywords] if score > 0]
