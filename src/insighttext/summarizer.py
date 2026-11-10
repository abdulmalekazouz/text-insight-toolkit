"""Extractive text summarization.

Scores each sentence in a document by how much TF-IDF "weight" it carries
relative to the rest of the document, then selects the highest-scoring
sentences as the summary. This is a simplified, scikit-learn-based variant
of classic extractive summarization techniques (e.g. the scoring step of
TextRank): sentences built from frequent-but-informative terms -- terms
that are common within the document but not so common that TF-IDF
discounts them -- tend to carry the document's main ideas.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

from insighttext.preprocessing import STOPWORDS, split_sentences


def score_sentences(sentences: list[str]) -> list[float]:
    """Scores each sentence by its total TF-IDF weight.

    Args:
        sentences: The document's sentences, in their original order.

    Returns:
        A list of scores parallel to `sentences` (same length and order).
        Each score is the sum of the TF-IDF weights of the terms in that
        sentence, computed with the rest of `sentences` as the corpus.
        Returns an empty list if `sentences` is empty, and a list of zeros
        if there is only one sentence (TF-IDF needs more than one document
        to be meaningful).
    """
    if not sentences:
        return []
    if len(sentences) == 1:
        return [0.0]

    # Use the same word pattern as insighttext.preprocessing.tokenize_words
    # (letters and apostrophes) so contractions in STOPWORDS actually match
    # the tokens scikit-learn extracts; otherwise "don't" in the stopword
    # list never matches scikit-learn's default "don" / "t" split tokens.
    vectorizer = TfidfVectorizer(stop_words=list(STOPWORDS), token_pattern=r"[A-Za-z']+")
    tfidf_matrix = vectorizer.fit_transform(sentences)
    return tfidf_matrix.sum(axis=1).A1.tolist()


def summarize(text: str, num_sentences: int = 3) -> str:
    """Produces an extractive summary of `text`.

    Args:
        text: The document to summarize.
        num_sentences: The maximum number of sentences to include in the
            summary. Must be a positive integer.

    Returns:
        The top-scoring sentences, joined with a single space, in their
        original order of appearance (not sorted by score) so the summary
        still reads coherently. If `text` has `num_sentences` or fewer
        sentences, the whole (sentence-split and rejoined) text is
        returned.

    Raises:
        ValueError: If `num_sentences` is not a positive integer.
    """
    if num_sentences <= 0:
        raise ValueError("num_sentences must be a positive integer")

    sentences = split_sentences(text)
    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    scores = score_sentences(sentences)
    ranked_indices = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
    top_indices = sorted(ranked_indices[:num_sentences])
    return " ".join(sentences[i] for i in top_indices)
