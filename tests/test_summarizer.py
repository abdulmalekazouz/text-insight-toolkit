"""Tests for insighttext.summarizer."""

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from insighttext.preprocessing import STOPWORDS, split_sentences
from insighttext.summarizer import score_sentences, summarize

SAMPLE_TEXT = (
    "Cats are popular pets. Cats love to sleep most of the day. "
    "Dogs are popular pets too. The weather today is sunny and warm."
)


def test_score_sentences_empty_input():
    assert score_sentences([]) == []


def test_score_sentences_single_sentence_is_zero():
    assert score_sentences(["Only one sentence here."]) == [0.0]


def test_score_sentences_matches_independent_tfidf_computation():
    sentences = split_sentences(SAMPLE_TEXT)
    vectorizer = TfidfVectorizer(stop_words=list(STOPWORDS), token_pattern=r"[A-Za-z']+")
    matrix = vectorizer.fit_transform(sentences)
    expected = matrix.sum(axis=1).A1.tolist()
    assert score_sentences(sentences) == pytest.approx(expected)


def test_score_sentences_returns_one_score_per_sentence():
    sentences = split_sentences(SAMPLE_TEXT)
    assert len(score_sentences(sentences)) == len(sentences)


def test_summarize_short_text_returns_everything():
    text = "Short text. Just two sentences."
    assert summarize(text, num_sentences=5) == "Short text. Just two sentences."


def test_summarize_preserves_original_sentence_order():
    summary = summarize(SAMPLE_TEXT, num_sentences=2)
    all_sentences = split_sentences(SAMPLE_TEXT)
    summary_sentences = split_sentences(summary)

    assert len(summary_sentences) == 2
    # The chosen sentences must appear in the same relative order as in the
    # source document, not re-ordered by score.
    indices = [all_sentences.index(s) for s in summary_sentences]
    assert indices == sorted(indices)


def test_summarize_rejects_non_positive_sentence_count():
    with pytest.raises(ValueError):
        summarize(SAMPLE_TEXT, num_sentences=0)
    with pytest.raises(ValueError):
        summarize(SAMPLE_TEXT, num_sentences=-1)


def test_summarize_empty_text_returns_empty_string():
    assert summarize("", num_sentences=3) == ""
