"""Tests for insighttext.keywords."""

import pytest

from insighttext.keywords import extract_keywords

SAMPLE_TEXT = (
    "Machine learning models learn patterns from data. "
    "Neural networks are a popular machine learning technique. "
    "The weather today is sunny and warm."
)


def test_extract_keywords_returns_requested_count_or_fewer():
    keywords = extract_keywords(SAMPLE_TEXT, num_keywords=3)
    assert 0 < len(keywords) <= 3


def test_extract_keywords_are_lowercase_and_not_stopwords():
    keywords = extract_keywords(SAMPLE_TEXT, num_keywords=10)
    for keyword in keywords:
        assert keyword == keyword.lower()
        assert keyword not in {"is", "the", "a", "and"}


def test_extract_keywords_ranks_domain_terms_above_filler():
    keywords = extract_keywords(SAMPLE_TEXT, num_keywords=3)
    # "machine"/"learning" appear in two of the three sentences and should
    # outrank single-appearance filler words from the unrelated sentence.
    assert any(term in keywords for term in ("machine", "learning"))
    assert "sunny" not in keywords


def test_extract_keywords_empty_text_returns_empty_list():
    assert extract_keywords("", num_keywords=5) == []


def test_extract_keywords_only_stopwords_returns_empty_list():
    assert extract_keywords("The a is of and.", num_keywords=5) == []


def test_extract_keywords_rejects_non_positive_count():
    with pytest.raises(ValueError):
        extract_keywords(SAMPLE_TEXT, num_keywords=0)
    with pytest.raises(ValueError):
        extract_keywords(SAMPLE_TEXT, num_keywords=-2)
