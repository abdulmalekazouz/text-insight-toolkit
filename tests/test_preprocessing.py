"""Tests for insighttext.preprocessing."""

from insighttext.preprocessing import remove_stopwords, split_sentences, tokenize_words


def test_split_sentences_handles_multiple_sentences():
    text = "This is one. This is two! Is this three?"
    assert split_sentences(text) == [
        "This is one.",
        "This is two!",
        "Is this three?",
    ]


def test_split_sentences_collapses_whitespace():
    text = "Line one.\n\n   Line two."
    assert split_sentences(text) == ["Line one.", "Line two."]


def test_split_sentences_empty_input_returns_empty_list():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_tokenize_words_lowercases_and_keeps_contractions():
    assert tokenize_words("Don't Panic!") == ["don't", "panic"]


def test_tokenize_words_ignores_punctuation_and_numbers():
    assert tokenize_words("Hello, world... 123") == ["hello", "world"]


def test_remove_stopwords_filters_common_words():
    words = ["the", "quick", "brown", "fox", "is", "running"]
    assert remove_stopwords(words) == ["quick", "brown", "fox", "running"]


def test_remove_stopwords_preserves_order_and_duplicates():
    words = ["data", "is", "data", "not", "the", "answer"]
    assert remove_stopwords(words) == ["data", "data", "answer"]
