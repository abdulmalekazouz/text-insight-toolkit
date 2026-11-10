"""Tests for insighttext.sentiment."""

from insighttext.sentiment import analyze_sentiment


def test_clearly_positive_text():
    result = analyze_sentiment("This movie was great and absolutely amazing!")
    assert result.label == "positive"
    assert result.score > 0
    assert "great" in result.positive_words
    assert "amazing" in result.positive_words
    assert result.negative_words == ()


def test_clearly_negative_text():
    result = analyze_sentiment("This movie was terrible and boring.")
    assert result.label == "negative"
    assert result.score < 0
    assert "terrible" in result.negative_words
    assert "boring" in result.negative_words


def test_negation_flips_polarity():
    result = analyze_sentiment("The movie was not good.")
    assert result.label == "negative"
    assert "good" in result.negative_words
    assert result.positive_words == ()


def test_negation_window_expires():
    # "good" is five tokens after "not", well outside the 3-token window,
    # so it should NOT be flipped.
    result = analyze_sentiment("I do not really truly honestly think the film was good.")
    assert "good" in result.positive_words


def test_text_with_no_sentiment_words_is_neutral():
    result = analyze_sentiment("The table has four legs and a flat top.")
    assert result.label == "neutral"
    assert result.score == 0.0
    assert result.positive_words == ()
    assert result.negative_words == ()


def test_balanced_mixed_sentiment_is_neutral():
    result = analyze_sentiment("The acting was great but the plot was terrible.")
    assert result.label == "neutral"
    assert result.score == 0.0
    assert "great" in result.positive_words
    assert "terrible" in result.negative_words


def test_empty_text_is_neutral():
    result = analyze_sentiment("")
    assert result.label == "neutral"
    assert result.score == 0.0


def test_duplicate_words_are_deduplicated_in_word_lists_but_not_in_score():
    result = analyze_sentiment("Good, good, good! But bad.")
    # Three "good" hits vs. one "bad" hit -> net positive score...
    assert result.score > 0
    # ...but each distinct word appears only once in the reported lists.
    assert result.positive_words == ("good",)
    assert result.negative_words == ("bad",)
