"""Tests for insighttext.cli."""

import pytest

from insighttext.cli import main

SAMPLE_TEXT = (
    "Cats are popular pets. Cats love to sleep most of the day. "
    "Dogs are popular pets too. The weather today is sunny and warm."
)


def test_summarize_with_text_argument(capsys):
    exit_code = main(["summarize", "--text", SAMPLE_TEXT, "--sentences", "2"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() != ""


def test_summarize_with_file_argument(tmp_path, capsys):
    input_path = tmp_path / "article.txt"
    input_path.write_text(SAMPLE_TEXT, encoding="utf-8")

    exit_code = main(["summarize", "--file", str(input_path), "--sentences", "1"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() != ""


def test_summarize_requires_input_source():
    with pytest.raises(SystemExit):
        main(["summarize"])


def test_summarize_rejects_both_input_sources(tmp_path):
    input_path = tmp_path / "article.txt"
    input_path.write_text(SAMPLE_TEXT, encoding="utf-8")

    with pytest.raises(SystemExit):
        main(["summarize", "--file", str(input_path), "--text", "also this"])


def test_keywords_with_text_argument(capsys):
    exit_code = main(["keywords", "--text", SAMPLE_TEXT, "--count", "3"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() != ""


def test_keywords_requires_input_source():
    with pytest.raises(SystemExit):
        main(["keywords"])
