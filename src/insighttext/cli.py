"""Command-line interface for insighttext.

Examples:
    insighttext summarize --file article.txt --sentences 3
    insighttext summarize --text "Some text to summarize." --sentences 1
    insighttext keywords --file article.txt --count 5
    insighttext sentiment --text "I loved this!"
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from insighttext.keywords import extract_keywords
from insighttext.sentiment import analyze_sentiment
from insighttext.summarizer import summarize


def _read_input(args: argparse.Namespace) -> str:
    """Reads the document to analyze from either --file or --text.

    Args:
        args: Parsed CLI arguments, expected to have `file` and `text`
            attributes (at most one of which is set).

    Returns:
        The document contents as a string.

    Raises:
        SystemExit: If neither or both of --file/--text were provided.
    """
    if args.file and args.text:
        raise SystemExit("error: pass either --file or --text, not both")
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    if args.text:
        return args.text
    raise SystemExit("error: pass either --file or --text")


def _cmd_summarize(args: argparse.Namespace) -> int:
    """Handles the `summarize` subcommand.

    Args:
        args: Parsed CLI arguments for the `summarize` subcommand.

    Returns:
        The process exit code (0 on success).
    """
    text = _read_input(args)
    print(summarize(text, num_sentences=args.sentences))
    return 0


def _cmd_keywords(args: argparse.Namespace) -> int:
    """Handles the `keywords` subcommand.

    Args:
        args: Parsed CLI arguments for the `keywords` subcommand.

    Returns:
        The process exit code (0 on success).
    """
    text = _read_input(args)
    keywords = extract_keywords(text, num_keywords=args.count)
    print(", ".join(keywords))
    return 0


def _cmd_sentiment(args: argparse.Namespace) -> int:
    """Handles the `sentiment` subcommand.

    Args:
        args: Parsed CLI arguments for the `sentiment` subcommand.

    Returns:
        The process exit code (0 on success).
    """
    text = _read_input(args)
    result = analyze_sentiment(text)
    print(f"{result.label} (score={result.score:+.2f})")
    if result.positive_words:
        print(f"  positive words: {', '.join(result.positive_words)}")
    if result.negative_words:
        print(f"  negative words: {', '.join(result.negative_words)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Builds the top-level argparse parser for the `insighttext` CLI.

    Returns:
        A configured `ArgumentParser` with all subcommands registered.
    """
    parser = argparse.ArgumentParser(
        prog="insighttext", description="Analyze text: summarize, extract keywords, and more."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize_parser = subparsers.add_parser(
        "summarize", help="Produce an extractive summary of a document"
    )
    summarize_parser.add_argument("--file", help="Path to a text file to summarize")
    summarize_parser.add_argument("--text", help="Raw text to summarize")
    summarize_parser.add_argument(
        "--sentences", type=int, default=3, help="Number of sentences in the summary"
    )
    summarize_parser.set_defaults(func=_cmd_summarize)

    keywords_parser = subparsers.add_parser(
        "keywords", help="Extract the top TF-IDF keywords from a document"
    )
    keywords_parser.add_argument("--file", help="Path to a text file to analyze")
    keywords_parser.add_argument("--text", help="Raw text to analyze")
    keywords_parser.add_argument(
        "--count", type=int, default=5, help="Number of keywords to extract"
    )
    keywords_parser.set_defaults(func=_cmd_keywords)

    sentiment_parser = subparsers.add_parser(
        "sentiment", help="Run lexicon-based sentiment analysis on a document"
    )
    sentiment_parser.add_argument("--file", help="Path to a text file to analyze")
    sentiment_parser.add_argument("--text", help="Raw text to analyze")
    sentiment_parser.set_defaults(func=_cmd_sentiment)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the `insighttext` CLI.

    Args:
        argv: Command-line arguments, excluding the program name. Defaults
            to `sys.argv[1:]` when not provided.

    Returns:
        The process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
