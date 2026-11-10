# Text Insight Toolkit

A lightweight text analysis toolkit: extractive summarization, TF-IDF
keyword extraction, lexicon-based sentiment analysis, a CLI, and a Flask
web UI. Every technique is deterministic and runs fully offline -- no API
keys, no downloaded language models, no network calls.

## Why this project

Classic NLP/ML techniques -- TF-IDF, extractive summarization, lexicon
sentiment scoring -- are a good way to show a complete, testable data
pipeline without depending on an external LLM API or a large downloaded
model. Built incrementally, one feature per branch/pull request.

## Installation

Requires Python 3.9+.

```bash
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

## Project layout

```
src/insighttext/
    preprocessing.py   Sentence/word tokenization and stopword filtering
tests/                  pytest suite
```

## Testing

```bash
pytest --cov=insighttext --cov-report=term-missing
ruff check src tests
```

## License

MIT -- see [LICENSE](LICENSE).
