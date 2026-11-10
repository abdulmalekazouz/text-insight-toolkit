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

## Features

- **Extractive summarization.** Scores each sentence by its TF-IDF weight
  relative to the rest of the document and returns the top-scoring
  sentences, in their original order.

## Project layout

```
src/insighttext/
    preprocessing.py   Sentence/word tokenization and stopword filtering
    summarizer.py       TF-IDF extractive summarization
    cli.py               argparse-based CLI
tests/                  pytest suite
```

## Usage

### CLI

```bash
insighttext summarize --file article.txt --sentences 3
insighttext summarize --text "A long paragraph to condense." --sentences 1
```

### Library

```python
from insighttext.summarizer import summarize

summary = summarize(open("article.txt").read(), num_sentences=3)
print(summary)
```

## Testing

```bash
pytest --cov=insighttext --cov-report=term-missing
ruff check src tests
```

## License

MIT -- see [LICENSE](LICENSE).
