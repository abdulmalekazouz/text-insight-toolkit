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
- **Keyword extraction.** Ranks terms by their peak TF-IDF weight across
  the document's sentences and returns the top-N as keywords.
- **Sentiment analysis.** Lexicon-based polarity scoring with a windowed
  negation rule (e.g. "not good" counts as negative), fully transparent
  about which words drove the result.

## Project layout

```
src/insighttext/
    preprocessing.py   Sentence/word tokenization and stopword filtering
    summarizer.py       TF-IDF extractive summarization
    keywords.py          TF-IDF keyword extraction
    sentiment.py          Lexicon-based sentiment analysis
    cli.py                 argparse-based CLI
tests/                  pytest suite
```

## Usage

### CLI

```bash
insighttext summarize --file article.txt --sentences 3
insighttext summarize --text "A long paragraph to condense." --sentences 1
insighttext keywords --file article.txt --count 5
insighttext sentiment --text "I absolutely loved this!"
```

### Library

```python
from insighttext.keywords import extract_keywords
from insighttext.sentiment import analyze_sentiment
from insighttext.summarizer import summarize

text = open("article.txt").read()
print(summarize(text, num_sentences=3))
print(extract_keywords(text, num_keywords=5))
print(analyze_sentiment(text))
```

## Testing

```bash
pytest --cov=insighttext --cov-report=term-missing
ruff check src tests
```

## License

MIT -- see [LICENSE](LICENSE).
