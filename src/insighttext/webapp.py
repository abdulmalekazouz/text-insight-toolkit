"""A small Flask front end over the insighttext library.

Exposes a single form where a user pastes in text and gets back a
summary, extracted keywords, and a sentiment analysis in one page.
"""

from __future__ import annotations

from flask import Flask, render_template, request

from insighttext.keywords import extract_keywords
from insighttext.sentiment import analyze_sentiment
from insighttext.summarizer import summarize

DEFAULT_SUMMARY_SENTENCES = 3
DEFAULT_KEYWORD_COUNT = 5


def create_app() -> Flask:
    """Builds and configures the insighttext Flask application.

    Returns:
        A Flask app with `/` (the form) and `/analyze` (the POST handler)
        routes registered.
    """
    app = Flask(__name__)

    @app.get("/")
    def index():
        """Renders the empty input form."""
        return render_template("index.html", text="", error=None, results=None)

    @app.post("/analyze")
    def analyze():
        """Runs summarization, keyword extraction, and sentiment analysis.

        Reads the submitted text from the `text` form field, runs all
        three analyses, and re-renders the form with the results (or with
        an error message if no text was submitted).
        """
        text = request.form.get("text", "").strip()
        if not text:
            return render_template(
                "index.html", text=text, error="Please enter some text to analyze.", results=None
            ), 400

        results = {
            "summary": summarize(text, num_sentences=DEFAULT_SUMMARY_SENTENCES),
            "keywords": extract_keywords(text, num_keywords=DEFAULT_KEYWORD_COUNT),
            "sentiment": analyze_sentiment(text),
        }
        return render_template("index.html", text=text, error=None, results=results)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
