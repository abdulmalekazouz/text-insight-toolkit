"""Tests for insighttext.webapp."""

import pytest

from insighttext.webapp import create_app

SAMPLE_TEXT = (
    "Machine learning models learn patterns from data. "
    "Neural networks are a popular machine learning technique. "
    "I absolutely love how well this technique works."
)


@pytest.fixture
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def test_index_page_loads(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Text Insight Toolkit" in response.data


def test_analyze_happy_path(client):
    response = client.post("/analyze", data={"text": SAMPLE_TEXT})
    assert response.status_code == 200
    assert b"Summary" in response.data
    assert b"Keywords" in response.data
    assert b"positive" in response.data.lower()


def test_analyze_rejects_empty_text(client):
    response = client.post("/analyze", data={"text": "   "})
    assert response.status_code == 400
    assert b"enter some text" in response.data
