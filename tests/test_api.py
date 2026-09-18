from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict():
    response = client.post(
        "/predict",
        json={"text": "I absolutely love this product!"}
    )

    assert response.status_code == 200

    data = response.json()

    assert data["sentiment"] in ["positive", "negative"]
    assert isinstance(data["confidence"], float)
    assert 0.0 <= data["confidence"] <= 1.0


def test_empty_text():
    response = client.post(
        "/predict",
        json={"text": ""}
    )

    assert response.status_code == 422
