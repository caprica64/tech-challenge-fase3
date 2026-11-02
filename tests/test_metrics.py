"""Testes de comportamento das métricas Prometheus (R6.1, R6.2).

``tests/test_api.py`` só confere que ``/metrics`` responde. Aqui verificamos que
as métricas **mudam** como os painéis do Grafana esperam:

    - ``http_requests_total`` incrementa por método/endpoint/status, inclusive
      para 4xx e 5xx (base da taxa de erro);
    - ``http_request_duration_seconds`` observa a duração por endpoint;
    - ``model_inference_duration_seconds`` e ``prediction_total`` só registram
      predições bem-sucedidas, rotuladas pela classe prevista.

As métricas vivem no registry global do ``prometheus_client`` e acumulam entre
testes, por isso cada teste compara o valor **antes e depois** da requisição.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY
from src.api.main import app

VALID_TEXT = "Patient presents with acute chest pain and shortness of breath"

#: Classes de urgência (valores possíveis do label de ``prediction_total``).
LABELS = ("normal", "atencao", "urgente")


def _value(name: str, **labels: str) -> float:
    """Valor atual de uma amostra do registry (0.0 se ainda não existir)."""
    return REGISTRY.get_sample_value(name, labels) or 0.0


def _requests(endpoint: str, status_code: str, method: str = "POST") -> float:
    return _value(
        "http_requests_total", method=method, endpoint=endpoint, status_code=status_code
    )


@pytest.fixture
def client() -> TestClient:
    """Cliente de teste com o ``lifespan`` ativo (carrega o modelo real)."""
    with TestClient(app) as test_client:
        yield test_client


def test_successful_predict_updates_http_and_model_metrics(client: TestClient) -> None:
    """Um ``/predict`` 200 incrementa contador, histogramas e ``prediction_total``."""
    requests_before = _requests("/predict", "200")
    duration_before = _value("http_request_duration_seconds_count", endpoint="/predict")
    inference_before = _value("model_inference_duration_seconds_count")
    predictions_before = {lbl: _value("prediction_total", label=lbl) for lbl in LABELS}

    response = client.post("/predict", json={"text": VALID_TEXT})

    assert response.status_code == 200
    label = response.json()["label"]
    assert _requests("/predict", "200") == requests_before + 1
    assert _value("http_request_duration_seconds_count", endpoint="/predict") == (
        duration_before + 1
    )
    assert _value("model_inference_duration_seconds_count") == inference_before + 1
    for lbl in LABELS:
        expected = predictions_before[lbl] + (1 if lbl == label else 0)
        assert _value("prediction_total", label=lbl) == expected


def test_validation_error_is_counted_without_prediction(client: TestClient) -> None:
    """422 entra em ``http_requests_total`` mas não conta como predição."""
    requests_before = _requests("/predict", "422")
    inference_before = _value("model_inference_duration_seconds_count")

    response = client.post("/predict", json={"text": "   "})

    assert response.status_code == 422
    assert _requests("/predict", "422") == requests_before + 1
    assert _value("model_inference_duration_seconds_count") == inference_before


def test_inference_failure_is_counted_as_5xx(client: TestClient) -> None:
    """Falha de inferência gera série ``status_code="500"`` (painel de taxa de erro)."""

    class _BrokenClassifier:
        model_version = "test-v1"

        def predict(self, text: str) -> dict:
            raise RuntimeError("boom")

    requests_before = _requests("/predict", "500")
    inference_before = _value("model_inference_duration_seconds_count")

    original = app.state.classifier
    app.state.classifier = _BrokenClassifier()
    try:
        response = client.post("/predict", json={"text": VALID_TEXT})
    finally:
        app.state.classifier = original

    assert response.status_code == 500
    assert _requests("/predict", "500") == requests_before + 1
    assert _value("model_inference_duration_seconds_count") == inference_before


def test_health_requests_use_route_template_label(client: TestClient) -> None:
    """O label ``endpoint`` usa o template da rota (``/health``)."""
    before = _requests("/health", "200", method="GET")

    client.get("/health")

    assert _requests("/health", "200", method="GET") == before + 1


def test_unknown_route_is_counted_as_404(client: TestClient) -> None:
    """Rotas inexistentes são contadas com status 404."""
    before = _requests("/does-not-exist", "404", method="GET")

    response = client.get("/does-not-exist")

    assert response.status_code == 404
    assert _requests("/does-not-exist", "404", method="GET") == before + 1
