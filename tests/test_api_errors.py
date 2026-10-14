"""Testes de tratamento de erros da API de triagem (tarefa 4.2, R2.4/R2.5).

Cobrem o contrato de erros do ``POST /predict``:

    - 200 — requisição válida devolve o schema de predição;
    - 422 — ``text`` vazio/apenas espaços (validação Pydantic) e
      ``InvalidInputError`` escapando para a inferência;
    - 503 — modelo não carregado no startup;
    - 500 — falha inesperada na inferência: mensagem **genérica**, sem vazar
      stack trace nem detalhes internos ao cliente.

Os testes usam o ``TestClient`` do FastAPI com o ``lifespan`` real (modelo
``.pkl`` presente em ``models/``) e manipulam ``app.state.classifier`` quando
precisam simular modelo ausente ou falha de inferência.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.model.inference import InvalidInputError

VALID_TEXT = "Patient presents with acute chest pain and shortness of breath"


@pytest.fixture
def client() -> TestClient:
    """Cliente de teste com o ``lifespan`` ativo (carrega o modelo real)."""
    with TestClient(app) as test_client:
        yield test_client


def test_predict_valid_returns_200_and_schema(client: TestClient) -> None:
    """Requisição válida -> 200 com o schema de predição completo (R2.2)."""
    response = client.post("/predict", json={"text": VALID_TEXT})

    assert response.status_code == 200
    body = response.json()
    assert body["label"] in {"normal", "atencao", "urgente"}
    assert 0.0 <= body["score"] <= 1.0
    assert set(body["scores"]) == {"normal", "atencao", "urgente"}
    assert isinstance(body["model_version"], str)
    assert body["inference_ms"] >= 0.0


@pytest.mark.parametrize("bad_text", ["", "   ", "\n\t "])
def test_predict_empty_or_whitespace_text_returns_422(
    client: TestClient, bad_text: str
) -> None:
    """``text`` vazio/apenas espaços -> 422 pela validação do schema (R2.4)."""
    response = client.post("/predict", json={"text": bad_text})
    assert response.status_code == 422


def test_predict_missing_text_returns_422(client: TestClient) -> None:
    """Corpo sem o campo ``text`` -> 422 (R2.4)."""
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_invalid_input_error_from_inference_maps_to_422(client: TestClient) -> None:
    """``InvalidInputError`` escapando para a inferência -> 422 (R2.4, R1.5).

    Simula o caso em que uma entrada inválida passa pelo schema (ex.: via um
    predict que valida internamente) forçando o classificador a levantar
    ``InvalidInputError``.
    """

    class _RaisingClassifier:
        model_version = "test-v1"

        def predict(self, text: str) -> dict:
            raise InvalidInputError("texto de entrada inválido")

    original = app.state.classifier
    app.state.classifier = _RaisingClassifier()
    try:
        response = client.post("/predict", json={"text": VALID_TEXT})
    finally:
        app.state.classifier = original

    assert response.status_code == 422
    assert response.json()["detail"] == "texto de entrada inválido"


def test_predict_model_not_loaded_returns_503(client: TestClient) -> None:
    """Modelo não carregado -> 503 com detalhe 'model not loaded' (R2.5)."""
    original = app.state.classifier
    app.state.classifier = None
    try:
        response = client.post("/predict", json={"text": VALID_TEXT})
    finally:
        app.state.classifier = original

    assert response.status_code == 503
    assert response.json()["detail"] == "model not loaded"


def test_predict_inference_failure_returns_500_without_stack_trace(
    client: TestClient,
) -> None:
    """Falha inesperada na inferência -> 500 genérico, sem vazar detalhes (R2.5).

    O ``detail`` devolvido ao cliente deve ser genérico e não conter a mensagem
    da exceção interna, o tipo da exceção, nem qualquer traço de stack trace.
    """
    secret_internal_message = "SECRET boom: /opt/models/model.pkl corrupt at 0xDEADBEEF"

    class _BrokenClassifier:
        model_version = "test-v1"

        def predict(self, text: str) -> dict:
            raise RuntimeError(secret_internal_message)

    original = app.state.classifier
    app.state.classifier = _BrokenClassifier()
    try:
        response = client.post("/predict", json={"text": VALID_TEXT})
    finally:
        app.state.classifier = original

    assert response.status_code == 500

    body = response.json()
    detail = body["detail"]
    # Mensagem genérica ao cliente (R2.5).
    assert detail == "internal error during inference"

    # Nenhum detalhe interno vaza no corpo da resposta.
    raw = response.text
    assert secret_internal_message not in raw
    assert "RuntimeError" not in raw
    assert "Traceback" not in raw
    assert "0xDEADBEEF" not in raw


def test_health_reports_model_loaded(client: TestClient) -> None:
    """``/health`` reporta ``model_loaded: true`` com modelo carregado (R2.3)."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
