"""Testes de contrato da API de triagem (tarefa 7.2, R8.4/R2.1/R2.3/R2.4).

Testes de "happy path" dos endpoints via ``TestClient``, complementares aos de
tratamento de erro em ``tests/test_api_errors.py``:

    - ``POST /predict`` com texto válido -> 200 com o schema de predição
      completo (``label`` válido, ``score`` em ``[0, 1]``, distribuição
      ``scores`` por classe, ``model_version`` e ``inference_ms``) (R2.1, R2.2);
    - ``POST /predict`` com corpo inválido (vazio/apenas espaços/ausente)
      -> 422 pela validação do schema (R2.4);
    - ``GET /health`` -> 200 com ``{status, model_loaded, model_version}``
      (R2.3);
    - ``GET /metrics`` -> 200, ``Content-Type`` ``text/plain`` e nomes de
      métricas Prometheus esperadas.

Nota sobre ``/metrics``: o endpoint e a instrumentação Prometheus pertencem à
tarefa 10 (ainda não implementada). Enquanto a rota ``/metrics`` não estiver
registrada na aplicação, o teste correspondente é **ignorado** com motivo
explícito (``pytest.mark.skipif`` baseado nas rotas do app). Assim que a tarefa
10 registrar ``/metrics``, o teste passa a rodar automaticamente — sem edição
aqui. Este arquivo **não** implementa ``/metrics`` (fora do escopo da 7.2).

Os testes usam o ``TestClient`` com o ``lifespan`` real, que carrega o modelo
``.pkl`` presente em ``models/``.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

#: Laudo válido (um exemplo representativo) para exercitar o caminho feliz.
VALID_TEXT = "Patient presents with acute chest pain and shortness of breath"

#: Classes de urgência esperadas na resposta (ordem canônica).
URGENCY_LABELS = {"normal", "atencao", "urgente"}


def _has_route(path: str) -> bool:
    """Indica se a aplicação já registrou uma rota com o ``path`` dado."""
    return any(getattr(route, "path", None) == path for route in app.routes)


#: ``/metrics`` só existe a partir da tarefa 10 (instrumentação Prometheus).
_METRICS_REGISTERED = _has_route("/metrics")
_METRICS_SKIP_REASON = (
    "endpoint GET /metrics ainda não registrado (tarefa 10 — métricas Prometheus); "
    "o teste roda automaticamente quando a rota existir."
)


@pytest.fixture
def client() -> TestClient:
    """Cliente de teste com o ``lifespan`` ativo (carrega o modelo real)."""
    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# POST /predict — 200 (contrato de sucesso)  (R2.1, R2.2)
# ---------------------------------------------------------------------------


def test_predict_valid_text_returns_200_and_schema(client: TestClient) -> None:
    """``POST /predict`` com texto válido -> 200 com o schema completo (R2.1)."""
    response = client.post("/predict", json={"text": VALID_TEXT})

    assert response.status_code == 200
    body = response.json()

    assert body["label"] in URGENCY_LABELS
    assert 0.0 <= body["score"] <= 1.0

    scores = body["scores"]
    assert set(scores) == URGENCY_LABELS
    assert all(0.0 <= v <= 1.0 for v in scores.values())
    # ``score`` é a probabilidade máxima e ``label`` a classe correspondente.
    assert body["score"] == pytest.approx(max(scores.values()))
    assert body["label"] == max(scores, key=scores.__getitem__)

    assert isinstance(body["model_version"], str) and body["model_version"]
    assert isinstance(body["inference_ms"], int | float)
    assert body["inference_ms"] >= 0.0


# ---------------------------------------------------------------------------
# POST /predict — 422 (corpo inválido)  (R2.4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_text", ["", "   ", "\n\t "])
def test_predict_empty_or_whitespace_returns_422(
    client: TestClient, bad_text: str
) -> None:
    """``text`` vazio/apenas espaços -> 422 pela validação do schema (R2.4)."""
    response = client.post("/predict", json={"text": bad_text})
    assert response.status_code == 422


def test_predict_missing_text_returns_422(client: TestClient) -> None:
    """Corpo sem o campo ``text`` -> 422 (R2.4)."""
    response = client.post("/predict", json={})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /health — 200  (R2.3)
# ---------------------------------------------------------------------------


def test_health_returns_200_and_schema(client: TestClient) -> None:
    """``GET /health`` -> 200 com ``{status, model_loaded, model_version}`` (R2.3)."""
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    # Com o modelo carregado, ``model_version`` é uma string não-vazia.
    assert isinstance(body["model_version"], str) and body["model_version"]


# ---------------------------------------------------------------------------
# GET /metrics — 200 text/plain (Prometheus)  — tarefa 10
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _METRICS_REGISTERED, reason=_METRICS_SKIP_REASON)
def test_metrics_returns_200_text_plain(client: TestClient) -> None:
    """``GET /metrics`` -> 200, ``text/plain`` e nomes de métricas Prometheus.

    Só executa quando a rota ``/metrics`` já estiver registrada (tarefa 10).
    """
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")

    payload = response.text
    assert "http_requests_total" in payload
