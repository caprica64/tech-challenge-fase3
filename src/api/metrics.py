"""Instrumentação de métricas Prometheus para a API de triagem (R6.1, R6.2).

Define as métricas do serviço de inferência via ``prometheus_client`` e as
expõe no formato de exposição Prometheus através do endpoint ``GET /metrics``.
O design (seção 3 — "Instrumentação de Métricas") especifica quatro métricas:

    - ``http_requests_total`` — Counter, labels ``method``/``endpoint``/
      ``status_code``: total de requisições + taxa de erro.
    - ``http_request_duration_seconds`` — Histogram, label ``endpoint``:
      latência (p50/p95 via ``histogram_quantile``).
    - ``model_inference_duration_seconds`` — Histogram, sem labels: latência
      isolada da inferência.
    - ``prediction_total`` — Counter, label ``label``: distribuição das classes
      previstas.

------------------------------------------------------------------------------
Registro único (evitar dupla-definição)
------------------------------------------------------------------------------
``prometheus_client`` levanta ``ValueError`` ao registrar uma mesma série de
tempo mais de uma vez no mesmo registry. Como este módulo define as métricas
em escopo de import, elas são criadas **uma única vez** por processo — o padrão
recomendado. Em cenários de reimport (certos fluxos de teste/reload) isso ainda
é seguro porque o módulo só é executado uma vez; não há criação de métricas por
requisição.

------------------------------------------------------------------------------
Middleware HTTP
------------------------------------------------------------------------------
``PrometheusMiddleware`` registra, por requisição, ``http_requests_total`` e
``http_request_duration_seconds``. Para evitar labels de alta cardinalidade, o
``endpoint`` usa o *template* da rota (ex.: ``/predict``) obtido do escopo da
requisição quando disponível, e não o caminho bruto da URL.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

if TYPE_CHECKING:  # pragma: no cover - apenas para type checking
    from collections.abc import Awaitable, Callable

    from fastapi import FastAPI
    from starlette.requests import Request
    from starlette.responses import Response

# ---------------------------------------------------------------------------
# Definição das métricas (escopo de módulo: criadas uma única vez por processo)
# ---------------------------------------------------------------------------

#: Total de requisições HTTP, rotulado por método, endpoint e status (R6.2).
#: Base para o painel "Total de requisições" e para a taxa de erro
#: (``status_code=~"5.."``).
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total de requisições HTTP processadas pela API.",
    labelnames=("method", "endpoint", "status_code"),
)

#: Duração das requisições HTTP (segundos), por endpoint (R6.2).
#: Permite calcular p50/p95 via ``histogram_quantile`` no dashboard.
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "Duração das requisições HTTP em segundos.",
    labelnames=("endpoint",),
)

#: Duração isolada da inferência do modelo (segundos), sem overhead HTTP.
MODEL_INFERENCE_DURATION_SECONDS = Histogram(
    "model_inference_duration_seconds",
    "Duração da inferência do modelo em segundos (sem overhead de HTTP).",
)

#: Total de predições por classe prevista — distribuição das saídas do modelo.
PREDICTION_TOTAL = Counter(
    "prediction_total",
    "Total de predições realizadas, rotulado pela classe prevista.",
    labelnames=("label",),
)


def _route_template(request: Request) -> str:
    """Devolve o *template* da rota da requisição (ex.: ``/predict``).

    Usa ``request.scope['route'].path`` quando a requisição casou com uma rota
    registrada, evitando labels de alta cardinalidade a partir do caminho bruto.
    Para requisições sem rota correspondente (ex.: 404), recai no caminho da URL.
    """
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    if path:
        return path
    return request.url.path


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware que instrumenta cada requisição HTTP (R6.2).

    Mede a duração da requisição e incrementa ``http_requests_total`` e
    ``http_request_duration_seconds``. Garante o registro das métricas mesmo
    quando o handler levanta uma exceção (contabiliza como status 500), sem
    suprimir o erro — a exceção é repropagada para o tratamento normal da app.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            duration = time.perf_counter() - start
            endpoint = _route_template(request)
            HTTP_REQUEST_DURATION_SECONDS.labels(endpoint=endpoint).observe(duration)
            HTTP_REQUESTS_TOTAL.labels(
                method=request.method,
                endpoint=endpoint,
                status_code=str(status_code),
            ).inc()


def observe_inference(duration_seconds: float, label: str) -> None:
    """Registra uma inferência bem-sucedida nas métricas do modelo.

    Observa a latência isolada da inferência em
    ``model_inference_duration_seconds`` e incrementa ``prediction_total`` para
    a classe prevista. Deve ser chamada pelo handler de ``/predict`` após uma
    predição bem-sucedida.

    Args:
        duration_seconds: duração da inferência em segundos.
        label: classe prevista (ex.: ``normal``, ``atencao``, ``urgente``).
    """
    MODEL_INFERENCE_DURATION_SECONDS.observe(duration_seconds)
    PREDICTION_TOTAL.labels(label=label).inc()


def render_metrics() -> tuple[bytes, str]:
    """Renderiza as métricas no formato de exposição Prometheus (R6.1).

    Returns:
        Tupla ``(payload, content_type)`` com o corpo serializado por
        ``generate_latest()`` e o ``Content-Type`` oficial do Prometheus
        (``CONTENT_TYPE_LATEST`` — ``text/plain; version=...``).
    """
    return generate_latest(), CONTENT_TYPE_LATEST


def instrument_app(app: FastAPI) -> None:
    """Aplica a instrumentação Prometheus à aplicação FastAPI (R6.1, R6.2).

    Adiciona o ``PrometheusMiddleware`` (métricas HTTP por requisição) e
    registra a rota ``GET /metrics`` que expõe as métricas via
    ``generate_latest()``.
    """
    from starlette.responses import Response

    app.add_middleware(PrometheusMiddleware)

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        """Expõe as métricas no formato Prometheus (R6.1)."""
        payload, content_type = render_metrics()
        return Response(content=payload, media_type=content_type)
