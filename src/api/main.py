"""API REST de triagem de laudos médicos — FastAPI (R2).

Expõe a inferência do classificador de urgência via HTTP:

    - ``POST /predict`` — classifica um laudo e devolve a classe de urgência,
      o score de confiança, a distribuição por classe, a versão do modelo e o
      tempo de inferência (R2.1, R2.2, R2.3).
    - ``GET /health``   — health check com o estado de carregamento do modelo
      (R2.3).

O modelo é carregado **uma única vez** no startup, via evento ``lifespan``, e
reutilizado entre requisições — nunca por requisição (R2.6). O classificador
carregado fica em ``app.state`` para que ``/health`` possa reportar
``model_loaded`` e ``/predict`` o reutilize.

Tratamento de erros (tarefa 4.2, R2.4/R2.5):

    - **422** — corpo inválido (``text`` vazio/ausente): validação Pydantic do
      ``PredictRequest`` (automático). Além disso, se uma entrada inválida
      escapar do schema, ``InvalidInputError`` levantada na inferência é
      traduzida para 422 por um ``exception_handler`` dedicado.
    - **503** — modelo não carregado no startup: ``/predict`` responde 503 com
      ``detail="model not loaded"``.
    - **500** — falha inesperada durante a inferência: a chamada a
      ``classifier.predict`` é protegida; qualquer exceção não-esperada vira
      HTTP 500 com uma **mensagem genérica**, sem vazar stack trace nem
      detalhes internos ao cliente. O erro completo é **logado** no servidor
      em formato estruturado.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from src.api.metrics import instrument_app, observe_inference
from src.api.schemas import HealthResponse, PredictRequest, PredictResponse
from src.model.inference import Classifier, InvalidInputError

logger = logging.getLogger("triagem.api")

#: Mensagem genérica devolvida ao cliente em falha de inferência (R2.5).
#: Nunca inclui stack trace nem detalhes internos — estes vão só para o log.
_INFERENCE_ERROR_DETAIL = "internal error during inference"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o modelo uma vez no startup e o disponibiliza em ``app.state``.

    Se o carregamento falhar (artefato ausente, etc.), a aplicação **sobe mesmo
    assim** com ``classifier = None``: ``/health`` reporta ``model_loaded:
    false`` e ``/predict`` responde 503, em vez de impedir o start (R2.5/R2.6).
    A falha de carregamento é logada em nível ``error`` com o stack trace
    completo no servidor (R2.5).
    """
    try:
        app.state.classifier = Classifier()
        logger.info(
            "model loaded on startup (version=%s)",
            app.state.classifier.model_version,
        )
    except Exception:
        app.state.classifier = None
        logger.exception("failed to load model on startup; /predict will return 503")
    yield


app = FastAPI(
    title="Triagem de Laudos Médicos",
    description="API de triagem automática de laudos médicos em níveis de urgência.",
    version="0.1.0",
    lifespan=lifespan,
)

# Instrumentação Prometheus (R6.1, R6.2): middleware de métricas HTTP por
# requisição + rota ``GET /metrics`` expondo o formato de exposição Prometheus.
instrument_app(app)


@app.exception_handler(InvalidInputError)
async def invalid_input_handler(request: Request, exc: InvalidInputError) -> JSONResponse:
    """Traduz ``InvalidInputError`` da inferência em HTTP 422 (R2.4, R1.5).

    Cobre o caso em que uma entrada inválida escapa da validação do schema
    (ex.: chamada direta à inferência). A mensagem da exceção descreve o
    problema de validação sem expor detalhes internos.
    """
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc)},
    )


def _get_classifier(request: Request) -> Classifier:
    """Devolve o classificador carregado ou responde 503 se indisponível (R2.5)."""
    classifier: Classifier | None = getattr(request.app.state, "classifier", None)
    if classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="model not loaded",
        )
    return classifier


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    """Classifica um laudo em um nível de urgência (R2.1, R2.2).

    Mede o tempo da inferência (``inference_ms``) e devolve a distribuição
    completa de probabilidades por classe, além da classe de maior score.

    Erros (R2.4, R2.5):
        - 422 — ``text`` inválido (vazio/ausente): validação do schema; e
          ``InvalidInputError`` da inferência via ``exception_handler``.
        - 503 — modelo não carregado no startup (``_get_classifier``).
        - 500 — falha inesperada na inferência: mensagem genérica ao cliente,
          erro completo logado no servidor (sem vazar stack trace).
    """
    classifier = _get_classifier(request)

    start = time.perf_counter()
    try:
        result = classifier.predict(payload.text)
    except InvalidInputError:
        # Entrada inválida que escapou do schema -> 422 (via exception_handler).
        raise
    except Exception as exc:  # noqa: BLE001 - toda falha inesperada vira 500 genérico
        # Loga o erro completo (stack trace) no servidor, mas devolve ao cliente
        # apenas uma mensagem genérica — sem vazar detalhes internos (R2.5).
        logger.exception("inference failure in /predict: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_INFERENCE_ERROR_DETAIL,
        ) from exc
    inference_duration_s = time.perf_counter() - start
    inference_ms = inference_duration_s * 1000.0

    # Métricas do modelo (R6.2): latência isolada da inferência e distribuição
    # das classes previstas. Só é registrado em predição bem-sucedida.
    observe_inference(inference_duration_s, result["label"])

    return PredictResponse(
        label=result["label"],
        score=result["score"],
        scores=result["scores"],
        model_version=classifier.model_version,
        inference_ms=inference_ms,
    )


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Health check: reporta se o modelo foi carregado no startup (R2.3)."""
    classifier: Classifier | None = getattr(request.app.state, "classifier", None)
    return HealthResponse(
        status="ok",
        model_loaded=classifier is not None,
        model_version=classifier.model_version if classifier is not None else None,
    )


# ---------------------------------------------------------------------------
# Adaptador AWS Lambda (R3.1, R10.3, R10.4, R10.6)
# ---------------------------------------------------------------------------
# A **mesma** aplicação FastAPI roda em AWS Lambda (imagem de container) sem
# reescrever a lógica, via o adaptador ASGI ``mangum``. O ``Dockerfile.lambda``
# aponta o handler da Lambda para ``src.api.main.handler``.
#
# Carregamento do modelo em escopo global / cold start:
#   - O ``mangum`` executa o ciclo de vida ASGI (``lifespan``) da aplicação, que
#     instancia o ``Classifier`` **uma única vez** por container (cold start).
#     Em invocações *warm*, o mesmo container — e portanto o mesmo
#     ``app.state.classifier`` — é reutilizado, sem recarregar o modelo (R2.6).
#   - Na Lambda, ``Classifier()`` detecta as env vars ``MODEL_BUCKET`` /
#     ``MODEL_KEY`` / ``MODEL_FORMAT`` e baixa ``model.onnx`` do S3 para
#     ``/tmp`` no cold start (ver ``src/model/inference.py``). Localmente essas
#     env vars não existem, então o modelo é carregado do diretório ``models/``.
#
# ``lifespan="auto"`` deixa o ``mangum`` rodar o lifespan quando suportado,
# sem quebrar o uso local com Uvicorn (que não importa este ``handler``).
try:
    from mangum import Mangum

    # Em um HTTP API com stage nomeado (ex.: /dev, /prod), o API Gateway inclui
    # o nome do stage no path repassado ao Lambda. Sem removê-lo, o FastAPI
    # veria /dev/health em vez de /health e responderia 404. O stage é injetado
    # pelo SAM na env var STAGE (ver template.yaml); montamos o base_path a
    # partir dela — dinâmico, sem hardcode, e inerte localmente (STAGE ausente).
    _stage = os.environ.get("STAGE", "").strip().strip("/")
    _base_path = f"/{_stage}" if _stage and _stage != "$default" else None
    handler = Mangum(app, lifespan="auto", api_gateway_base_path=_base_path)
except ImportError:  # pragma: no cover - mangum é dependência de runtime (Lambda)
    # Em ambientes locais/CI sem ``mangum`` instalado, a API continua rodando
    # normalmente via Uvicorn; o ``handler`` da Lambda simplesmente não existe.
    handler = None
