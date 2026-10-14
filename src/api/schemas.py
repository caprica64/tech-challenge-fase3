"""Schemas Pydantic da API de triagem de laudos médicos (R2.1, R2.2, R2.3).

Define os contratos de entrada e saída dos endpoints da API:

    - ``PredictRequest``  — corpo de ``POST /predict`` (texto do laudo).
    - ``PredictResponse`` — resposta de ``POST /predict`` (classe, score,
      distribuição por classe, versão do modelo e tempo de inferência).
    - ``HealthResponse``  — resposta de ``GET /health`` (status e estado do
      modelo).

A validação de ``text`` (obrigatório, não-vazio após ``strip``) é feita pelo
Pydantic e resulta automaticamente em HTTP 422 quando violada (R2.4, R1.5).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class UrgencyLabel(str, Enum):
    """Classes de urgência previstas pela triagem (ordem canônica)."""

    normal = "normal"
    atencao = "atencao"
    urgente = "urgente"


class PredictRequest(BaseModel):
    """Corpo da requisição de ``POST /predict`` (R2.1).

    ``text`` é obrigatório e não pode ser vazio nem conter apenas espaços em
    branco: ``min_length=1`` combinado com ``strip_whitespace`` garante que uma
    string em branco seja rejeitada com HTTP 422 (R2.4, R1.5).
    """

    text: str = Field(
        ...,
        min_length=1,
        description="Texto do laudo médico a ser triado.",
        json_schema_extra={
            "example": "Patient presents with acute chest pain and shortness of breath",
        },
    )

    model_config = {
        "str_strip_whitespace": True,
    }


class PredictResponse(BaseModel):
    """Resposta de ``POST /predict`` (R2.2, R2.3).

    Attributes:
        label: classe de urgência prevista (``normal``/``atencao``/``urgente``).
        score: confiança (probabilidade máxima) em ``[0, 1]``.
        scores: distribuição completa de probabilidades por classe.
        model_version: identificador da versão do modelo que gerou a predição.
        inference_ms: tempo da inferência em milissegundos.
    """

    label: UrgencyLabel
    score: float = Field(..., ge=0.0, le=1.0)
    scores: dict[str, float]
    model_version: str
    inference_ms: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "label": "urgente",
                "score": 0.93,
                "scores": {"normal": 0.02, "atencao": 0.05, "urgente": 0.93},
                "model_version": "onnx-v1",
                "inference_ms": 4.1,
            },
        },
    }


class HealthResponse(BaseModel):
    """Resposta de ``GET /health`` (R2.3).

    Attributes:
        status: ``"ok"`` quando a API está no ar.
        model_loaded: ``True`` se o modelo foi carregado com sucesso no startup.
        model_version: versão do modelo carregado, ou ``None`` se indisponível.
    """

    status: str
    model_loaded: bool
    model_version: str | None = None
