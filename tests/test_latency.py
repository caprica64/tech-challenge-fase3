"""Teste de SLA de latência da inferência otimizada (R7.4).

Exporta um pipeline pequeno para ONNX, carrega-o pelo ``Classifier`` (o mesmo
caminho usado pela API) e verifica que o p95 de inferência fica abaixo da meta
de 100 ms. A meta real tem folga de ~2 ordens de grandeza, então o teste não é
sensível a ruído de máquinas de CI — ele só pega regressões grosseiras (ex.:
modelo recarregado a cada chamada, sessão ONNX recriada por requisição).
"""

from __future__ import annotations

import statistics
import time

import joblib
import pytest
from src.data.generate_dataset import generate_samples
from src.model.export_onnx import export_onnx
from src.model.inference import Classifier, reset_classifier
from src.model.train import MODEL_FILENAME, build_pipeline

#: Meta de latência p95 local (ms), conforme o README.
P95_TARGET_MS = 100.0

WARMUP = 20
REQUESTS = 200
TEXT = "Patient presents with acute crushing chest pain and signs of organ failure."


@pytest.fixture(scope="module")
def onnx_classifier(tmp_path_factory: pytest.TempPathFactory) -> Classifier:
    """``Classifier`` carregando um ``model.onnx`` exportado de um pipeline pequeno."""
    model_dir = tmp_path_factory.mktemp("latency_model")
    df = generate_samples(300, seed=42)
    pipeline = build_pipeline().fit(df["text"], df["label"])
    joblib.dump(pipeline, model_dir / MODEL_FILENAME)
    export_onnx(model_path=model_dir / MODEL_FILENAME, output_path=model_dir / "model.onnx")

    reset_classifier()
    yield Classifier(model_format="onnx", model_dir=model_dir)
    reset_classifier()


def test_onnx_model_is_actually_loaded(onnx_classifier: Classifier) -> None:
    """Garante que o teste mede o ONNX, não o fallback ``.pkl``."""
    assert onnx_classifier.model_format == "onnx"


def test_onnx_inference_p95_below_target(onnx_classifier: Classifier) -> None:
    """p95 da inferência ONNX < 100 ms."""
    for _ in range(WARMUP):
        onnx_classifier.predict(TEXT)

    latencies_ms = []
    for _ in range(REQUESTS):
        start = time.perf_counter()
        onnx_classifier.predict(TEXT)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    p95 = statistics.quantiles(latencies_ms, n=100)[94]
    assert p95 < P95_TARGET_MS, f"p95={p95:.2f} ms excede a meta de {P95_TARGET_MS} ms"
