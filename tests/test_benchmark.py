"""Testes do benchmark comparativo .pkl vs .onnx (tarefa 13.2, R7/R11).

Cobrem a lógica pura e reutilizável de ``scripts.benchmark`` sem depender de
artefatos de modelo pesados:

    - ``summarize_latencies`` / ``_percentile`` — agregação de latências;
    - ``_gain_pct`` — ganho percentual do otimizado sobre o baseline (R11.3);
    - ``evaluate_classifier`` — accuracy/f1 a partir de um classificador falso
      (R7.5), evitando treinar/carregar um modelo real;
    - ``build_results_payload`` + ``render_markdown`` — estrutura do JSON e do
      Markdown de saída, incluindo a verificação da meta p95 < 100 ms (R7.4).
"""

from __future__ import annotations

from scripts.benchmark import (
    P95_TARGET_MS,
    AccuracySummary,
    LatencySummary,
    ModelResult,
    _gain_pct,
    _percentile,
    build_results_payload,
    evaluate_classifier,
    render_markdown,
    summarize_latencies,
)


class _FakeClassifier:
    """Classificador falso: devolve o rótulo mapeado para cada texto exato."""

    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping

    def predict(self, text: str) -> dict:
        return {"label": self._mapping[text], "score": 1.0, "scores": {}}


def _latency(p50: float, p95: float, mean: float) -> LatencySummary:
    return LatencySummary(count=100, p50_ms=p50, p95_ms=p95, mean_ms=mean, min_ms=p50, max_ms=p95)


# -- percentil / resumo de latências ----------------------------------------


def test_percentile_endpoints_and_interpolation() -> None:
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert _percentile(values, 0) == 10.0
    assert _percentile(values, 100) == 50.0
    assert _percentile(values, 50) == 30.0


def test_summarize_latencies_basic_stats() -> None:
    summary = summarize_latencies([5.0, 1.0, 3.0, 2.0, 4.0])
    assert summary.count == 5
    assert summary.min_ms == 1.0
    assert summary.max_ms == 5.0
    assert summary.p50_ms == 3.0
    assert summary.mean_ms == 3.0


def test_summarize_latencies_rejects_empty() -> None:
    import pytest

    with pytest.raises(ValueError):
        summarize_latencies([])


# -- ganho percentual --------------------------------------------------------


def test_gain_pct_positive_when_optimized_faster() -> None:
    # 100ms -> 75ms => 25% de ganho.
    assert _gain_pct(100.0, 75.0) == 25.0


def test_gain_pct_negative_on_regression() -> None:
    assert _gain_pct(100.0, 125.0) == -25.0


def test_gain_pct_zero_baseline_is_safe() -> None:
    assert _gain_pct(0.0, 10.0) == 0.0


# -- avaliação de acurácia ---------------------------------------------------


def test_evaluate_classifier_perfect_predictions() -> None:
    texts = ["a", "b", "c"]
    labels = ["normal", "atencao", "urgente"]
    clf = _FakeClassifier(dict(zip(texts, labels, strict=True)))

    result = evaluate_classifier(clf, texts, labels)
    assert isinstance(result, AccuracySummary)
    assert result.n_samples == 3
    assert result.accuracy == 1.0
    assert result.f1_macro == 1.0
    assert set(result.f1_per_class) == {"normal", "atencao", "urgente"}


def test_evaluate_classifier_half_wrong() -> None:
    texts = ["a", "b", "c", "d"]
    labels = ["normal", "normal", "urgente", "urgente"]
    # Erra as duas 'normal' (prevê 'atencao').
    preds = {"a": "atencao", "b": "atencao", "c": "urgente", "d": "urgente"}
    clf = _FakeClassifier(preds)

    result = evaluate_classifier(clf, texts, labels)
    assert result.accuracy == 0.5


def test_evaluate_classifier_rejects_length_mismatch() -> None:
    import pytest

    with pytest.raises(ValueError):
        evaluate_classifier(_FakeClassifier({}), ["a"], ["normal", "atencao"])


# -- payload estruturado + markdown -----------------------------------------


def _make_results() -> tuple[ModelResult, ModelResult]:
    baseline = ModelResult(
        label="original (.pkl)",
        model_format="pkl",
        model_path="models/model.pkl",
        latency=_latency(10.0, 12.0, 10.5),
        accuracy=AccuracySummary(
            n_samples=50,
            accuracy=0.80,
            f1_macro=0.78,
            f1_per_class={"normal": 0.7, "atencao": 0.8, "urgente": 0.84},
        ),
    )
    optimized = ModelResult(
        label="otimizado (.onnx)",
        model_format="onnx",
        model_path="models/model.onnx",
        latency=_latency(7.0, 9.0, 7.5),
        accuracy=AccuracySummary(
            n_samples=50,
            accuracy=0.80,
            f1_macro=0.78,
            f1_per_class={"normal": 0.7, "atencao": 0.8, "urgente": 0.84},
        ),
    )
    return baseline, optimized


def test_build_payload_includes_gains_and_target_pass() -> None:
    baseline, optimized = _make_results()
    payload = build_results_payload(baseline, optimized, requests=100, warmup=10)

    assert payload["models"]["baseline"]["latency_ms"]["p95"] == 12.0
    assert payload["models"]["optimized"]["latency_ms"]["p95"] == 9.0
    # Ganho p95: (12 - 9) / 12 = 25%.
    assert payload["latency_gain"]["p95_pct"] == 25.0
    # Meta p95 < 100 ms avaliada sobre o otimizado -> PASS.
    assert payload["p95_target_check"]["evaluated_model"] == "otimizado (.onnx)"
    assert payload["p95_target_check"]["pass"] is True
    assert payload["metadata"]["p95_target_ms"] == P95_TARGET_MS


def test_build_payload_target_fail_when_p95_above_limit() -> None:
    baseline, _ = _make_results()
    slow = ModelResult(
        label="otimizado (.onnx)",
        model_format="onnx",
        model_path="models/model.onnx",
        latency=_latency(90.0, 150.0, 100.0),
        accuracy=None,
    )
    payload = build_results_payload(baseline, slow, requests=100, warmup=10)
    assert payload["p95_target_check"]["pass"] is False


def test_build_payload_baseline_only_has_no_gains() -> None:
    baseline, _ = _make_results()
    payload = build_results_payload(baseline, None, requests=100, warmup=10)
    assert payload["latency_gain"] is None
    assert "optimized" not in payload["models"]
    # Sem otimizado, a meta é verificada sobre o baseline.
    assert payload["p95_target_check"]["evaluated_model"] == "original (.pkl)"


def test_render_markdown_contains_tables_and_verdict() -> None:
    baseline, optimized = _make_results()
    payload = build_results_payload(baseline, optimized, requests=100, warmup=10)
    md = render_markdown(payload)

    assert "## Latência (ms)" in md
    assert "## Acurácia / F1" in md
    assert "Meta p95 < 100 ms" in md
    assert "PASS" in md
    assert "+25.0%" in md  # ganho p95
    assert "Trade-off" in md
