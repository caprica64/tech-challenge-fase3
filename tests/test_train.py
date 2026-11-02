"""Testes do pipeline de treino (R1, R9.7): dados, reprodutibilidade e artefatos.

Cobre ``src/model/train.py`` isoladamente, usando diretórios temporários (não
toca ``data/`` nem ``models/`` do projeto):

    - ``load_dataset``: fallback sintético quando o CSV falta, validação de
      colunas e descarte de linhas vazias;
    - ``train_model``: reprodutibilidade (seed fixa) e *quality gate* mínimo;
    - ``evaluate_model``: métricas cobrem as três classes;
    - ``run_training``: persistência de ``model.pkl`` + ``metrics.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from src.data.generate_dataset import generate_samples
from src.model.train import (
    DEFAULT_HYPERPARAMS,
    DEFAULT_TEST_SIZE,
    METRICS_FILENAME,
    MODEL_FILENAME,
    MODEL_VERSION,
    URGENCY_CLASSES,
    build_pipeline,
    evaluate_model,
    load_dataset,
    main,
    resolve_hyperparams,
    run_training,
    train_model,
)

#: F1 macro mínimo no dataset sintético. Acaso (3 classes) fica em ~0.33; um
#: valor bem acima disso detecta regressões grosseiras no pipeline (ex.:
#: vetorizador quebrado, rótulos embaralhados) sem tornar o teste frágil.
MIN_F1_MACRO = 0.8


@pytest.fixture(scope="module")
def synthetic_df() -> pd.DataFrame:
    """Dataset sintético de tamanho moderado, compartilhado pelo módulo."""
    return generate_samples(600, seed=42)


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------


def test_load_dataset_generates_fallback_when_missing(tmp_path: Path) -> None:
    """CSV ausente -> gera o dataset sintético no caminho e o carrega."""
    data_path = tmp_path / "data" / "laudos.csv"

    df = load_dataset(data_path)

    assert data_path.exists()
    assert list(df.columns) == ["text", "label"]
    assert len(df) >= 2000


def test_load_dataset_rejects_missing_columns(tmp_path: Path) -> None:
    """CSV sem ``text``/``label`` -> ``ValueError`` com as colunas faltantes."""
    data_path = tmp_path / "laudos.csv"
    pd.DataFrame({"report": ["a"], "urgency": ["normal"]}).to_csv(data_path, index=False)

    with pytest.raises(ValueError, match="label"):
        load_dataset(data_path)


def test_load_dataset_drops_blank_and_null_rows(tmp_path: Path) -> None:
    """Linhas com texto vazio/nulo ou rótulo nulo são descartadas."""
    data_path = tmp_path / "laudos.csv"
    pd.DataFrame(
        {
            "text": ["valid report", "   ", None, "no label"],
            "label": ["normal", "atencao", "urgente", None],
        }
    ).to_csv(data_path, index=False)

    df = load_dataset(data_path)

    assert list(df["text"]) == ["valid report"]


def test_load_dataset_raises_when_empty_after_cleaning(tmp_path: Path) -> None:
    """Se nada sobra após a limpeza, levanta ``ValueError``."""
    data_path = tmp_path / "laudos.csv"
    pd.DataFrame({"text": ["  ", ""], "label": ["normal", "urgente"]}).to_csv(
        data_path, index=False
    )

    with pytest.raises(ValueError, match="vazio"):
        load_dataset(data_path)


# ---------------------------------------------------------------------------
# train_model / evaluate_model
# ---------------------------------------------------------------------------


def test_train_model_is_reproducible(synthetic_df: pd.DataFrame) -> None:
    """Mesmos dados -> mesmo split e mesmas predições (seed fixa, R9.7)."""
    _, y_test_a, y_pred_a = train_model(synthetic_df)
    _, y_test_b, y_pred_b = train_model(synthetic_df)

    pd.testing.assert_series_equal(y_test_a, y_test_b)
    pd.testing.assert_series_equal(y_pred_a, y_pred_b)


def test_train_model_split_is_stratified(synthetic_df: pd.DataFrame) -> None:
    """O conjunto de teste contém todas as classes, na proporção do dataset."""
    _, y_test, _ = train_model(synthetic_df, test_size=0.2)

    assert len(y_test) == int(len(synthetic_df) * 0.2)
    counts = y_test.value_counts()
    assert set(counts.index) == set(URGENCY_CLASSES)
    assert counts.max() - counts.min() <= 1


def test_trained_model_meets_quality_gate(synthetic_df: pd.DataFrame) -> None:
    """*Quality gate*: F1 macro bem acima do acaso no dataset sintético."""
    _, y_test, y_pred = train_model(synthetic_df)

    metrics = evaluate_model(y_test, y_pred)

    assert metrics["f1_macro"] >= MIN_F1_MACRO


def test_evaluate_model_reports_every_class_even_if_absent() -> None:
    """``f1_per_class`` sempre traz as três classes (0.0 quando ausente)."""
    y_true = pd.Series(["normal", "normal", "urgente"])
    y_pred = pd.Series(["normal", "urgente", "urgente"])

    metrics = evaluate_model(y_true, y_pred)

    assert set(metrics["f1_per_class"]) == set(URGENCY_CLASSES)
    assert metrics["f1_per_class"]["atencao"] == 0.0
    assert metrics["accuracy"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# run_training (fluxo completo usado pela CLI e pela DAG)
# ---------------------------------------------------------------------------


def test_run_training_persists_model_and_metrics(tmp_path: Path) -> None:
    """Gera ``model.pkl`` carregável e ``metrics.json`` com o schema esperado."""
    data_path = tmp_path / "laudos.csv"
    generate_samples(300, seed=42).to_csv(data_path, index=False)
    model_dir = tmp_path / "models"

    metrics = run_training(data_path=data_path, model_dir=model_dir)

    model_path = model_dir / MODEL_FILENAME
    metrics_path = model_dir / METRICS_FILENAME
    assert model_path.exists()
    assert metrics_path.exists()

    pipeline = joblib.load(model_path)
    prediction = pipeline.predict(["acute chest pain with organ failure"])[0]
    assert prediction in URGENCY_CLASSES

    payload = json.loads(metrics_path.read_text())
    assert payload["model_version"] == MODEL_VERSION
    assert payload["classes"] == list(URGENCY_CLASSES)
    assert payload["accuracy"] == pytest.approx(metrics["accuracy"])
    assert set(payload["f1_per_class"]) == set(URGENCY_CLASSES)
    assert "trained_at" in payload
    assert payload["hyperparams"] == DEFAULT_HYPERPARAMS
    assert payload["test_size"] == DEFAULT_TEST_SIZE


# ---------------------------------------------------------------------------
# Hiperparâmetros
# ---------------------------------------------------------------------------


def test_build_pipeline_uses_defaults() -> None:
    """Sem overrides, o pipeline usa ``DEFAULT_HYPERPARAMS``."""
    pipeline = build_pipeline()

    tfidf, clf = pipeline.named_steps["tfidf"], pipeline.named_steps["clf"]
    assert tfidf.ngram_range == (1, DEFAULT_HYPERPARAMS["ngram_max"])
    assert tfidf.min_df == DEFAULT_HYPERPARAMS["min_df"]
    assert clf.get_params()["C"] == DEFAULT_HYPERPARAMS["C"]
    assert clf.max_iter == DEFAULT_HYPERPARAMS["max_iter"]


def test_build_pipeline_applies_partial_overrides() -> None:
    """Overrides parciais mudam só as chaves informadas."""
    pipeline = build_pipeline({"C": 0.3, "ngram_max": 3})

    assert pipeline.named_steps["clf"].C == 0.3
    assert pipeline.named_steps["tfidf"].ngram_range == (1, 3)
    assert pipeline.named_steps["tfidf"].min_df == DEFAULT_HYPERPARAMS["min_df"]


def test_resolve_hyperparams_normalizes_types() -> None:
    """Valores vindos de JSON/formulário (ex.: ``"2"``, ``1``) são normalizados."""
    hp = resolve_hyperparams({"C": 1, "ngram_max": "3"})

    assert hp["C"] == 1.0 and isinstance(hp["C"], float)
    assert hp["ngram_max"] == 3 and isinstance(hp["ngram_max"], int)


def test_resolve_hyperparams_rejects_unknown_key() -> None:
    """Chave desconhecida (ex.: typo) falha cedo em vez de ser ignorada."""
    with pytest.raises(ValueError, match="desconhecidos"):
        resolve_hyperparams({"c": 0.5})


@pytest.mark.parametrize(
    "overrides",
    [{"C": 0}, {"C": -1.0}, {"max_iter": 0}, {"ngram_max": 0}, {"min_df": 0}],
)
def test_resolve_hyperparams_rejects_out_of_range(overrides: dict) -> None:
    """Valores fora da faixa válida levantam ``ValueError``."""
    with pytest.raises(ValueError):
        resolve_hyperparams(overrides)


def test_hyperparams_change_the_trained_model(synthetic_df: pd.DataFrame) -> None:
    """Hiperparâmetros diferentes produzem modelos diferentes."""
    default_pipeline, _, _ = train_model(synthetic_df)
    custom_pipeline, _, _ = train_model(synthetic_df, hyperparams={"ngram_max": 1})

    default_vocab = default_pipeline.named_steps["tfidf"].vocabulary_
    custom_vocab = custom_pipeline.named_steps["tfidf"].vocabulary_
    assert len(custom_vocab) < len(default_vocab)
    assert all(" " not in term for term in custom_vocab), "ngram_max=1 não deve ter bigramas"


def test_cli_passes_hyperparams_to_metrics(tmp_path: Path) -> None:
    """``python -m src.model.train --C ... --ngram-max ...`` grava os valores usados."""
    data_path = tmp_path / "laudos.csv"
    generate_samples(300, seed=42).to_csv(data_path, index=False)
    model_dir = tmp_path / "models"

    exit_code = main(
        [
            "--data", str(data_path),
            "--model-dir", str(model_dir),
            "--C", "0.5",
            "--max-iter", "300",
            "--ngram-max", "1",
            "--min-df", "1",
            "--test-size", "0.25",
        ]
    )

    assert exit_code == 0
    payload = json.loads((model_dir / METRICS_FILENAME).read_text())
    assert payload["hyperparams"] == {"C": 0.5, "max_iter": 300, "ngram_max": 1, "min_df": 1}
    assert payload["test_size"] == 0.25


def test_cli_rejects_invalid_hyperparam(tmp_path: Path) -> None:
    """Valor inválido na CLI encerra com erro de uso (exit code 2)."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--data", str(tmp_path / "laudos.csv"), "--C", "0"])

    assert exc_info.value.code == 2
