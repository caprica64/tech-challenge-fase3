"""Treino do classificador de triagem de laudos médicos (R1).

Monta um ``Pipeline`` scikit-learn de **TF-IDF + Logistic Regression** (R1.3),
treina com **seed fixa** e **split estratificado reprodutível** (R9.7), avalia
(accuracy + f1 por classe, R1.6) e persiste os artefatos para a API reutilizar
(R1.4):

    - ``models/model.pkl``    — pipeline sklearn serializado via joblib.
    - ``models/metrics.json`` — accuracy, f1 por classe, data do treino, versão.

------------------------------------------------------------------------------
Reprodutibilidade (R9.7)
------------------------------------------------------------------------------
Todo o não-determinismo é fixado por ``RANDOM_STATE = 42``: o
``train_test_split`` (com ``stratify=y`` para preservar a proporção das classes)
e o solver da ``LogisticRegression``. A mesma entrada sempre produz o mesmo
modelo e as mesmas métricas.

------------------------------------------------------------------------------
Fonte de dados
------------------------------------------------------------------------------
Lê ``data/laudos.csv`` (colunas ``text,label``) produzido por
``src/data/prepare_dataset.py`` (dataset público) ou ``generate_dataset.py``
(fallback sintético). Se o CSV não existir, o treino **gera automaticamente** o
dataset sintético (via ``generate_dataset``) para que o treino seja executável
em CI/offline sem passos manuais.

As funções ``load_dataset``, ``build_pipeline``, ``train_model``,
``evaluate_model`` e ``save_artifacts`` são expostas separadamente para reúso
pela DAG do Airflow e pelo exportador ONNX (sem duplicar lógica).

Uso:
    python -m src.model.train
    python -m src.model.train --data data/laudos.csv --model-dir models
    python -m src.model.train --test-size 0.2
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.generate_dataset import generate_dataset
from src.model.preprocessing import build_vectorizer

#: Seed fixa para todo o não-determinismo do treino (R9.7).
RANDOM_STATE = 42

#: Proporção padrão do conjunto de teste no split estratificado.
DEFAULT_TEST_SIZE = 0.2

#: Versão lógica do artefato gravada em metrics.json.
MODEL_VERSION = "pkl-v1"

#: Caminhos padrão dos artefatos de saída.
DEFAULT_DATA_PATH = Path("data/laudos.csv")
DEFAULT_MODEL_DIR = Path("models")
MODEL_FILENAME = "model.pkl"
METRICS_FILENAME = "metrics.json"

#: Classes de urgência esperadas (mesma ordem canônica do módulo de dados).
URGENCY_CLASSES: tuple[str, ...] = ("normal", "atencao", "urgente")


def load_dataset(data_path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Carrega o dataset de treino, gerando o fallback sintético se necessário.

    Se ``data_path`` não existir, gera o dataset sintético determinístico em
    ``data_path`` (via ``generate_dataset``), garantindo que o treino rode em
    ambientes de CI/offline sem passos manuais.

    Args:
        data_path: caminho do CSV com colunas ``text,label``.

    Returns:
        ``DataFrame`` com as colunas ``text`` e ``label`` (sem linhas vazias).

    Raises:
        ValueError: se o CSV não contiver as colunas esperadas ou ficar vazio.
    """
    if not data_path.exists():
        print(
            f"Dataset não encontrado em {data_path}; "
            "gerando dataset sintético (fallback CI/offline)."
        )
        generate_dataset(data_path)

    df = pd.read_csv(data_path)

    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset {data_path} não contém as colunas esperadas {sorted(missing)}. "
            f"Colunas encontradas: {list(df.columns)}."
        )

    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].astype("string").str.strip().str.len() > 0]
    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError(f"Dataset {data_path} está vazio após remover linhas inválidas.")

    return df


def build_pipeline() -> Pipeline:
    """Monta o ``Pipeline`` TF-IDF + Logistic Regression (R1.3).

    O vetorizador vem de ``preprocessing.build_vectorizer`` para manter o
    pré-processamento coeso entre treino e inferência (R9.6). O classificador
    usa seed fixa (R9.7) e ``max_iter`` folgado para garantir convergência.

    Returns:
        Um ``Pipeline`` sklearn não treinado.
    """
    return Pipeline(
        steps=[
            ("tfidf", build_vectorizer()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def train_model(
    df: pd.DataFrame,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
) -> tuple[Pipeline, pd.Series, pd.Series]:
    """Treina o pipeline com split estratificado reprodutível (R9.7).

    Args:
        df: dataset com colunas ``text,label``.
        test_size: proporção do conjunto de teste.

    Returns:
        Uma tupla ``(pipeline_treinado, y_test, y_pred)`` para avaliação.
    """
    x = df["text"]
    y = df["label"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    y_pred = pd.Series(pipeline.predict(x_test), index=y_test.index)
    return pipeline, y_test, y_pred


def evaluate_model(y_test: pd.Series, y_pred: pd.Series) -> dict:
    """Calcula accuracy e f1 por classe (R1.6).

    Args:
        y_test: rótulos verdadeiros do conjunto de teste.
        y_pred: rótulos previstos pelo modelo.

    Returns:
        Dicionário com ``accuracy``, ``f1_macro`` e ``f1_per_class`` (mapa
        classe -> f1), cobrindo as três classes de urgência.
    """
    labels = list(URGENCY_CLASSES)
    accuracy = float(accuracy_score(y_test, y_pred))
    f1_macro = float(f1_score(y_test, y_pred, labels=labels, average="macro", zero_division=0))
    per_class = f1_score(y_test, y_pred, labels=labels, average=None, zero_division=0)
    f1_per_class = {cls: float(score) for cls, score in zip(labels, per_class, strict=True)}

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_per_class": f1_per_class,
    }


def save_artifacts(
    pipeline: Pipeline,
    metrics: dict,
    *,
    model_dir: Path = DEFAULT_MODEL_DIR,
) -> tuple[Path, Path]:
    """Persiste o modelo (``model.pkl``) e as métricas (``metrics.json``) (R1.4).

    Args:
        pipeline: pipeline sklearn treinado.
        metrics: métricas de avaliação (ver ``evaluate_model``).
        model_dir: diretório de saída dos artefatos.

    Returns:
        Tupla ``(caminho_modelo, caminho_metrics)``.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / MODEL_FILENAME
    metrics_path = model_dir / METRICS_FILENAME

    joblib.dump(pipeline, model_path)

    metrics_payload = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(UTC).isoformat(),
        "classes": list(URGENCY_CLASSES),
        **metrics,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False) + "\n")

    return model_path, metrics_path


def run_training(
    *,
    data_path: Path = DEFAULT_DATA_PATH,
    model_dir: Path = DEFAULT_MODEL_DIR,
    test_size: float = DEFAULT_TEST_SIZE,
) -> dict:
    """Executa o fluxo completo de treino: carregar, treinar, avaliar, persistir.

    Função de conveniência reutilizável pela DAG do Airflow.

    Args:
        data_path: caminho do dataset.
        model_dir: diretório de saída dos artefatos.
        test_size: proporção do conjunto de teste.

    Returns:
        As métricas de avaliação (ver ``evaluate_model``).
    """
    df = load_dataset(data_path)
    pipeline, y_test, y_pred = train_model(df, test_size=test_size)
    metrics = evaluate_model(y_test, y_pred)
    model_path, metrics_path = save_artifacts(pipeline, metrics, model_dir=model_dir)

    print(f"Modelo salvo em {model_path}")
    print(f"Métricas salvas em {metrics_path}")
    print(f"Accuracy: {metrics['accuracy']:.4f} | F1 macro: {metrics['f1_macro']:.4f}")
    print("F1 por classe:")
    for cls in URGENCY_CLASSES:
        print(f"  {cls:8s}: {metrics['f1_per_class'][cls]:.4f}")

    return metrics


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada de linha de comando."""
    parser = argparse.ArgumentParser(
        description=(
            "Treina o pipeline TF-IDF + Logistic Regression de triagem de laudos "
            "e persiste models/model.pkl + models/metrics.json."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Caminho do dataset CSV (padrão: {DEFAULT_DATA_PATH}).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Diretório de saída dos artefatos (padrão: {DEFAULT_MODEL_DIR}).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help=f"Proporção do conjunto de teste (padrão: {DEFAULT_TEST_SIZE}).",
    )
    args = parser.parse_args(argv)

    run_training(
        data_path=args.data,
        model_dir=args.model_dir,
        test_size=args.test_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
