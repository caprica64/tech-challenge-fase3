"""Fallback sintético do dataset de triagem para CI/offline.

Este módulo é o **fallback** do preparo de dados (R9.2): quando o dataset
público (Medical Abstracts TC Corpus, ver ``prepare_dataset.py``) não está
disponível — por exemplo em ambientes de CI sem acesso à rede — geramos um
dataset sintético rotulado **em inglês** (R9.1), suficiente para rodar o treino
e os testes de forma determinística, sem bloquear o pipeline.

------------------------------------------------------------------------------
Natureza dos dados (R9.5, R9.8)
------------------------------------------------------------------------------
Os dados são **100% sintéticos**, construídos por composição de templates e
vocabulário clínico genérico. Não há qualquer informação pessoal identificável
(PII) real: nomes, idades e identificadores são placeholders gerados
programaticamente. Isso satisfaz R9.8 (sem PII real) por construção.

------------------------------------------------------------------------------
Estratégia de geração
------------------------------------------------------------------------------
Para cada classe de urgência ``{normal, atencao, urgente}`` definimos:
    - um conjunto de *templates* de frase (com marcadores ``{...}``);
    - vocabulários específicos da classe (sintomas, achados, condições) que
      refletem a acuidade típica daquela urgência.

A combinação de templates x vocabulários gera uma grande variedade de laudos
plausíveis por classe. A geração usa uma **seed fixa** (``DEFAULT_SEED = 42``,
R9.8/R9.7) para reprodutibilidade total: a mesma seed sempre produz o mesmo
CSV. As amostras são distribuídas de forma balanceada entre as três classes.

------------------------------------------------------------------------------
Saída
------------------------------------------------------------------------------
Grava a saída canônica em ``data/laudos.csv`` com as colunas ``text,label``
(mesmo contrato de ``prepare_dataset.py``), com no mínimo 2.000 amostras (R9.3).

Uso:
    python -m src.data.generate_dataset
    python -m src.data.generate_dataset --output data/laudos.csv --samples 2400
    python -m src.data.generate_dataset --seed 42
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

#: Seed padrão para geração reprodutível (R9.7, R9.8).
DEFAULT_SEED = 42

#: Número mínimo/padrão de amostras exigido pelo projeto (R9.3).
DEFAULT_SAMPLES = 2400

#: Classes de urgência válidas na saída canônica (mesma ordem de prepare_dataset).
URGENCY_CLASSES: tuple[str, ...] = ("normal", "atencao", "urgente")


# ---------------------------------------------------------------------------
# Templates por classe de urgência
# ---------------------------------------------------------------------------
# Cada classe tem templates de frase e vocabulários que refletem a acuidade
# clínica típica. Textos em inglês (R9.1). Marcadores entre chaves são
# preenchidos a partir dos vocabulários da respectiva classe.

_TEMPLATES: dict[str, tuple[str, ...]] = {
    "normal": (
        "Patient reports {symptom}. Vital signs stable, no acute distress noted.",
        "Routine follow-up for {condition}. Findings are within normal limits.",
        "Mild {symptom} for the past few days. Examination unremarkable.",
        "Stable {condition} on review. No new complaints reported today.",
        "Minor {symptom} resolving with rest. No concerning features identified.",
        "Annual check-up shows {finding}. Advised routine monitoring.",
        "Patient presents with {symptom}; otherwise well and afebrile.",
        "Non-urgent evaluation of {condition}. Reassurance and home care advised.",
    ),
    "atencao": (
        "Patient presents with persistent {symptom} that warrants further evaluation.",
        "Moderate {condition} with {finding}; recommend follow-up and monitoring.",
        "Recurrent {symptom} not fully controlled; clinical review suggested.",
        "Abnormal {finding} noted; non-emergent workup recommended.",
        "Worsening {symptom} over several weeks; schedule outpatient assessment.",
        "Suspected {condition} based on {finding}; further testing indicated.",
        "Patient with {symptom} requiring closer observation and reassessment.",
        "Intermittent {symptom}; findings suggest {condition} to be investigated.",
    ),
    "urgente": (
        "Patient presents with acute {symptom} and signs of {finding}.",
        "Severe {symptom} with hemodynamic instability; immediate attention required.",
        "Sudden onset {symptom} suggestive of {condition}; urgent intervention needed.",
        "Critical {finding} with rapidly deteriorating status; escalate immediately.",
        "Acute {condition} with {symptom}; patient requires emergency care.",
        "Life-threatening {finding}; prioritize for immediate assessment.",
        "Patient unresponsive with {symptom}; initiate emergency protocol.",
        "Severe {symptom} and {finding} consistent with {condition}; act now.",
    ),
}

_VOCAB: dict[str, dict[str, tuple[str, ...]]] = {
    "normal": {
        "symptom": (
            "mild headache",
            "occasional fatigue",
            "a sore throat",
            "mild seasonal allergies",
            "slight muscle soreness",
            "a minor cough",
            "mild back discomfort",
            "transient dizziness",
        ),
        "condition": (
            "well-controlled hypertension",
            "stable thyroid function",
            "a healed minor injury",
            "a benign skin lesion",
            "mild gastritis",
            "a resolving common cold",
            "stable chronic condition",
            "a routine dermatologic concern",
        ),
        "finding": (
            "normal blood pressure",
            "unremarkable laboratory results",
            "a clear chest examination",
            "normal cardiac rhythm",
            "no significant abnormalities",
            "stable weight and vitals",
            "normal reflexes",
            "a healthy baseline",
        ),
    },
    "atencao": {
        "symptom": (
            "intermittent chest discomfort",
            "unexplained weight loss",
            "persistent abdominal pain",
            "recurring migraines",
            "shortness of breath on exertion",
            "elevated blood sugar readings",
            "ongoing joint swelling",
            "progressive fatigue",
        ),
        "condition": (
            "possible early diabetes",
            "moderate anemia",
            "a suspicious pulmonary nodule",
            "borderline hypertension",
            "a thyroid irregularity",
            "a gastrointestinal disorder",
            "an atypical skin lesion",
            "a cardiac arrhythmia",
        ),
        "finding": (
            "a mildly elevated marker",
            "an abnormal electrocardiogram",
            "a borderline imaging result",
            "elevated inflammatory markers",
            "an irregular pulse",
            "a slightly enlarged lymph node",
            "an unexpected lab value",
            "a non-specific opacity on imaging",
        ),
    },
    "urgente": {
        "symptom": (
            "crushing chest pain",
            "severe shortness of breath",
            "profuse bleeding",
            "sudden loss of consciousness",
            "acute abdominal rigidity",
            "slurred speech and facial droop",
            "severe respiratory distress",
            "uncontrolled seizures",
        ),
        "condition": (
            "acute myocardial infarction",
            "a suspected stroke",
            "septic shock",
            "a pulmonary embolism",
            "severe internal hemorrhage",
            "acute respiratory failure",
            "anaphylaxis",
            "a ruptured aneurysm",
        ),
        "finding": (
            "dangerously low blood pressure",
            "critically low oxygen saturation",
            "signs of organ failure",
            "a rapidly falling heart rate",
            "massive active bleeding",
            "acute neurological deficit",
            "severe metabolic acidosis",
            "cardiogenic shock",
        ),
    },
}


def _render(template: str, vocab: dict[str, tuple[str, ...]], rng: random.Random) -> str:
    """Preenche os marcadores de um template com termos sorteados do vocabulário.

    Args:
        template: frase com marcadores ``{symptom}``, ``{condition}``, ``{finding}``.
        vocab: vocabulário da classe (mapa marcador -> opções).
        rng: gerador de números aleatórios determinístico.

    Returns:
        O texto do laudo renderizado.
    """
    return template.format(
        symptom=rng.choice(vocab["symptom"]),
        condition=rng.choice(vocab["condition"]),
        finding=rng.choice(vocab["finding"]),
    )


def generate_samples(n_samples: int, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Gera um dataset sintético balanceado de laudos rotulados por urgência.

    As amostras são distribuídas de forma (quase) uniforme entre as três classes
    de urgência. A geração é determinística para uma dada ``seed`` (R9.7, R9.8).

    Args:
        n_samples: número total de amostras a gerar (deve ser >= 1).
        seed: seed do gerador aleatório (padrão ``DEFAULT_SEED``).

    Returns:
        ``DataFrame`` com exatamente as colunas ``text`` e ``label``.

    Raises:
        ValueError: se ``n_samples`` for menor que 1.
    """
    if n_samples < 1:
        raise ValueError("n_samples deve ser >= 1.")

    rng = random.Random(seed)

    # Distribui as amostras de forma balanceada entre as classes; o resto é
    # alocado às primeiras classes para totalizar exatamente ``n_samples``.
    n_classes = len(URGENCY_CLASSES)
    base = n_samples // n_classes
    remainder = n_samples % n_classes
    per_class = {
        cls: base + (1 if i < remainder else 0)
        for i, cls in enumerate(URGENCY_CLASSES)
    }

    rows: list[dict[str, str]] = []
    for label in URGENCY_CLASSES:
        templates = _TEMPLATES[label]
        vocab = _VOCAB[label]
        for _ in range(per_class[label]):
            template = rng.choice(templates)
            rows.append({"text": _render(template, vocab, rng), "label": label})

    # Embaralha de forma determinística para não agrupar por classe no arquivo.
    rng.shuffle(rows)

    return pd.DataFrame(rows, columns=["text", "label"])


def generate_dataset(
    output_path: Path,
    n_samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Gera o dataset sintético e grava a saída canônica em CSV.

    Args:
        output_path: caminho de saída (ex.: ``data/laudos.csv``).
        n_samples: número total de amostras (padrão ``DEFAULT_SAMPLES``, >= 2000).
        seed: seed do gerador aleatório (padrão ``DEFAULT_SEED``).

    Returns:
        O ``DataFrame`` final gravado (colunas ``text,label``).
    """
    df = generate_samples(n_samples, seed=seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def _print_summary(df: pd.DataFrame, output_path: Path) -> None:
    """Imprime um resumo da distribuição de classes na saída."""
    counts = df["label"].value_counts()
    print(f"Dataset sintético gerado: {len(df)} amostras -> {output_path}")
    print("Distribuição por classe de urgência:")
    for cls in URGENCY_CLASSES:
        print(f"  {cls:8s}: {int(counts.get(cls, 0))}")


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada de linha de comando."""
    parser = argparse.ArgumentParser(
        description=(
            "Gera um dataset sintético (fallback CI/offline) em data/laudos.csv "
            "com colunas text,label e >= 2000 amostras rotuladas em inglês."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/laudos.csv"),
        help="Caminho do CSV de saída (padrão: data/laudos.csv).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help=f"Número de amostras a gerar (padrão: {DEFAULT_SAMPLES}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed do gerador aleatório (padrão: {DEFAULT_SEED}).",
    )
    args = parser.parse_args(argv)

    df = generate_dataset(args.output, n_samples=args.samples, seed=args.seed)
    _print_summary(df, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
