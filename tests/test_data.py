"""Testes do preparo de dados (R9): fallback sintético e mapeamento de urgência.

Cobre, sem acesso à rede:

    - ``generate_samples`` / ``generate_dataset``: determinismo por seed,
      balanceamento entre classes, tamanho mínimo exigido (>= 2.000 amostras)
      e saída canônica ``text,label``;
    - ``map_to_urgency``: mapeamento rótulo do corpus (1..5) -> urgência e
      descarte de linhas inválidas;
    - ``normalize_text``: normalização leve e determinística.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from src.data.generate_dataset import (
    DEFAULT_SAMPLES,
    URGENCY_CLASSES,
    generate_dataset,
    generate_samples,
)
from src.data.prepare_dataset import (
    LABEL_TO_URGENCY,
    SOURCE_LABEL_COLUMN,
    SOURCE_TEXT_COLUMN,
    DatasetError,
    map_to_urgency,
)
from src.model.preprocessing import normalize_text

# ---------------------------------------------------------------------------
# generate_samples / generate_dataset (fallback sintético)
# ---------------------------------------------------------------------------


def test_generate_samples_is_deterministic_for_same_seed() -> None:
    """Mesma seed -> mesmo dataset, linha a linha (R9.7)."""
    first = generate_samples(300, seed=7)
    second = generate_samples(300, seed=7)

    pd.testing.assert_frame_equal(first, second)


def test_generate_samples_differs_for_different_seed() -> None:
    """Seeds diferentes produzem datasets diferentes."""
    assert not generate_samples(300, seed=1).equals(generate_samples(300, seed=2))


@pytest.mark.parametrize("n_samples", [3, 100, 301])
def test_generate_samples_is_balanced(n_samples: int) -> None:
    """As classes diferem em no máximo 1 amostra e o total é exato."""
    df = generate_samples(n_samples)
    counts = df["label"].value_counts()

    assert len(df) == n_samples
    assert set(counts.index) == set(URGENCY_CLASSES)
    assert counts.max() - counts.min() <= 1


def test_generate_samples_has_no_empty_text() -> None:
    """Nenhum laudo gerado é vazio ou só espaços."""
    df = generate_samples(500)
    assert (df["text"].str.strip().str.len() > 0).all()


@pytest.mark.parametrize("n_samples", [0, -5])
def test_generate_samples_rejects_non_positive(n_samples: int) -> None:
    """``n_samples < 1`` é rejeitado com ``ValueError``."""
    with pytest.raises(ValueError):
        generate_samples(n_samples)


def test_generate_dataset_writes_canonical_csv_with_min_samples(tmp_path: Path) -> None:
    """O CSV padrão tem ``text,label`` e >= 2.000 amostras (requisito do desafio)."""
    output = tmp_path / "nested" / "laudos.csv"

    generate_dataset(output)

    df = pd.read_csv(output)
    assert list(df.columns) == ["text", "label"]
    assert len(df) == DEFAULT_SAMPLES >= 2000
    assert set(df["label"]) == set(URGENCY_CLASSES)


# ---------------------------------------------------------------------------
# map_to_urgency (dataset público -> classes de urgência)
# ---------------------------------------------------------------------------


def _raw(rows: list[tuple[object, object]]) -> pd.DataFrame:
    """Monta um ``DataFrame`` no formato do corpus de origem."""
    return pd.DataFrame(rows, columns=[SOURCE_TEXT_COLUMN, SOURCE_LABEL_COLUMN])


def test_label_mapping_covers_all_source_labels_with_valid_classes() -> None:
    """Os 5 rótulos do corpus mapeiam para as 3 classes, todas representadas."""
    assert set(LABEL_TO_URGENCY) == {1, 2, 3, 4, 5}
    assert set(LABEL_TO_URGENCY.values()) == set(URGENCY_CLASSES)


def test_map_to_urgency_maps_each_source_label() -> None:
    """Cada rótulo inteiro vira a classe de urgência correspondente."""
    raw = _raw([(f"abstract {i}", i) for i in range(1, 6)])

    result = map_to_urgency(raw)

    assert list(result.columns) == ["text", "label"]
    assert list(result["label"]) == [LABEL_TO_URGENCY[i] for i in range(1, 6)]


def test_map_to_urgency_strips_text_and_drops_invalid_rows() -> None:
    """Texto vazio/ausente e rótulos fora de 1..5 são descartados."""
    raw = _raw(
        [
            ("  valid neoplasm abstract  ", 1),
            ("", 2),
            ("   ", 3),
            (None, 4),
            ("unknown label", 6),
            ("non numeric label", "x"),
            ("valid general abstract", "5"),
        ]
    )

    result = map_to_urgency(raw)

    assert list(result["text"]) == ["valid neoplasm abstract", "valid general abstract"]
    assert list(result["label"]) == ["urgente", "normal"]


def test_map_to_urgency_raises_when_nothing_valid() -> None:
    """Sem nenhuma linha válida, levanta ``DatasetError``."""
    with pytest.raises(DatasetError):
        map_to_urgency(_raw([("", 1), ("text", 99)]))


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Acute   CHEST pain ", "acute chest pain"),
        ("line1\nline2\tline3", "line1 line2 line3"),
        ("already normal", "already normal"),
        (123, "123"),
    ],
)
def test_normalize_text(raw: object, expected: str) -> None:
    """Minúsculas, espaços colapsados, sem bordas; não-string vira ``str``."""
    assert normalize_text(raw) == expected  # type: ignore[arg-type]
