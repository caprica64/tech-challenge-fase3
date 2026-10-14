"""Preparo do dataset público para a triagem de laudos médicos.

Fonte primária (R9.1, R9.2): **Medical Abstracts TC Corpus**
(https://github.com/sebischair/Medical-Abstracts-TC-Corpus), um dataset público
em inglês de classificação de textos médicos, com > 2.000 amostras (R9.3) e
licença Creative Commons.

Estrutura dos arquivos de origem (CSV):
    colunas: ``condition_label,medical_abstract``
    - ``medical_abstract``: o texto do laudo/abstract (inglês).
    - ``condition_label``: inteiro de 1 a 5 identificando a condição clínica.

Rótulos originais do corpus (``medical_tc_labels.csv``):
    1 -> neoplasms
    2 -> digestive system diseases
    3 -> nervous system diseases
    4 -> cardiovascular diseases
    5 -> general pathological conditions

------------------------------------------------------------------------------
Mapeamento de rótulos (R9.4)
------------------------------------------------------------------------------
O corpus classifica *condições clínicas*, não *urgência*. Como o sistema de
triagem opera em três níveis de urgência ``{normal, atencao, urgente}``, derivamos
a urgência a partir da gravidade/acuidade típica de cada classe de condição. A
regra é uma heurística didática (o foco do projeto é o ciclo de MLOps, não a
acurácia clínica) e está centralizada e documentada aqui:

    urgente  <- neoplasms (1), cardiovascular diseases (4)
        Condições de alta acuidade e potencialmente fatais (câncer, eventos
        cardiovasculares) que, em um fluxo de triagem, demandam priorização
        imediata.

    atencao  <- digestive system diseases (2), nervous system diseases (3)
        Condições relevantes que normalmente requerem avaliação, porém com
        menor risco imediato de vida que o grupo "urgente".

    normal   <- general pathological conditions (5)
        Categoria ampla e inespecífica de condições patológicas gerais, tratada
        como baixa urgência / rotina de triagem.

A natureza dos dados (R9.5): **dados públicos** (Medical Abstracts TC Corpus),
sem PII real — adequado ao projeto sem necessidade de anonimização adicional.

------------------------------------------------------------------------------
Saída
------------------------------------------------------------------------------
Grava a saída canônica em ``data/laudos.csv`` com as colunas ``text,label``
(R9, modelo de dados do design), onde ``label`` é uma das três classes de
urgência.

Observação: este script foca no caminho do **dataset público**. Se os dados não
puderem ser carregados/baixados, ele falha de forma clara (ver ``DatasetError``).
O fallback sintético para CI/offline é tratado em ``generate_dataset.py``
(tarefa 2.2), separadamente.

Uso:
    python -m src.data.prepare_dataset
    python -m src.data.prepare_dataset --output data/laudos.csv
"""

from __future__ import annotations

import argparse
import io
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd


def _build_ssl_context() -> ssl.SSLContext | None:
    """Cria um contexto SSL usando o bundle de CAs do ``certifi``, se disponível.

    Em alguns ambientes (ex.: macOS com Python standalone) o Python não encontra
    o bundle de certificados do sistema, causando ``CERTIFICATE_VERIFY_FAILED``.
    Quando ``certifi`` está instalado, usamos seu bundle para a verificação TLS;
    caso contrário, retornamos ``None`` para usar o contexto padrão.
    """
    try:
        import certifi
    except ImportError:
        return None
    return ssl.create_default_context(cafile=certifi.where())

# ---------------------------------------------------------------------------
# Constantes do dataset público
# ---------------------------------------------------------------------------

#: URLs brutas dos CSVs de treino e teste do Medical Abstracts TC Corpus.
_RAW_BASE_URL = (
    "https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main"
)
DATASET_URLS: tuple[str, ...] = (
    f"{_RAW_BASE_URL}/medical_tc_train.csv",
    f"{_RAW_BASE_URL}/medical_tc_test.csv",
)

#: Nome da coluna de texto no corpus de origem.
SOURCE_TEXT_COLUMN = "medical_abstract"
#: Nome da coluna de rótulo (inteiro 1..5) no corpus de origem.
SOURCE_LABEL_COLUMN = "condition_label"

#: Classes de urgência válidas na saída canônica.
URGENCY_CLASSES: tuple[str, ...] = ("normal", "atencao", "urgente")

#: Mapeamento do rótulo inteiro do corpus (1..5) -> classe de urgência.
#: Ver docstring do módulo para a justificativa clínica/didática do mapeamento.
LABEL_TO_URGENCY: dict[int, str] = {
    1: "urgente",  # neoplasms
    2: "atencao",  # digestive system diseases
    3: "atencao",  # nervous system diseases
    4: "urgente",  # cardiovascular diseases
    5: "normal",  # general pathological conditions
}

#: Timeout (segundos) para o download de cada arquivo do dataset.
_DOWNLOAD_TIMEOUT = 60


class DatasetError(RuntimeError):
    """Erro ao obter/preparar o dataset público.

    Levantado quando o dataset não pode ser baixado ou está em formato
    inesperado. O preparo falha de forma explícita (sem fallback silencioso);
    o fallback sintético é responsabilidade de ``generate_dataset.py``.
    """


def _download_csv(url: str) -> pd.DataFrame:
    """Baixa um CSV do dataset público e o carrega em um ``DataFrame``.

    Args:
        url: URL bruta do arquivo CSV.

    Returns:
        O ``DataFrame`` carregado.

    Raises:
        DatasetError: se o download falhar (rede/HTTP) ou o conteúdo não puder
            ser interpretado como CSV.
    """
    try:
        context = _build_ssl_context()
        with urllib.request.urlopen(  # noqa: S310
            url, timeout=_DOWNLOAD_TIMEOUT, context=context
        ) as response:
            raw = response.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        raise DatasetError(
            f"Não foi possível baixar o dataset público em {url}: {exc}. "
            "Verifique a conectividade de rede ou use o fallback sintético "
            "(src/data/generate_dataset.py) para ambientes de CI/offline."
        ) from exc

    try:
        return pd.read_csv(io.BytesIO(raw))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
        raise DatasetError(
            f"Conteúdo baixado de {url} não pôde ser interpretado como CSV: {exc}."
        ) from exc


def load_raw_dataset(urls: tuple[str, ...] = DATASET_URLS) -> pd.DataFrame:
    """Carrega e concatena os splits brutos do Medical Abstracts TC Corpus.

    Args:
        urls: URLs dos CSVs de origem (por padrão, treino + teste do corpus).

    Returns:
        ``DataFrame`` concatenado com as colunas de origem do corpus.

    Raises:
        DatasetError: se nenhum arquivo puder ser carregado ou se as colunas
            esperadas estiverem ausentes.
    """
    frames = [_download_csv(url) for url in urls]
    if not frames:
        raise DatasetError("Nenhuma URL de dataset informada.")

    raw = pd.concat(frames, ignore_index=True)

    missing = {SOURCE_TEXT_COLUMN, SOURCE_LABEL_COLUMN} - set(raw.columns)
    if missing:
        raise DatasetError(
            f"Dataset de origem não contém as colunas esperadas {sorted(missing)}. "
            f"Colunas encontradas: {list(raw.columns)}."
        )

    return raw


def map_to_urgency(raw: pd.DataFrame) -> pd.DataFrame:
    """Mapeia o dataset bruto para a saída canônica ``text,label`` (R9.4).

    Aplica ``LABEL_TO_URGENCY`` sobre a coluna de rótulo inteiro e normaliza o
    texto (remoção de espaços nas pontas), descartando linhas com texto vazio ou
    com rótulo fora de 1..5.

    Args:
        raw: ``DataFrame`` cru com as colunas do corpus de origem.

    Returns:
        ``DataFrame`` com exatamente as colunas ``text`` e ``label``.

    Raises:
        DatasetError: se nenhuma amostra válida restar após o mapeamento.
    """
    text = raw[SOURCE_TEXT_COLUMN].astype("string").str.strip()
    labels = pd.to_numeric(raw[SOURCE_LABEL_COLUMN], errors="coerce").map(LABEL_TO_URGENCY)

    df = pd.DataFrame({"text": text, "label": labels})

    # Remove entradas com texto vazio/ausente (R1.5 na origem dos dados) e
    # rótulos fora de 1..5 (que não têm mapeamento de urgência).
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.len() > 0]

    result = df[["text", "label"]].reset_index(drop=True)
    if result.empty:
        raise DatasetError(
            "Nenhuma amostra válida após o mapeamento de rótulos. "
            "Verifique o formato do dataset de origem."
        )
    return result


def prepare_dataset(output_path: Path, urls: tuple[str, ...] = DATASET_URLS) -> pd.DataFrame:
    """Prepara o dataset público e grava a saída canônica em CSV.

    Args:
        output_path: caminho de saída (ex.: ``data/laudos.csv``).
        urls: URLs dos CSVs de origem.

    Returns:
        O ``DataFrame`` final gravado (colunas ``text,label``).

    Raises:
        DatasetError: se o dataset não puder ser obtido/preparado.
    """
    raw = load_raw_dataset(urls)
    df = map_to_urgency(raw)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def _print_summary(df: pd.DataFrame, output_path: Path) -> None:
    """Imprime um resumo da distribuição de classes na saída."""
    counts = df["label"].value_counts()
    print(f"Dataset preparado: {len(df)} amostras -> {output_path}")
    print("Distribuição por classe de urgência:")
    for cls in URGENCY_CLASSES:
        print(f"  {cls:8s}: {int(counts.get(cls, 0))}")


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada de linha de comando."""
    parser = argparse.ArgumentParser(
        description=(
            "Prepara o dataset público (Medical Abstracts TC Corpus) em "
            "data/laudos.csv com colunas text,label."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/laudos.csv"),
        help="Caminho do CSV de saída (padrão: data/laudos.csv).",
    )
    args = parser.parse_args(argv)

    try:
        df = prepare_dataset(args.output)
    except DatasetError as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1

    _print_summary(df, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
