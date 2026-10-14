"""Exportação do pipeline de triagem para ONNX (R7.1).

Converte o **pipeline scikit-learn completo** (``TfidfVectorizer`` +
``LogisticRegression``) persistido em ``models/model.pkl`` para um único grafo
**ONNX** (``models/model.onnx``) via ``skl2onnx``. Vetorizador e classificador
são exportados juntos (o ``skl2onnx`` suporta o pipeline inteiro), evitando
divergência entre o pré-processamento de treino e o de inferência (ver
design.md §1 e §8).

------------------------------------------------------------------------------
Entrada do grafo
------------------------------------------------------------------------------
O pipeline recebe uma coluna de texto. O tipo de entrada ONNX é, portanto, um
tensor de strings ``[None, 1]`` (``StringTensorType``): a primeira dimensão
(batch) é livre e a segunda é a única *feature* textual.

------------------------------------------------------------------------------
Ordem das classes — decisão explícita (ZipMap HABILITADO)
------------------------------------------------------------------------------
Um ponto sutil: o ``LogisticRegression`` do sklearn ordena ``classes_`` de forma
**alfabética** (``atencao``, ``normal``, ``urgente``), que **difere** da ordem
canônica ``URGENCY_CLASSES`` (``normal``, ``atencao``, ``urgente``) usada em
vários lugares do projeto.

Se exportássemos com ``zipmap=False``, a saída de probabilidades viria como um
**array posicional** — e a inferência (``inference._predict_proba_onnx``)
precisaria conhecer a ordem real das classes para rotular corretamente. Como
``inference._load_onnx`` assume ``URGENCY_CLASSES`` (ordem canônica, não
alfabética), zipar um array posicional contra essa ordem **trocaria os
rótulos**.

Por isso exportamos com o **ZipMap HABILITADO** (padrão do ``skl2onnx``): a
saída de probabilidades vem como uma sequência de mapas ``{classe: prob}``,
carregando os **nomes reais das classes dentro do próprio grafo**. Assim o
artefato ``.onnx`` é **auto-descritivo** quanto à ordem das classes e casa com
o ramo de tratamento de ``dict`` em ``inference._predict_proba_onnx`` —
garantindo que ``.pkl`` e ``.onnx`` concordem independentemente da ordem.

**Não** passe opções que desabilitem o ZipMap.

Expõe uma função reutilizável ``export_onnx(...)`` (reaproveitada pela DAG do
Airflow, tarefa 9) e um ``main()``/CLI.

Uso:
    python -m src.model.export_onnx
    python -m src.model.export_onnx --model models/model.pkl --output models/model.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from src.model.train import DEFAULT_MODEL_DIR, MODEL_FILENAME

#: Nome do artefato ONNX exportado (igual ao esperado por ``inference.py``).
ONNX_FILENAME = "model.onnx"

#: Caminhos padrão de entrada (pipeline .pkl) e saída (grafo .onnx).
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / MODEL_FILENAME
DEFAULT_OUTPUT_PATH = DEFAULT_MODEL_DIR / ONNX_FILENAME


def export_onnx(
    model_path: Path | str = DEFAULT_MODEL_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Converte o pipeline sklearn de ``model_path`` para ONNX em ``output_path``.

    Carrega o ``Pipeline`` (TF-IDF + Logistic Regression) serializado via joblib
    e o converte para um único grafo ONNX com ``skl2onnx``, mantendo o **ZipMap
    habilitado** (ver docstring do módulo) para que a saída de probabilidades
    seja auto-descritiva quanto à ordem das classes.

    Args:
        model_path: caminho do pipeline sklearn de entrada (``model.pkl``).
        output_path: caminho de destino do grafo ONNX (``model.onnx``).

    Returns:
        O caminho (``Path``) do artefato ONNX gravado.

    Raises:
        FileNotFoundError: se o pipeline de entrada não existir.
    """
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import StringTensorType

    model_path = Path(model_path)
    output_path = Path(output_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Pipeline de entrada não encontrado: {model_path}. "
            "Treine o modelo primeiro (python -m src.model.train)."
        )

    pipeline = joblib.load(model_path)

    # Entrada: uma coluna de texto -> tensor de strings [None, 1].
    initial_types = [("input", StringTensorType([None, 1]))]

    # ZipMap HABILITADO (padrão do skl2onnx): não passamos options para
    # desabilitá-lo. A saída de probabilidades vem como {classe: prob}.
    onnx_model = to_onnx(pipeline, initial_types=initial_types)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(onnx_model.SerializeToString())

    print(f"Pipeline exportado para ONNX em {output_path}")
    return output_path


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada de linha de comando."""
    parser = argparse.ArgumentParser(
        description=(
            "Exporta o pipeline sklearn (TF-IDF + Logistic Regression) de "
            "models/model.pkl para ONNX (models/model.onnx) via skl2onnx."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Caminho do pipeline .pkl de entrada (padrão: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Caminho do artefato .onnx de saída (padrão: {DEFAULT_OUTPUT_PATH}).",
    )
    args = parser.parse_args(argv)

    export_onnx(model_path=args.model, output_path=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
