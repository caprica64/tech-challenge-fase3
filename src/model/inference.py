"""Inferência de triagem de laudos médicos (R1.1, R1.2, R1.5).

Carrega o artefato do modelo treinado e expõe ``predict(text)`` devolvendo a
classe de urgência, o score de confiança e a distribuição completa de
probabilidades por classe:

    predict(text) -> {"label": str, "score": float, "scores": dict[str, float]}

------------------------------------------------------------------------------
Seleção do artefato (ONNX por padrão, fallback ``.pkl``)
------------------------------------------------------------------------------
Por decisão de design (seção "Pipeline de ML"), a inferência prefere o modelo
otimizado em **ONNX Runtime** (``models/model.onnx``) e faz *fallback* para o
pipeline scikit-learn serializado (``models/model.pkl`` via joblib) quando o
ONNX não está disponível. Isso mantém o mesmo comportamento entre o modelo
original e o otimizado (R7) e permite que a inferência funcione **antes** de a
exportação ONNX existir (tarefa 13.1): se ``model.onnx`` estiver ausente — ou
se o ``onnxruntime`` não estiver instalado — o carregamento recai
automaticamente no ``.pkl``.

A seleção pode ser controlada por variáveis de ambiente (iguais entre ambiente
local e nuvem, conforme o design de deploy AWS):

    - ``MODEL_FORMAT``  — ``onnx`` (padrão, com fallback) ou ``pkl`` (força o
      pipeline sklearn).
    - ``MODEL_DIR``     — diretório dos artefatos (padrão: ``models``).
    - ``MODEL_PATH``    — caminho explícito de um artefato específico
      (``.onnx`` ou ``.pkl``); tem precedência sobre ``MODEL_DIR``/``MODEL_FORMAT``.

------------------------------------------------------------------------------
Carregamento do artefato a partir do S3 (AWS Lambda — R10.3/R10.4/R10.6)
------------------------------------------------------------------------------
Para o deploy em AWS Lambda (imagem de container), o modelo **não** é embutido
na imagem: ele é baixado do **Amazon S3** no *cold start* e reutilizado entre
invocações *warm*. Isso é controlado por variáveis de ambiente passadas pela
Lambda (parametrizadas pelo SAM, nunca hardcoded):

    - ``MODEL_BUCKET`` — nome do bucket S3 com o artefato. **Quando definido**,
      ativa o caminho de download do S3; quando **ausente**, o comportamento
      local acima permanece inalterado.
    - ``MODEL_KEY``    — chave (``key``) do objeto no bucket (ex.: ``model.onnx``).
      Se ausente, assume-se ``model.<MODEL_FORMAT>`` (ex.: ``model.onnx``).
    - ``MODEL_FORMAT`` — ``onnx`` (padrão) ou ``pkl`` — define também a extensão
      do arquivo baixado em ``/tmp`` e como ele é carregado.

No *cold start*, o artefato é baixado uma única vez para ``/tmp`` (único
diretório gravável na Lambda) e, a partir daí, o carregamento segue exatamente
a mesma lógica local (via ``MODEL_PATH`` apontando para o arquivo em ``/tmp``).
Em invocações *warm*, o ``Classifier`` já instanciado (escopo global via o
``lifespan`` do FastAPI sob ``mangum``) é reutilizado, sem novo download.

O ``boto3`` é importado **de forma tardia (lazy)**, apenas quando o caminho do
S3 é efetivamente usado — assim ambientes locais/CI sem ``boto3`` instalado não
são afetados. O ``boto3`` também é listado em ``requirements.txt`` para que a
imagem Lambda (que instala ``requirements.txt``) e testes locais do caminho S3
tenham a dependência disponível.

------------------------------------------------------------------------------
Validação de entrada (R1.5)
------------------------------------------------------------------------------
Texto vazio, só espaços em branco ou de tipo inválido são rejeitados com
``InvalidInputError`` — uma exceção dedicada que a camada de API traduz em um
erro de validação HTTP (422). A inferência em si nunca é chamada com entrada
inválida.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import joblib

from src.model.preprocessing import normalize_text
from src.model.train import (
    DEFAULT_MODEL_DIR,
    MODEL_FILENAME,
    URGENCY_CLASSES,
)

if TYPE_CHECKING:  # pragma: no cover - apenas para type checking
    from collections.abc import Sequence

#: Nome do artefato ONNX exportado (ver tarefa 13.1 / ``export_onnx.py``).
ONNX_FILENAME = "model.onnx"

#: Formato padrão de carregamento: ONNX preferencial, com fallback para pkl.
DEFAULT_MODEL_FORMAT = "onnx"

#: Diretório gravável usado como destino do download do S3 na Lambda.
#: ``/tmp`` é o único sistema de arquivos gravável no runtime da Lambda.
S3_DOWNLOAD_DIR = Path("/tmp")  # noqa: S108 - destino intencional (Lambda writable fs)


class InvalidInputError(ValueError):
    """Entrada de inferência inválida (texto vazio, em branco ou não-string).

    Herda de ``ValueError`` para integração natural, mas é um tipo dedicado
    para que a camada de API possa distingui-la e devolver um erro de
    validação (HTTP 422) — R1.5.
    """


class ModelNotLoadedError(RuntimeError):
    """Nenhum artefato de modelo pôde ser carregado (``.onnx`` nem ``.pkl``)."""


def _validate_text(text: str) -> str:
    """Valida e normaliza o texto de entrada (R1.5).

    Args:
        text: o laudo em texto submetido para classificação.

    Returns:
        O texto normalizado (ver ``preprocessing.normalize_text``), já
        garantidamente não-vazio.

    Raises:
        InvalidInputError: se ``text`` não for ``str`` ou ficar vazio após
            remover espaços em branco.
    """
    if not isinstance(text, str):
        raise InvalidInputError(
            f"Texto de entrada deve ser uma string, recebido {type(text).__name__}."
        )
    if not text.strip():
        raise InvalidInputError("Texto de entrada vazio ou apenas espaços em branco.")
    return normalize_text(text)


def _resolve_artifact(
    *,
    model_format: str,
    model_dir: Path,
    model_path: str | None,
) -> tuple[str, Path]:
    """Resolve qual artefato carregar, aplicando o fallback ONNX -> pkl.

    Precedência:
        1. ``model_path`` explícito (extensão define o formato).
        2. ``model_format='pkl'`` -> ``model_dir/model.pkl``.
        3. ``model_format='onnx'`` -> ``model_dir/model.onnx`` se existir e o
           ``onnxruntime`` estiver instalado; caso contrário, fallback para
           ``model_dir/model.pkl``.

    Returns:
        Tupla ``(formato_resolvido, caminho_do_artefato)`` com formato em
        ``{"onnx", "pkl"}``.

    Raises:
        ModelNotLoadedError: se nenhum artefato utilizável for encontrado.
    """
    if model_path:
        path = Path(model_path)
        if not path.exists():
            raise ModelNotLoadedError(f"Artefato de modelo não encontrado: {path}")
        fmt = "onnx" if path.suffix == ".onnx" else "pkl"
        return fmt, path

    pkl_path = model_dir / MODEL_FILENAME

    if model_format == "pkl":
        if not pkl_path.exists():
            raise ModelNotLoadedError(f"Artefato de modelo não encontrado: {pkl_path}")
        return "pkl", pkl_path

    # model_format == "onnx" (padrão): preferir ONNX, cair para pkl.
    onnx_path = model_dir / ONNX_FILENAME
    if onnx_path.exists() and _onnxruntime_available():
        return "onnx", onnx_path

    if pkl_path.exists():
        return "pkl", pkl_path

    raise ModelNotLoadedError(
        f"Nenhum artefato de modelo encontrado em {model_dir} "
        f"({ONNX_FILENAME} ou {MODEL_FILENAME})."
    )


def _onnxruntime_available() -> bool:
    """Indica se o ``onnxruntime`` pode ser importado (import tardio)."""
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def _download_model_from_s3(
    *,
    bucket: str,
    key: str,
    destination: Path,
) -> Path:
    """Baixa o artefato do modelo do S3 para o sistema de arquivos local.

    Executado no *cold start* da Lambda (escopo global), uma única vez por
    container; invocações *warm* reutilizam o ``Classifier`` já carregado e não
    chamam esta função novamente (R10.3/R10.4/R10.6).

    O ``boto3`` é importado **de forma tardia** aqui dentro, de modo que apenas
    ambientes que efetivamente usam o caminho do S3 (a imagem Lambda) precisem
    da dependência — ambientes locais/CII sem ``boto3`` não são afetados,
    porque esta função só é chamada quando ``MODEL_BUCKET`` está definido.

    Args:
        bucket: nome do bucket S3.
        key: chave (``key``) do objeto no bucket.
        destination: caminho local de destino (criado se necessário).

    Returns:
        O ``destination`` após o download concluído.
    """
    import boto3  # import tardio: só necessário no caminho S3 (Lambda)

    destination.parent.mkdir(parents=True, exist_ok=True)
    s3_client = boto3.client("s3")
    s3_client.download_file(bucket, key, str(destination))
    return destination


def _resolve_s3_model_path(
    *,
    bucket: str,
    key: str | None,
    model_format: str,
    download_dir: Path,
) -> Path:
    """Resolve a chave/destino e baixa o artefato do S3, devolvendo o caminho local.

    A chave do objeto é ``key`` quando fornecida; caso contrário, assume-se
    ``model.<model_format>`` (ex.: ``model.onnx``). O arquivo baixado preserva
    a extensão correta em ``download_dir`` para que a lógica de carregamento
    local (``_resolve_artifact`` via ``MODEL_PATH``) identifique o formato.

    Args:
        bucket: nome do bucket S3 (``MODEL_BUCKET``).
        key: chave do objeto (``MODEL_KEY``) ou ``None`` para o padrão.
        model_format: ``"onnx"`` ou ``"pkl"``.
        download_dir: diretório gravável de destino (``/tmp`` na Lambda).

    Returns:
        Caminho local do artefato baixado.
    """
    ext = "onnx" if model_format == "onnx" else "pkl"
    resolved_key = key or f"model.{ext}"
    # Nome local: usa o basename da chave para preservar a extensão real.
    local_name = Path(resolved_key).name or f"model.{ext}"
    destination = download_dir / local_name
    return _download_model_from_s3(bucket=bucket, key=resolved_key, destination=destination)


class Classifier:
    """Classificador de urgência carregado a partir de um artefato treinado.

    Encapsula o modelo (ONNX ou sklearn) e expõe ``predict``. A API deve
    instanciar **uma vez** no startup (R2.6) e reutilizar entre requisições.

    Attributes:
        model_format: ``"onnx"`` ou ``"pkl"`` — formato efetivamente carregado.
        model_path: caminho do artefato carregado.
        model_version: identificador de versão para expor na API.
        classes: ordem canônica das classes de urgência do modelo.
    """

    def __init__(
        self,
        *,
        model_format: str | None = None,
        model_dir: Path | str | None = None,
        model_path: str | None = None,
        model_bucket: str | None = None,
        model_key: str | None = None,
    ) -> None:
        fmt = (model_format or os.getenv("MODEL_FORMAT") or DEFAULT_MODEL_FORMAT).lower()
        resolved_dir = Path(model_dir or os.getenv("MODEL_DIR") or DEFAULT_MODEL_DIR)
        resolved_path = model_path or os.getenv("MODEL_PATH")

        # Caminho AWS Lambda: quando ``MODEL_BUCKET`` está definido (e nenhum
        # ``MODEL_PATH`` explícito tem precedência), baixa o artefato do S3 para
        # ``/tmp`` no cold start e carrega a partir de lá. Inerte localmente,
        # pois só é ativado quando ``MODEL_BUCKET`` existe (R10.3/R10.4/R10.6).
        bucket = model_bucket or os.getenv("MODEL_BUCKET")
        if bucket and not resolved_path:
            key = model_key or os.getenv("MODEL_KEY")
            resolved_path = str(
                _resolve_s3_model_path(
                    bucket=bucket,
                    key=key,
                    model_format=fmt,
                    download_dir=S3_DOWNLOAD_DIR,
                )
            )

        self.model_format, self.model_path = _resolve_artifact(
            model_format=fmt,
            model_dir=resolved_dir,
            model_path=resolved_path,
        )

        self._pipeline = None
        self._session = None
        self._onnx_output_names: list[str] = []

        if self.model_format == "onnx":
            self._load_onnx()
        else:
            self._load_pkl()

        self.model_version = f"{self.model_format}-v1"

    # -- carregamento de artefatos ------------------------------------------

    def _load_pkl(self) -> None:
        """Carrega o pipeline sklearn serializado via joblib."""
        self._pipeline = joblib.load(self.model_path)
        classes = getattr(self._pipeline, "classes_", None)
        self.classes: tuple[str, ...] = (
            tuple(str(c) for c in classes) if classes is not None else URGENCY_CLASSES
        )

    def _load_onnx(self) -> None:
        """Carrega a ``InferenceSession`` do ONNX Runtime."""
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )
        self._onnx_output_names = [o.name for o in self._session.get_outputs()]
        self.classes = URGENCY_CLASSES

    # -- predição -----------------------------------------------------------

    def predict(self, text: str) -> dict:
        """Classifica um laudo de texto (R1.1, R1.2, R1.5).

        Args:
            text: o laudo em texto a classificar.

        Returns:
            Dicionário ``{"label", "score", "scores"}`` onde:
                - ``label``: classe prevista em ``{normal, atencao, urgente}``;
                - ``score``: probabilidade máxima (confiança) em ``[0, 1]``;
                - ``scores``: mapa classe -> probabilidade (soma ~= 1.0).

        Raises:
            InvalidInputError: se o texto for vazio/inválido (R1.5).
        """
        normalized = _validate_text(text)

        if self.model_format == "onnx":
            scores = self._predict_proba_onnx(normalized)
        else:
            scores = self._predict_proba_pkl(normalized)

        label = max(scores, key=scores.__getitem__)
        return {
            "label": label,
            "score": scores[label],
            "scores": scores,
        }

    def _predict_proba_pkl(self, text: str) -> dict[str, float]:
        """Probabilidades por classe via ``predict_proba`` do pipeline sklearn."""
        proba = self._pipeline.predict_proba([text])[0]
        return _zip_scores(self.classes, proba)

    def _predict_proba_onnx(self, text: str) -> dict[str, float]:
        """Probabilidades por classe via sessão ONNX Runtime.

        O pipeline sklearn exportado por ``skl2onnx`` produz duas saídas: os
        rótulos (``label``) e as probabilidades (``probabilities``). Esta última
        é retornada como lista de dicionários (uma por amostra) ou como um array
        2D, dependendo da versão do conversor; ambos os formatos são tratados.
        """
        inputs = {self._session.get_inputs()[0].name: [[text]]}
        outputs = self._session.run(self._onnx_output_names, inputs)

        proba_output = self._select_proba_output(outputs)
        row = proba_output[0]

        if isinstance(row, dict):
            # skl2onnx (ZipMap=True): {classe: probabilidade}
            return {str(cls): float(score) for cls, score in row.items()}

        # ZipMap desativado: array alinhado a self.classes.
        return _zip_scores(self.classes, row)

    def _select_proba_output(self, outputs: Sequence) -> Sequence:
        """Seleciona a saída de probabilidades entre as saídas do grafo ONNX."""
        for name, value in zip(self._onnx_output_names, outputs, strict=True):
            if "prob" in name.lower():
                return value
        # Fallback: a última saída costuma ser a de probabilidades.
        return outputs[-1]


def _zip_scores(classes: Sequence[str], proba: Sequence[float]) -> dict[str, float]:
    """Associa classes a probabilidades em um dicionário ``{classe: score}``."""
    return {str(cls): float(score) for cls, score in zip(classes, proba, strict=True)}


# ---------------------------------------------------------------------------
# Singleton de conveniência para a API (carregar uma vez, reutilizar).
# ---------------------------------------------------------------------------
_classifier: Classifier | None = None
_classifier_lock = threading.Lock()


def get_classifier() -> Classifier:
    """Devolve o ``Classifier`` compartilhado, carregando-o na primeira chamada.

    Thread-safe (double-checked locking). A API deve chamar isto no evento de
    ``lifespan`` para carregar o modelo **uma vez** (R2.6).
    """
    global _classifier
    if _classifier is None:
        with _classifier_lock:
            if _classifier is None:
                _classifier = Classifier()
    return _classifier


def predict(text: str) -> dict:
    """Classifica ``text`` usando o classificador compartilhado (R1.1, R1.2).

    Função de conveniência de nível de módulo que reutiliza o ``Classifier``
    carregado por ``get_classifier``.

    Args:
        text: o laudo em texto a classificar.

    Returns:
        Dicionário ``{"label", "score", "scores"}`` (ver ``Classifier.predict``).

    Raises:
        InvalidInputError: se o texto for vazio/inválido (R1.5).
    """
    return get_classifier().predict(text)


def reset_classifier() -> None:
    """Descarta o classificador em cache (útil para testes/recarga de modelo)."""
    global _classifier
    with _classifier_lock:
        _classifier = None
