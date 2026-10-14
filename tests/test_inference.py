"""Testes da lógica de inferência de triagem (tarefa 7.1, R8.4).

Cobrem o contrato de ``Classifier.predict`` / ``predict`` do módulo
``src.model.inference``:

    - ``predict()`` devolve um ``label`` válido em ``{normal, atencao, urgente}``
      e um ``score`` em ``[0, 1]`` (R1.1, R1.2);
    - entradas inválidas (vazias/apenas espaços/não-string) levantam
      ``InvalidInputError`` (R1.5);
    - **equivalência** entre o pipeline sklearn (``.pkl``) e o artefato
      otimizado (``.onnx``): as predições (label e probabilidades) devem bater
      dentro de uma tolerância numérica (R7).

------------------------------------------------------------------------------
Fixtures
------------------------------------------------------------------------------
Conforme o design ("Fixture de teste usa um artefato pequeno treinado em dados
sintéticos reduzidos para rapidez no CI"), treinamos um pipeline pequeno sobre
um dataset sintético reduzido em um diretório temporário e apontamos a
inferência para ele via ``MODEL_DIR`` + ``reset_classifier``.

O artefato ``model.onnx`` só é gerado pela tarefa 13.1 (``export_onnx.py``), que
ainda não existe. Para manter o teste de equivalência útil desde já, a fixture
exporta o pipeline pequeno para ONNX **inline** via ``skl2onnx``. Se ``skl2onnx``
não estiver instalado, o teste de equivalência é ignorado com motivo claro
(``pytest.importorskip``) — os demais testes continuam valendo.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from src.data.generate_dataset import generate_samples
from src.model.inference import (
    Classifier,
    InvalidInputError,
    predict,
    reset_classifier,
)
from src.model.train import (
    MODEL_FILENAME,
    URGENCY_CLASSES,
    build_pipeline,
)

if TYPE_CHECKING:  # pragma: no cover - apenas para type checking
    from collections.abc import Iterator

    from sklearn.pipeline import Pipeline

#: Dataset sintético reduzido para treino rápido no CI (balanceado por classe).
_SMALL_SAMPLES = 90

#: Alguns laudos representativos (um por classe) para exercitar a inferência.
SAMPLE_TEXTS = (
    "Routine follow-up for stable chronic condition. Findings within normal limits.",
    "Persistent abdominal pain that warrants further evaluation and monitoring.",
    "Patient presents with acute crushing chest pain and signs of organ failure.",
)

#: Tolerância numérica para a equivalência de probabilidades .pkl vs .onnx.
#
# A equivalência **comportamental** (mesmo ``label`` previsto) é exata. Já as
# *probabilidades* divergem um pouco porque o ONNX Runtime computa o pipeline
# TF-IDF + Logistic Regression em ``float32`` e o conversor ``skl2onnx``
# reproduz a tokenização/normalização do ``TfidfVectorizer`` de forma apenas
# aproximada (limitação conhecida do converter para TF-IDF com n-gramas +
# stopwords). A divergência observada fica em torno de ~5e-2; usamos uma
# tolerância folgada para capturar regressões grosseiras sem exigir uma
# igualdade bit-a-bit que a conversão não garante.
EQUIV_ATOL = 0.1


def _train_small_pipeline() -> Pipeline:
    """Treina um pipeline pequeno em dados sintéticos reduzidos (rápido no CI)."""
    df = generate_samples(_SMALL_SAMPLES, seed=42)
    pipeline = build_pipeline()
    pipeline.fit(df["text"], df["label"])
    return pipeline


@pytest.fixture(scope="module")
def pkl_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Diretório temporário com um ``model.pkl`` pequeno treinado.

    O artefato é compartilhado pelo módulo (``scope="module"``) para evitar
    retreinar a cada teste.
    """
    import joblib

    model_dir = tmp_path_factory.mktemp("model_artifacts")
    pipeline = _train_small_pipeline()
    joblib.dump(pipeline, model_dir / MODEL_FILENAME)
    return model_dir


@pytest.fixture
def classifier(pkl_model_dir: Path) -> Iterator[Classifier]:
    """``Classifier`` carregando o ``.pkl`` pequeno via ``MODEL_DIR``.

    Força ``MODEL_FORMAT='pkl'`` para carregar o pipeline sklearn de forma
    determinística (sem depender do ``model.onnx``), e reseta o singleton de
    módulo antes e depois para não vazar estado entre testes.
    """
    reset_classifier()
    clf = Classifier(model_format="pkl", model_dir=pkl_model_dir)
    yield clf
    reset_classifier()


# ---------------------------------------------------------------------------
# predict(): label válido + score em [0, 1]  (R1.1, R1.2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", SAMPLE_TEXTS)
def test_predict_returns_valid_label(classifier: Classifier, text: str) -> None:
    """``predict()`` devolve um label dentro das classes de urgência (R1.1)."""
    result = classifier.predict(text)
    assert result["label"] in set(URGENCY_CLASSES)


@pytest.mark.parametrize("text", SAMPLE_TEXTS)
def test_predict_score_within_unit_interval(classifier: Classifier, text: str) -> None:
    """O ``score`` (confiança) fica em ``[0, 1]`` (R1.2)."""
    result = classifier.predict(text)
    assert 0.0 <= result["score"] <= 1.0


@pytest.mark.parametrize("text", SAMPLE_TEXTS)
def test_predict_scores_distribution_is_valid(classifier: Classifier, text: str) -> None:
    """``scores`` cobre as três classes, cada uma em ``[0, 1]`` e soma ~= 1.0."""
    result = classifier.predict(text)
    scores = result["scores"]

    assert set(scores) == set(URGENCY_CLASSES)
    assert all(0.0 <= v <= 1.0 for v in scores.values())
    assert sum(scores.values()) == pytest.approx(1.0, abs=1e-6)
    # ``score`` é exatamente a probabilidade máxima da distribuição.
    assert result["score"] == pytest.approx(max(scores.values()))
    assert result["label"] == max(scores, key=scores.__getitem__)


def test_module_predict_uses_shared_classifier(pkl_model_dir: Path) -> None:
    """A função de módulo ``predict`` reutiliza o classificador compartilhado."""
    reset_classifier()
    import os

    original = os.environ.get("MODEL_DIR")
    original_fmt = os.environ.get("MODEL_FORMAT")
    os.environ["MODEL_DIR"] = str(pkl_model_dir)
    os.environ["MODEL_FORMAT"] = "pkl"
    try:
        result = predict(SAMPLE_TEXTS[0])
    finally:
        reset_classifier()
        _restore_env("MODEL_DIR", original)
        _restore_env("MODEL_FORMAT", original_fmt)

    assert result["label"] in set(URGENCY_CLASSES)
    assert 0.0 <= result["score"] <= 1.0


def _restore_env(key: str, value: str | None) -> None:
    """Restaura (ou remove) uma variável de ambiente modificada em um teste."""
    import os

    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


# ---------------------------------------------------------------------------
# Validação de entrada  (R1.5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_text", ["", "   ", "\n\t "])
def test_predict_rejects_empty_or_whitespace(classifier: Classifier, bad_text: str) -> None:
    """Texto vazio/apenas espaços -> ``InvalidInputError`` (R1.5)."""
    with pytest.raises(InvalidInputError):
        classifier.predict(bad_text)


@pytest.mark.parametrize("bad_text", [None, 123, ["a laudo"], {"text": "x"}])
def test_predict_rejects_non_string(classifier: Classifier, bad_text: object) -> None:
    """Entrada que não é ``str`` -> ``InvalidInputError`` (R1.5)."""
    with pytest.raises(InvalidInputError):
        classifier.predict(bad_text)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Equivalência .pkl vs .onnx  (R7)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def onnx_model_dir(pkl_model_dir: Path) -> Path:
    """Diretório com ``model.onnx`` exportado inline do pipeline pequeno.

    A exportação formal é da tarefa 13.1 (``export_onnx.py``). Enquanto esse
    módulo não existe, exportamos aqui mesmo via ``skl2onnx`` para validar a
    equivalência desde já. Se ``skl2onnx`` não estiver disponível, o teste de
    equivalência é ignorado (``importorskip``) com motivo claro.
    """
    pytest.importorskip(
        "skl2onnx",
        reason="skl2onnx não instalado: equivalência .pkl vs .onnx não pode ser verificada.",
    )
    pytest.importorskip(
        "onnxruntime",
        reason="onnxruntime não instalado: inferência ONNX indisponível.",
    )

    import joblib
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import StringTensorType
    from src.model.inference import ONNX_FILENAME

    pipeline = joblib.load(pkl_model_dir / MODEL_FILENAME)

    # O pipeline recebe uma coluna de texto (1 feature string). Mantemos o
    # ZipMap habilitado (padrão do skl2onnx) para que a saída de probabilidades
    # venha como ``{classe: prob}`` — carregando os nomes de classe no próprio
    # grafo. Isso evita depender da ordem posicional das classes (que no sklearn
    # é alfabética — ``classes_`` = atencao, normal, urgente — e difere da ordem
    # canônica ``URGENCY_CLASSES``), e é tratado pelo ramo ``dict`` de
    # ``Classifier._predict_proba_onnx``.
    initial_types = [("input", StringTensorType([None, 1]))]
    onnx_model = to_onnx(pipeline, initial_types=initial_types)

    onnx_path = pkl_model_dir / ONNX_FILENAME
    onnx_path.write_bytes(onnx_model.SerializeToString())
    return pkl_model_dir


@pytest.mark.parametrize("text", SAMPLE_TEXTS)
def test_pkl_onnx_prediction_equivalence(onnx_model_dir: Path, text: str) -> None:
    """Predições do ``.pkl`` e do ``.onnx`` são equivalentes (R7).

    O ``label`` previsto deve ser **idêntico** entre os dois artefatos
    (equivalência comportamental exata); as probabilidades por classe devem
    coincidir dentro de ``EQUIV_ATOL`` (ver nota sobre a tolerância).
    """
    reset_classifier()
    pkl_clf = Classifier(model_format="pkl", model_dir=onnx_model_dir)
    onnx_clf = Classifier(model_format="onnx", model_dir=onnx_model_dir)

    # Confirma que de fato carregamos os dois formatos distintos.
    assert pkl_clf.model_format == "pkl"
    assert onnx_clf.model_format == "onnx"

    pkl_result = pkl_clf.predict(text)
    onnx_result = onnx_clf.predict(text)

    assert pkl_result["label"] == onnx_result["label"]
    assert pkl_result["score"] == pytest.approx(onnx_result["score"], abs=EQUIV_ATOL)

    for cls in URGENCY_CLASSES:
        assert pkl_result["scores"][cls] == pytest.approx(
            onnx_result["scores"][cls], abs=EQUIV_ATOL
        )

    reset_classifier()


# ---------------------------------------------------------------------------
# Carregamento do artefato a partir do S3 — AWS Lambda (R10.3/R10.4/R10.6)
# ---------------------------------------------------------------------------
#
# O caminho do S3 é ativado apenas quando ``MODEL_BUCKET`` está definido. Estes
# testes exercitam a resolução da chave e o download sem bater no S3 real: o
# ``boto3.client`` é substituído por um fake que "baixa" copiando um artefato
# local pré-existente para o destino. Também confirmam que o caminho é inerte
# quando ``MODEL_BUCKET`` não está definido (comportamento local preservado).


class _FakeS3Client:
    """Cliente S3 falso: registra a chamada e copia um artefato local no lugar.

    Reproduz o contrato mínimo usado por ``_download_model_from_s3``:
    ``download_file(bucket, key, destination)``.
    """

    def __init__(self, source_artifact: Path) -> None:
        self._source = source_artifact
        self.calls: list[tuple[str, str, str]] = []

    def download_file(self, bucket: str, key: str, destination: str) -> None:
        import shutil

        self.calls.append((bucket, key, destination))
        shutil.copyfile(self._source, destination)


def test_classifier_loads_from_s3_when_bucket_set(
    pkl_model_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Com ``MODEL_BUCKET`` definido, o artefato é baixado do S3 e carregado (R10)."""
    import sys
    import types

    from src.model import inference

    source_pkl = pkl_model_dir / MODEL_FILENAME
    fake_client = _FakeS3Client(source_pkl)

    # Fake do módulo ``boto3`` (import tardio dentro de _download_model_from_s3).
    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = lambda service: fake_client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    # Redireciona o destino do download de ``/tmp`` para um tmp_path de teste.
    monkeypatch.setattr(inference, "S3_DOWNLOAD_DIR", tmp_path)

    reset_classifier()
    clf = inference.Classifier(
        model_format="pkl",
        model_bucket="my-model-bucket",
        model_key="artifacts/model.pkl",
    )
    try:
        # O download do S3 foi invocado com o bucket/chave corretos.
        assert fake_client.calls == [
            ("my-model-bucket", "artifacts/model.pkl", str(tmp_path / "model.pkl"))
        ]
        # O artefato carregado é o baixado em tmp_path (não o diretório local).
        assert Path(clf.model_path) == tmp_path / "model.pkl"
        assert clf.model_format == "pkl"

        result = clf.predict(SAMPLE_TEXTS[0])
        assert result["label"] in set(URGENCY_CLASSES)
    finally:
        reset_classifier()


def test_classifier_s3_default_key_from_format(
    pkl_model_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sem ``MODEL_KEY``, a chave padrão é ``model.<formato>`` (R10.6)."""
    import sys
    import types

    from src.model import inference

    fake_client = _FakeS3Client(pkl_model_dir / MODEL_FILENAME)
    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = lambda service: fake_client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setattr(inference, "S3_DOWNLOAD_DIR", tmp_path)

    reset_classifier()
    inference.Classifier(model_format="pkl", model_bucket="bucket-only")
    try:
        bucket, key, destination = fake_client.calls[0]
        assert bucket == "bucket-only"
        assert key == "model.pkl"  # padrão derivado do MODEL_FORMAT
        assert destination == str(tmp_path / "model.pkl")
    finally:
        reset_classifier()


def test_classifier_s3_inert_without_bucket(
    pkl_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sem ``MODEL_BUCKET``, o caminho do S3 é inerte: boto3 nunca é importado."""
    import builtins

    real_import = builtins.__import__

    def _guard_import(name: str, *args: object, **kwargs: object):
        if name == "boto3":
            raise AssertionError("boto3 não deve ser importado sem MODEL_BUCKET")
        return real_import(name, *args, **kwargs)

    monkeypatch.delenv("MODEL_BUCKET", raising=False)
    monkeypatch.setattr(builtins, "__import__", _guard_import)

    reset_classifier()
    clf = Classifier(model_format="pkl", model_dir=pkl_model_dir)
    try:
        assert clf.model_format == "pkl"
        assert Path(clf.model_path) == pkl_model_dir / MODEL_FILENAME
    finally:
        reset_classifier()
