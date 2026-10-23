"""DAG de treino/retreino do modelo de triagem de laudos médicos (R5).

Orquestra o pipeline de treino reaproveitando — **sem duplicar lógica** — as
funções já implementadas em ``src/model`` e ``src/data``. A DAG expõe o fluxo
exigido pelo enunciado (``load_data → train → save``) detalhado em seis tarefas
coesas (R5.2):

    load_data → preprocess → train → evaluate → export_onnx → save_artifacts

------------------------------------------------------------------------------
Reúso de código (R5.1, decisão de design)
------------------------------------------------------------------------------
Cada ``PythonOperator`` chama uma função de ``src/model/train.py``
(``load_dataset``, ``build_pipeline``/``train_model``, ``evaluate_model``,
``save_artifacts``) ou de ``src/model/export_onnx.py`` (exportação ONNX). A DAG
não reimplementa treino, avaliação nem persistência: ela apenas os encadeia e
transporta o estado entre tarefas via XCom (caminho do CSV e métricas) e via
sistema de arquivos (o pipeline treinado é serializado por ``save_artifacts``).

------------------------------------------------------------------------------
Agendamento e falhas (R5.3, R5.4, R5.5)
------------------------------------------------------------------------------
- ``schedule=None`` → execução **manual** por trigger (R5.5). O agendamento
  periódico é possível trocando ``schedule`` por um cron/preset (ex.:
  ``"@weekly"``), mantido comentado para documentar a opção.
- ``default_args`` com ``retries`` controlados.
- O encadeamento linear (``>>``) faz com que a **falha de qualquer tarefa
  interrompa as dependentes** — comportamento padrão do Airflow (R5.4).
- Em caso de sucesso, ``save_artifacts`` **persiste um novo artefato**
  (``models/model.pkl`` + ``models/metrics.json``), satisfazendo R5.3.

------------------------------------------------------------------------------
Parâmetros do treino
------------------------------------------------------------------------------
A DAG declara ``params`` (``C``, ``max_iter``, ``ngram_max``, ``min_df``,
``test_size``) com os mesmos padrões de ``src.model.train``. Na UI, o
**Trigger DAG** abre um formulário para alterá-los por execução; pela CLI:

    airflow dags trigger train_triage_model --conf '{"C": 0.5, "ngram_max": 3}'

Os valores usados ficam gravados em ``models/metrics.json`` (``hyperparams``
e ``test_size``). O modelo é treinado **uma única vez** (tarefa ``train``);
``save_artifacts`` reaproveita esse artefato em vez de retreinar.

------------------------------------------------------------------------------
Importações
------------------------------------------------------------------------------
``src/model/export_onnx.py`` é entregue por outra tarefa (13.1) e pode ainda
não existir. Para que o **parse da DAG não quebre**, o módulo é importado de
forma **tardia (lazy)** dentro do callable ``export_onnx``: se o módulo (ou o
``skl2onnx``) estiver ausente, a tarefa registra um aviso claro e segue sem
falhar o pipeline — tornando-se plenamente funcional assim que 13.1 for
entregue. As importações de ``src.model.train`` também são feitas dentro dos
callables para manter o parse da DAG leve e livre de trabalho pesado.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

if TYPE_CHECKING:  # pragma: no cover - apenas para type checking
    from airflow.models.taskinstance import TaskInstance

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolução de caminhos do projeto
# ---------------------------------------------------------------------------
# A DAG vive em ``airflow/dags/``; a raiz do projeto está dois níveis acima.
# Adicioná-la ao ``sys.path`` garante que ``import src.model...`` funcione
# independentemente do diretório de trabalho do worker do Airflow.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#: Caminhos padrão dos artefatos, ancorados na raiz do projeto.
DATA_PATH = PROJECT_ROOT / "data" / "laudos.csv"
MODEL_DIR = PROJECT_ROOT / "models"

#: Chaves de XCom usadas para transportar estado entre as tarefas.
XCOM_DATA_PATH = "data_path"
XCOM_ROW_COUNT = "row_count"
XCOM_METRICS = "metrics"

#: Chaves dos ``params`` da DAG repassadas ao pipeline como hiperparâmetros
#: (``test_size`` é tratado à parte, pois controla o split e não o pipeline).
HYPERPARAM_KEYS = ("C", "max_iter", "ngram_max", "min_df")


def _hyperparams(params: dict[str, Any]) -> dict[str, Any]:
    """Extrai dos ``params`` da execução os hiperparâmetros do pipeline."""
    return {key: params[key] for key in HYPERPARAM_KEYS}


# ---------------------------------------------------------------------------
# Callables das tarefas (reutilizam src/model e src/data; sem duplicar lógica)
# ---------------------------------------------------------------------------
def load_data(ti: TaskInstance, **_: Any) -> int:
    """Carrega o dataset de treino (gera o fallback sintético se faltar).

    Reutiliza ``src.model.train.load_dataset``, que já lê ``data/laudos.csv`` e,
    quando ausente, gera o dataset sintético determinístico (CI/offline).

    Publica em XCom o caminho do CSV e o número de linhas para as tarefas
    seguintes.

    Returns:
        O número de amostras carregadas.
    """
    from src.model.train import load_dataset

    df = load_dataset(DATA_PATH)
    row_count = int(len(df))
    logger.info("Dataset carregado: %d amostras de %s", row_count, DATA_PATH)

    ti.xcom_push(key=XCOM_DATA_PATH, value=str(DATA_PATH))
    ti.xcom_push(key=XCOM_ROW_COUNT, value=row_count)
    return row_count


def preprocess(ti: TaskInstance, params: dict[str, Any], **_: Any) -> dict[str, int]:
    """Valida os dados e materializa o pré-processamento TF-IDF (R1.3).

    O pré-processamento do projeto vive no ``TfidfVectorizer`` (ver
    ``src/model/preprocessing.py``), que é parte do ``Pipeline`` serializado —
    não há etapa manual a reproduzir. Esta tarefa confirma, de forma barata e
    determinística, que o vetorizador casa com o corpus carregado: constrói o
    pipeline e ajusta o vetorizador no texto, falhando cedo (R5.4) se o
    vocabulário resultar vazio (ex.: dataset degenerado).

    Returns:
        Um resumo com o tamanho do vocabulário TF-IDF e o nº de amostras.
    """
    from src.model.train import build_pipeline, load_dataset

    df = load_dataset(DATA_PATH)
    # Mesmo vetorizador (ngram_max/min_df) que será usado no treino.
    vectorizer = build_pipeline(_hyperparams(params)).named_steps["tfidf"]
    matrix = vectorizer.fit_transform(df["text"])
    vocab_size = int(len(vectorizer.vocabulary_))

    if vocab_size == 0:
        raise ValueError(
            "Vetorizador TF-IDF produziu vocabulário vazio; "
            "verifique o dataset de entrada."
        )

    logger.info(
        "Pré-processamento validado: vocabulário=%d termos, matriz=%s",
        vocab_size,
        matrix.shape,
    )
    return {"vocab_size": vocab_size, "n_samples": int(len(df))}


def train(ti: TaskInstance, params: dict[str, Any], **_: Any) -> None:
    """Treina o pipeline com os ``params`` da execução e serializa o modelo.

    Reutiliza ``train_model`` (split estratificado reprodutível) e grava o
    artefato do pipeline via ``save_artifacts``, para que ``export_onnx`` e
    ``save_artifacts`` operem sobre **este** modelo treinado (sem retreino).

    Publica as métricas de avaliação em XCom para a tarefa ``evaluate``.
    """
    from src.model.train import (
        evaluate_model,
        save_artifacts,
        train_model,
    )
    from src.model.train import load_dataset as _load_dataset

    hyperparams = _hyperparams(params)
    test_size = params["test_size"]
    logger.info("Treinando com hiperparâmetros=%s | test_size=%s", hyperparams, test_size)

    df = _load_dataset(DATA_PATH)
    pipeline, y_test, y_pred = train_model(df, test_size=test_size, hyperparams=hyperparams)
    metrics = evaluate_model(y_test, y_pred)

    # Persiste o pipeline treinado (model.pkl) já nesta etapa para que as
    # tarefas seguintes tenham o artefato disponível no filesystem.
    model_path, _ = save_artifacts(
        pipeline,
        metrics,
        model_dir=MODEL_DIR,
        hyperparams=hyperparams,
        test_size=test_size,
    )
    logger.info("Modelo treinado e serializado em %s", model_path)

    ti.xcom_push(key=XCOM_METRICS, value=metrics)


def evaluate(ti: TaskInstance, **_: Any) -> dict[str, Any]:
    """Verifica/propaga as métricas de avaliação (accuracy + f1 por classe, R1.6).

    Lê as métricas calculadas em ``train`` via XCom. Reutiliza
    ``evaluate_model`` como fonte da verdade (feito em ``train``); aqui apenas
    valida que as métricas existem e as registra no log para auditoria.

    Returns:
        As métricas de avaliação.
    """
    metrics = ti.xcom_pull(key=XCOM_METRICS, task_ids="train")
    if not metrics:
        raise ValueError("Métricas ausentes no XCom; a tarefa 'train' falhou?")

    logger.info(
        "Avaliação: accuracy=%.4f | f1_macro=%.4f | f1_por_classe=%s",
        metrics["accuracy"],
        metrics["f1_macro"],
        metrics["f1_per_class"],
    )
    return metrics


def export_onnx(**_: Any) -> str:
    """Exporta o pipeline treinado para ONNX (R7.1), de forma tolerante.

    O exportador (``src/model/export_onnx.py``) é entregue pela tarefa 13.1 e
    pode ainda não existir. A importação é **tardia** para não quebrar o parse
    da DAG: se o módulo ou o ``skl2onnx`` estiverem ausentes, a tarefa registra
    um aviso claro e segue sem falhar o pipeline. Assim que 13.1 for entregue,
    esta tarefa passa a exportar ``models/model.onnx`` automaticamente.

    Returns:
        Uma string descrevendo o resultado (caminho exportado ou motivo do skip).
    """
    try:
        from src.model import export_onnx as export_onnx_module
    except ImportError:
        logger.warning(
            "src/model/export_onnx.py ainda não disponível (tarefa 13.1); "
            "pulando a exportação ONNX. O pipeline segue sem falhar."
        )
        return "skipped: export_onnx indisponível"

    # Caminhos explícitos ancorados em MODEL_DIR: os defaults do exportador são
    # relativos ao cwd, que no worker do Airflow (/opt/airflow) não é a raiz do
    # projeto — sem isto o model.pkl não é encontrado e o ONNX não é gerado.
    try:
        result = export_onnx_module.export_onnx(
            model_path=MODEL_DIR / "model.pkl",
            output_path=MODEL_DIR / "model.onnx",
        )
    except Exception:  # noqa: BLE001 - exportação é não-crítica neste estágio
        logger.exception(
            "Falha ao exportar ONNX; o pipeline segue com o artefato .pkl."
        )
        return "skipped: falha na exportação ONNX"

    logger.info("Exportação ONNX concluída: %s", result)
    return f"exported: {result}"


def save_artifacts(ti: TaskInstance, params: dict[str, Any], **_: Any) -> dict[str, str]:
    """Persiste o artefato final do modelo + métricas (R5.3, R1.4).

    Reaproveita o pipeline treinado pela tarefa ``train`` (``models/model.pkl``)
    e as métricas do XCom — **sem retreinar** — e grava ``models/model.pkl`` +
    ``models/metrics.json`` (com os hiperparâmetros da execução) via
    ``save_artifacts`` de ``src.model.train``, garantindo que uma execução
    bem-sucedida da DAG **persista um novo artefato** (R5.3).

    Returns:
        Mapa com os caminhos persistidos do modelo e das métricas.
    """
    import joblib
    from src.model.train import MODEL_FILENAME
    from src.model.train import save_artifacts as _save_artifacts

    metrics = ti.xcom_pull(key=XCOM_METRICS, task_ids="train")
    if not metrics:
        raise ValueError("Métricas ausentes no XCom; a tarefa 'train' falhou?")

    pipeline = joblib.load(MODEL_DIR / MODEL_FILENAME)
    model_path, metrics_path = _save_artifacts(
        pipeline,
        metrics,
        model_dir=MODEL_DIR,
        hyperparams=_hyperparams(params),
        test_size=params["test_size"],
    )

    logger.info(
        "Artefatos persistidos: modelo=%s | métricas=%s", model_path, metrics_path
    )
    return {"model_path": str(model_path), "metrics_path": str(metrics_path)}


# ---------------------------------------------------------------------------
# Definição da DAG
# ---------------------------------------------------------------------------
default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="train_triage_model",
    description="Treino/retreino do modelo de triagem de laudos médicos (R5).",
    default_args=default_args,
    # schedule=None → trigger manual (R5.5). Para agendar, troque por um cron
    # ou preset, ex.: schedule="@weekly".
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "triagem", "treino"],
    # Formulário do "Trigger DAG" na UI (ou --conf na CLI). Padrões iguais a
    # src.model.train.DEFAULT_HYPERPARAMS / DEFAULT_TEST_SIZE (checado em
    # tests/test_dag.py).
    params={
        "C": Param(
            1.0,
            type="number",
            exclusiveMinimum=0,
            description="Inverso da regularização da LogisticRegression "
            "(maior = menos regularização).",
        ),
        "max_iter": Param(
            1000,
            type="integer",
            minimum=1,
            description="Iterações máximas do solver da LogisticRegression.",
        ),
        "ngram_max": Param(
            2,
            type="integer",
            minimum=1,
            maximum=3,
            description="Maior n-grama do TF-IDF (1 = só palavras; 2 = + bigramas).",
        ),
        "min_df": Param(
            2,
            type="integer",
            minimum=1,
            description="Nº mínimo de laudos em que um termo precisa aparecer.",
        ),
        "test_size": Param(
            0.2,
            type="number",
            minimum=0.05,
            maximum=0.5,
            description="Proporção do dataset reservada para avaliação.",
        ),
    },
) as dag:
    load_data_task = PythonOperator(
        task_id="load_data",
        python_callable=load_data,
    )
    preprocess_task = PythonOperator(
        task_id="preprocess",
        python_callable=preprocess,
    )
    train_task = PythonOperator(
        task_id="train",
        python_callable=train,
    )
    evaluate_task = PythonOperator(
        task_id="evaluate",
        python_callable=evaluate,
    )
    export_onnx_task = PythonOperator(
        task_id="export_onnx",
        python_callable=export_onnx,
    )
    save_artifacts_task = PythonOperator(
        task_id="save_artifacts",
        python_callable=save_artifacts,
    )

    # Encadeamento linear: a falha de qualquer tarefa interrompe as dependentes
    # (comportamento padrão do Airflow, R5.4).
    (
        load_data_task
        >> preprocess_task
        >> train_task
        >> evaluate_task
        >> export_onnx_task
        >> save_artifacts_task
    )
