"""Testes da DAG ``train_triage_model`` (R5) sem precisar do Airflow instalado.

O Airflow não faz parte do ``requirements.txt`` (é pesado e roda na stack de
``airflow/``). Para testar a DAG no CI, substituímos ``airflow.DAG``,
``airflow.models.param.Param`` e ``airflow.operators.python.PythonOperator``
por *fakes* mínimos que registram as tarefas e o encadeamento ``>>``. Assim
validamos:

    - a estrutura: ``dag_id``, trigger manual, ordem linear das 6 tarefas;
    - os ``params`` da DAG: padrões iguais aos de ``src.model.train``;
    - que ``params`` customizados chegam ao modelo salvo e ao ``metrics.json``;
    - a execução ponta a ponta dos callables (com um ``TaskInstance`` falso
      para o XCom), gerando ``model.pkl``, ``model.onnx`` e ``metrics.json``
      no ``MODEL_DIR`` da DAG — mesmo quando o diretório de trabalho do worker
      não é a raiz do projeto.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import joblib
import pytest
from src.data.generate_dataset import generate_samples
from src.model.train import DEFAULT_HYPERPARAMS, DEFAULT_TEST_SIZE

DAG_FILE = Path(__file__).resolve().parents[1] / "airflow" / "dags" / "train_triage_model.py"

EXPECTED_ORDER = [
    "load_data",
    "preprocess",
    "train",
    "evaluate",
    "export_onnx",
    "save_artifacts",
]


class _FakeDAG:
    """Substituto de ``airflow.DAG``: guarda argumentos e tarefas do contexto."""

    current: _FakeDAG | None = None

    def __init__(self, dag_id: str, **kwargs: Any) -> None:
        self.dag_id = dag_id
        self.kwargs = kwargs
        self.tasks: dict[str, _FakeOperator] = {}

    def __enter__(self) -> _FakeDAG:
        _FakeDAG.current = self
        return self

    def __exit__(self, *exc: object) -> None:
        _FakeDAG.current = None


class _FakeOperator:
    """Substituto de ``PythonOperator``: registra-se na DAG e suporta ``>>``."""

    def __init__(self, task_id: str, python_callable: Any, **_: Any) -> None:
        self.task_id = task_id
        self.python_callable = python_callable
        self.downstream: list[_FakeOperator] = []
        assert _FakeDAG.current is not None, "operador criado fora do contexto da DAG"
        _FakeDAG.current.tasks[task_id] = self

    def __rshift__(self, other: _FakeOperator) -> _FakeOperator:
        self.downstream.append(other)
        return other


class _FakeParam:
    """Substituto de ``airflow.models.param.Param``: guarda valor e schema."""

    def __init__(self, default: Any, **schema: Any) -> None:
        self.value = default
        self.schema = schema


class _FakeTaskInstance:
    """``TaskInstance`` mínimo: XCom em memória."""

    def __init__(self) -> None:
        self.xcom: dict[str, Any] = {}

    def xcom_push(self, key: str, value: Any) -> None:
        self.xcom[key] = value

    def xcom_pull(self, key: str, task_ids: str | None = None) -> Any:
        return self.xcom.get(key)


@pytest.fixture
def dag_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Importa o arquivo da DAG com o Airflow substituído pelos *fakes*."""
    airflow = types.ModuleType("airflow")
    airflow.DAG = _FakeDAG  # type: ignore[attr-defined]
    models = types.ModuleType("airflow.models")
    param = types.ModuleType("airflow.models.param")
    param.Param = _FakeParam  # type: ignore[attr-defined]
    operators = types.ModuleType("airflow.operators")
    python = types.ModuleType("airflow.operators.python")
    python.PythonOperator = _FakeOperator  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "airflow", airflow)
    monkeypatch.setitem(sys.modules, "airflow.models", models)
    monkeypatch.setitem(sys.modules, "airflow.models.param", param)
    monkeypatch.setitem(sys.modules, "airflow.operators", operators)
    monkeypatch.setitem(sys.modules, "airflow.operators.python", python)

    spec = importlib.util.spec_from_file_location("train_triage_model_under_test", DAG_FILE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _linear_order(dag: _FakeDAG) -> list[str]:
    """Percorre a cadeia a partir da única tarefa sem upstream."""
    downstream_ids = {t.task_id for task in dag.tasks.values() for t in task.downstream}
    roots = [task for task in dag.tasks.values() if task.task_id not in downstream_ids]
    assert len(roots) == 1, f"esperada 1 tarefa raiz, encontradas {len(roots)}"

    order, node = [], roots[0]
    while True:
        order.append(node.task_id)
        if not node.downstream:
            return order
        assert len(node.downstream) == 1, f"{node.task_id} ramifica; esperado encadeamento linear"
        node = node.downstream[0]


# ---------------------------------------------------------------------------
# Estrutura
# ---------------------------------------------------------------------------


def test_dag_metadata(dag_module: types.ModuleType) -> None:
    """``dag_id`` esperado, trigger manual e sem *catchup*."""
    dag = dag_module.dag

    assert dag.dag_id == "train_triage_model"
    assert dag.kwargs["schedule"] is None
    assert dag.kwargs["catchup"] is False


def test_dag_tasks_are_chained_linearly(dag_module: types.ModuleType) -> None:
    """As 6 tarefas existem e seguem ingestão -> treino -> exportação -> salvamento."""
    assert _linear_order(dag_module.dag) == EXPECTED_ORDER


def test_dag_param_defaults_match_training_defaults(dag_module: types.ModuleType) -> None:
    """Os padrões do formulário da DAG são os mesmos do treino via CLI."""
    defaults = _default_params(dag_module)

    assert set(defaults) == {*DEFAULT_HYPERPARAMS, "test_size"}
    assert {k: defaults[k] for k in DEFAULT_HYPERPARAMS} == DEFAULT_HYPERPARAMS
    assert defaults["test_size"] == DEFAULT_TEST_SIZE


# ---------------------------------------------------------------------------
# Execução ponta a ponta
# ---------------------------------------------------------------------------


def _default_params(dag_module: types.ModuleType) -> dict[str, Any]:
    """Valores padrão dos ``params`` da DAG (o que o Airflow injeta no contexto)."""
    return {name: param.value for name, param in dag_module.dag.kwargs["params"].items()}


def _run_dag(
    dag_module: types.ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    params: dict[str, Any],
) -> tuple[dict[str, Any], _FakeTaskInstance, Path, Path]:
    """Executa as tarefas em ordem num projeto temporário.

    O diretório de trabalho é deliberadamente diferente da raiz do projeto
    (como no worker do Airflow, onde o cwd é ``/opt/airflow``).

    Returns:
        ``(resultados_por_tarefa, ti, model_dir, worker_cwd)``.
    """
    data_path = tmp_path / "project" / "data" / "laudos.csv"
    model_dir = tmp_path / "project" / "models"
    data_path.parent.mkdir(parents=True)
    generate_samples(300, seed=42).to_csv(data_path, index=False)
    monkeypatch.setattr(dag_module, "DATA_PATH", data_path)
    monkeypatch.setattr(dag_module, "MODEL_DIR", model_dir)
    worker_cwd = tmp_path / "worker_cwd"
    worker_cwd.mkdir()
    monkeypatch.chdir(worker_cwd)

    ti = _FakeTaskInstance()
    results = {
        task_id: dag_module.dag.tasks[task_id].python_callable(ti=ti, params=params)
        for task_id in _linear_order(dag_module.dag)
    }
    return results, ti, model_dir, worker_cwd


def test_dag_run_produces_model_artifacts(
    dag_module: types.ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Executar as tarefas em ordem gera ``model.pkl``, ``model.onnx`` e métricas.

    Todos os artefatos devem ir para o ``MODEL_DIR`` da DAG, não para
    ``./models`` relativo ao cwd do worker.
    """
    results, ti, model_dir, worker_cwd = _run_dag(
        dag_module, tmp_path, monkeypatch, _default_params(dag_module)
    )

    assert results["load_data"] == 300
    assert ti.xcom["row_count"] == 300
    assert results["preprocess"]["vocab_size"] > 0
    assert set(results["evaluate"]["f1_per_class"]) == {"normal", "atencao", "urgente"}
    assert results["export_onnx"].startswith("exported"), results["export_onnx"]

    assert (model_dir / "model.pkl").exists()
    assert (model_dir / "model.onnx").exists()
    metrics = json.loads((model_dir / "metrics.json").read_text())
    assert 0.0 <= metrics["accuracy"] <= 1.0
    # save_artifacts reaproveita o treino de 'train' (sem retreinar).
    assert metrics["accuracy"] == ti.xcom["metrics"]["accuracy"]
    assert not (worker_cwd / "models").exists(), "artefatos vazaram para o cwd do worker"


def test_dag_run_applies_custom_params(
    dag_module: types.ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``params`` do Trigger chegam ao pipeline salvo e ao ``metrics.json``."""
    params = {"C": 0.5, "max_iter": 500, "ngram_max": 1, "min_df": 1, "test_size": 0.3}

    _, _, model_dir, _ = _run_dag(dag_module, tmp_path, monkeypatch, params)

    pipeline = joblib.load(model_dir / "model.pkl")
    assert pipeline.named_steps["tfidf"].ngram_range == (1, 1)
    assert pipeline.named_steps["tfidf"].min_df == 1
    assert pipeline.named_steps["clf"].C == 0.5
    assert pipeline.named_steps["clf"].max_iter == 500

    metrics = json.loads((model_dir / "metrics.json").read_text())
    assert metrics["hyperparams"] == {"C": 0.5, "max_iter": 500, "ngram_max": 1, "min_df": 1}
    assert metrics["test_size"] == 0.3


def test_evaluate_fails_without_metrics_from_train(dag_module: types.ModuleType) -> None:
    """``evaluate`` falha se ``train`` não publicou métricas no XCom."""
    with pytest.raises(ValueError, match="XCom"):
        dag_module.evaluate(ti=_FakeTaskInstance())
