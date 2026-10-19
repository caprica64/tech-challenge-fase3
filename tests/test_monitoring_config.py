"""Testes da configuração da stack de monitoramento (R6.3–R6.6).

Validam, sem subir containers, que os arquivos de ``monitoring/`` e o
``docker-compose.yml`` estão coerentes entre si e com a API:

    - o dashboard do Grafana tem os >= 3 painéis exigidos (requisições,
      latência, taxa de erro) e aponta para o datasource provisionado;
    - toda métrica usada nas queries do dashboard é de fato exposta pela API
      em ``/metrics`` (evita painéis vazios por nome de métrica errado);
    - o Prometheus faz scrape do serviço/porta da API definidos no Compose.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from src.api.main import app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MONITORING = PROJECT_ROOT / "monitoring"
DASHBOARD_PATH = MONITORING / "grafana" / "dashboards" / "triagem.json"
DATASOURCE_PATH = MONITORING / "grafana" / "provisioning" / "datasources" / "datasource.yml"
PROMETHEUS_PATH = MONITORING / "prometheus" / "prometheus.yml"
COMPOSE_PATH = PROJECT_ROOT / "docker-compose.yml"

#: Nomes de métricas citados numa expressão PromQL (sufixos dos tipos usados).
_METRIC_RE = re.compile(r"\b([a-z_]+?(?:_total|_seconds))(?:_bucket|_sum|_count)?\b")


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dashboard() -> dict:
    return json.loads(DASHBOARD_PATH.read_text(encoding="utf-8"))


def _panels(dashboard: dict) -> list[dict]:
    """Painéis do dashboard, incluindo os aninhados em *rows*."""
    panels: list[dict] = []
    for panel in dashboard.get("panels", []):
        panels.append(panel)
        panels.extend(panel.get("panels", []))
    return [p for p in panels if p.get("type") != "row"]


def _exprs(dashboard: dict) -> list[str]:
    return [t["expr"] for p in _panels(dashboard) for t in p.get("targets", []) if "expr" in t]


def _referenced_metrics(dashboard: dict) -> set[str]:
    return {m for expr in _exprs(dashboard) for m in _METRIC_RE.findall(expr)}


# ---------------------------------------------------------------------------
# Dashboard do Grafana
# ---------------------------------------------------------------------------


def test_dashboard_has_at_least_three_panels(dashboard: dict) -> None:
    """Requisito do desafio: dashboard com pelo menos 3 painéis."""
    assert len(_panels(dashboard)) >= 3


def test_dashboard_covers_required_signals(dashboard: dict) -> None:
    """Há queries de total de requisições, latência (percentil) e taxa de erro."""
    exprs = _exprs(dashboard)

    assert any("http_requests_total" in e for e in exprs), "falta painel de requisições"
    assert any(
        "histogram_quantile" in e and "http_request_duration_seconds_bucket" in e
        for e in exprs
    ), "falta painel de latência"
    assert any('status_code=~"5..' in e for e in exprs), "falta painel de taxa de erro"


def test_every_panel_has_a_query(dashboard: dict) -> None:
    """Nenhum painel ficou sem target/expressão."""
    for panel in _panels(dashboard):
        exprs = [t.get("expr") for t in panel.get("targets", [])]
        assert any(exprs), f"painel sem query: {panel.get('title')!r}"


def test_dashboard_uses_provisioned_datasource_uid(dashboard: dict) -> None:
    """Todos os ``datasource.uid`` do dashboard batem com o datasource provisionado."""
    datasources = _load_yaml(DATASOURCE_PATH)["datasources"]
    provisioned_uids = {ds["uid"] for ds in datasources if ds["type"] == "prometheus"}

    used_uids = {
        ref["uid"]
        for panel in _panels(dashboard)
        for ref in [panel.get("datasource")] + [t.get("datasource") for t in panel["targets"]]
        if isinstance(ref, dict) and "uid" in ref
    }

    assert used_uids, "dashboard não referencia nenhum datasource"
    assert used_uids <= provisioned_uids


def test_dashboard_metrics_are_exposed_by_api(dashboard: dict) -> None:
    """Toda métrica consultada no dashboard aparece em ``GET /metrics``.

    Faz uma predição antes para que métricas com labels (``prediction_total``)
    já tenham séries exportadas.
    """
    referenced = _referenced_metrics(dashboard)
    assert referenced, "nenhuma métrica encontrada nas queries do dashboard"

    with TestClient(app) as client:
        assert client.post("/predict", json={"text": "acute chest pain"}).status_code == 200
        exposed = client.get("/metrics").text

    # Histogramas só aparecem como séries _bucket/_count/_sum.
    missing = {
        m
        for m in referenced
        if not re.search(rf"^{m}(?:_bucket|_count|_sum)?[{{ ]", exposed, re.MULTILINE)
    }
    assert not missing, f"métricas do dashboard ausentes em /metrics: {sorted(missing)}"


# ---------------------------------------------------------------------------
# Prometheus x Docker Compose
# ---------------------------------------------------------------------------


def test_prometheus_scrapes_api_service_from_compose() -> None:
    """O alvo do scrape é ``<serviço da API>:<porta interna>`` do Compose."""
    prometheus = _load_yaml(PROMETHEUS_PATH)
    compose = _load_yaml(COMPOSE_PATH)

    api_port = compose["services"]["api"]["ports"][0].split(":")[-1]
    targets = {
        target
        for job in prometheus["scrape_configs"]
        for static in job.get("static_configs", [])
        for target in static["targets"]
    }

    assert f"api:{api_port}" in targets
    for job in prometheus["scrape_configs"]:
        assert job.get("metrics_path", "/metrics") == "/metrics"


def test_compose_runs_api_prometheus_and_grafana_on_shared_network() -> None:
    """API, Prometheus e Grafana estão no Compose e na mesma rede."""
    services = _load_yaml(COMPOSE_PATH)["services"]

    assert {"api", "prometheus", "grafana"} <= set(services)
    names = ("api", "prometheus", "grafana")
    networks = [set(services[name].get("networks", [])) for name in names]
    assert set.intersection(*networks), "serviços não compartilham uma rede"
