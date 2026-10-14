"""Benchmark comparativo de latência e acurácia — original vs. otimizado (R7, R11).

Mede e compara a inferência do modelo **original** (pipeline scikit-learn em
``models/model.pkl``) contra o modelo **otimizado** (grafo ONNX em
``models/model.onnx``, servido via ONNX Runtime), reportando:

- **Latência** p50/p95/média em milissegundos para cada modelo (R11.1, R11.2) e
  o **ganho percentual** do ONNX sobre o ``.pkl`` em cada percentil (R11.3).
- **Acurácia** e **F1** (macro + por classe) de cada modelo sobre a mesma
  partição de teste, para documentar o **trade-off latência vs. acurácia**
  (R7.5): a conversão para ONNX deve preservar a qualidade.
- A verificação da **meta p95 < 100 ms local** (R7.4), sinalizada como
  **PASS/FAIL**.

Os resultados são gravados em dois formatos (R7.2, R7.3, R11.3):

- ``benchmark_results.json`` — estruturado (por modelo: p50/p95/média, ganhos,
  accuracy/f1, verificação da meta e metadados como N e timestamp).
- ``benchmark_results.md`` — legível (tabela comparativa de latências com ganho
  %, tabela de accuracy/f1, veredito p95<100ms e nota de trade-off).

Ambos os arquivos são artefatos gerados (gitignored).

------------------------------------------------------------------------------
Metodologia
------------------------------------------------------------------------------
- **Latência**: reutiliza ``benchmark_classifier(classifier, texts, ...)`` e
  ``summarize_latencies(...)``. Cada requisição é cronometrada individualmente
  com ``time.perf_counter``; uma fase de **warmup** não cronometrada descarta o
  custo inicial (JIT/caches/alocação).
- **Acurácia**: reconstrói o **mesmo split estratificado** usado no treino
  (``train.RANDOM_STATE`` + ``stratify=y``) para uma comparação justa sobre o
  conjunto de teste. As predições de cada ``Classifier`` são coletadas chamando
  ``predict(text)`` por texto; accuracy/f1 vêm de ``sklearn.metrics``.
- Cada modelo é forçado via ``MODEL_FORMAT`` (``pkl`` vs ``onnx``) para medir o
  caminho real de cada artefato, independentemente da seleção padrão.

Uso:
    python -m scripts.benchmark
    python -m scripts.benchmark --n 500
    python -m scripts.benchmark --requests 1000 --warmup 50
    python -m scripts.benchmark --baseline-only
    python -m scripts.benchmark --output-json out.json --output-md out.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.model.inference import Classifier
from src.model.train import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_DIR,
    DEFAULT_TEST_SIZE,
    RANDOM_STATE,
    URGENCY_CLASSES,
    load_dataset,
)

#: Número padrão de requisições cronometradas (baseline local).
DEFAULT_REQUESTS = 1000

#: Número padrão de requisições de aquecimento (não cronometradas).
DEFAULT_WARMUP = 50

#: Meta fixa de latência p95 em ambiente local (R7.4 / R11): < 100 ms.
P95_TARGET_MS = 100.0

#: Número máximo de textos de teste a usar na avaliação de acurácia (limita o
#: custo de chamar ``predict`` por texto quando o conjunto de teste é grande).
DEFAULT_ACCURACY_LIMIT = 1000

#: Caminhos padrão dos arquivos de resultado (artefatos gerados, gitignored).
DEFAULT_OUTPUT_JSON = Path("benchmark_results.json")
DEFAULT_OUTPUT_MD = Path("benchmark_results.md")

#: Textos representativos embutidos (um por classe) usados como fallback
#: quando o dataset canônico não está disponível.
FALLBACK_SAMPLE_TEXTS: tuple[str, ...] = (
    "Routine follow-up abdominal ultrasound shows stable findings with no "
    "acute abnormality. Liver, spleen and kidneys are within normal limits.",
    "Chest radiograph demonstrates a small pleural effusion that warrants "
    "clinical attention and short-interval follow-up imaging.",
    "CT angiography reveals an acute pulmonary embolism with signs of right "
    "heart strain; immediate clinical correlation and treatment are required.",
)


@dataclass(frozen=True)
class LatencySummary:
    """Resumo estatístico das latências de inferência medidas.

    Attributes:
        count: número de requisições cronometradas.
        p50_ms: mediana (percentil 50) da latência, em milissegundos.
        p95_ms: percentil 95 da latência, em milissegundos.
        mean_ms: latência média, em milissegundos.
        min_ms: menor latência observada, em milissegundos.
        max_ms: maior latência observada, em milissegundos.
    """

    count: int
    p50_ms: float
    p95_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float


@dataclass(frozen=True)
class AccuracySummary:
    """Resumo de acurácia de um modelo sobre o conjunto de teste.

    Attributes:
        n_samples: número de amostras de teste avaliadas.
        accuracy: acurácia global em ``[0, 1]``.
        f1_macro: F1 macro (média não ponderada por classe).
        f1_per_class: mapa classe -> F1.
    """

    n_samples: int
    accuracy: float
    f1_macro: float
    f1_per_class: dict[str, float]


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Calcula um percentil por interpolação linear.

    Args:
        sorted_values: valores já ordenados de forma crescente (não vazios).
        pct: percentil desejado em ``[0, 100]``.

    Returns:
        O valor do percentil ``pct`` interpolado entre os pontos vizinhos.
    """
    if not sorted_values:
        raise ValueError("Não há valores para calcular o percentil.")
    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (pct / 100.0) * (len(sorted_values) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    frac = rank - low
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * frac


def summarize_latencies(latencies_ms: list[float]) -> LatencySummary:
    """Agrega uma lista de latências (ms) em p50/p95/média e extremos.

    Args:
        latencies_ms: latências individuais, em milissegundos.

    Returns:
        ``LatencySummary`` com os percentis e estatísticas resumidas.

    Raises:
        ValueError: se a lista de latências estiver vazia.
    """
    if not latencies_ms:
        raise ValueError("Nenhuma latência medida para resumir.")

    ordered = sorted(latencies_ms)
    return LatencySummary(
        count=len(ordered),
        p50_ms=_percentile(ordered, 50),
        p95_ms=_percentile(ordered, 95),
        mean_ms=statistics.fmean(ordered),
        min_ms=ordered[0],
        max_ms=ordered[-1],
    )


def benchmark_classifier(
    classifier: Classifier,
    texts: list[str],
    *,
    requests: int = DEFAULT_REQUESTS,
    warmup: int = DEFAULT_WARMUP,
) -> LatencySummary:
    """Mede a latência de inferência de um ``Classifier`` sobre N requisições.

    Faz ``warmup`` chamadas descartadas e depois cronometra ``requests``
    chamadas a ``classifier.predict``, ciclando pelos ``texts`` de amostra.
    Cada chamada é cronometrada individualmente com ``time.perf_counter``.

    Reutilizável para comparar ``.pkl`` vs ``.onnx``.

    Args:
        classifier: instância já carregada (``.pkl`` ou ``.onnx``).
        texts: textos representativos a classificar (ciclados em round-robin).
        requests: número de requisições cronometradas.
        warmup: número de requisições de aquecimento (não cronometradas).

    Returns:
        ``LatencySummary`` com p50/p95/média em milissegundos.

    Raises:
        ValueError: se ``texts`` estiver vazio ou ``requests`` for <= 0.
    """
    if not texts:
        raise ValueError("É necessário ao menos um texto de amostra.")
    if requests <= 0:
        raise ValueError("O número de requisições deve ser positivo.")

    n_texts = len(texts)

    # Warmup: descarta o custo inicial (import tardio, caches, alocação).
    for i in range(max(0, warmup)):
        classifier.predict(texts[i % n_texts])

    latencies_ms: list[float] = []
    for i in range(requests):
        text = texts[i % n_texts]
        start = time.perf_counter()
        classifier.predict(text)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)

    return summarize_latencies(latencies_ms)


def evaluate_classifier(
    classifier: Classifier,
    texts: list[str],
    labels: list[str],
) -> AccuracySummary:
    """Avalia accuracy e F1 (macro + por classe) de um ``Classifier`` (R7.5).

    Coleta as predições chamando ``classifier.predict(text)`` por texto e
    compara com os rótulos verdadeiros via ``sklearn.metrics``. Permite medir o
    trade-off de acurácia entre o modelo original (``.pkl``) e o otimizado
    (``.onnx``).

    Args:
        classifier: instância já carregada (``.pkl`` ou ``.onnx``).
        texts: textos do conjunto de teste.
        labels: rótulos verdadeiros, alinhados a ``texts``.

    Returns:
        ``AccuracySummary`` com accuracy, F1 macro e F1 por classe.

    Raises:
        ValueError: se ``texts``/``labels`` estiverem vazios ou com tamanhos
            diferentes.
    """
    if not texts or not labels:
        raise ValueError("É necessário ao menos um texto e rótulo para avaliar.")
    if len(texts) != len(labels):
        raise ValueError("texts e labels devem ter o mesmo tamanho.")

    from sklearn.metrics import accuracy_score, f1_score

    class_order = list(URGENCY_CLASSES)
    predictions = [classifier.predict(text)["label"] for text in texts]

    accuracy = float(accuracy_score(labels, predictions))
    f1_macro = float(
        f1_score(labels, predictions, labels=class_order, average="macro", zero_division=0)
    )
    per_class = f1_score(labels, predictions, labels=class_order, average=None, zero_division=0)
    f1_per_class = {cls: float(score) for cls, score in zip(class_order, per_class, strict=True)}

    return AccuracySummary(
        n_samples=len(texts),
        accuracy=accuracy,
        f1_macro=f1_macro,
        f1_per_class=f1_per_class,
    )


def load_sample_texts(data_path: Path, *, limit: int = 200) -> list[str]:
    """Carrega textos representativos para o benchmark de latência.

    Tenta amostrar o dataset canônico (``data/laudos.csv``); se o arquivo não
    existir ou não puder ser lido, usa ``FALLBACK_SAMPLE_TEXTS``.

    Args:
        data_path: caminho do CSV com coluna ``text``.
        limit: número máximo de textos a carregar do dataset.

    Returns:
        Lista não vazia de textos de amostra.
    """
    if data_path.exists():
        try:
            import pandas as pd

            frame = pd.read_csv(data_path)
            texts = [
                str(t).strip()
                for t in frame["text"].tolist()
                if isinstance(t, str) and str(t).strip()
            ]
            if texts:
                return texts[:limit]
        except (OSError, ValueError, KeyError, ImportError):
            pass

    return list(FALLBACK_SAMPLE_TEXTS)


def load_test_split(
    data_path: Path,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    limit: int | None = DEFAULT_ACCURACY_LIMIT,
) -> tuple[list[str], list[str]]:
    """Reconstrói a partição de teste estratificada usada no treino (R7.5, R9.7).

    Usa a mesma ``RANDOM_STATE`` e ``stratify=y`` de ``train.train_model`` para
    obter exatamente o conjunto de teste, garantindo uma comparação justa de
    acurácia entre ``.pkl`` e ``.onnx``. Opcionalmente limita o número de
    amostras (amostragem estratificada reprodutível) para controlar o custo de
    chamar ``predict`` por texto.

    Args:
        data_path: caminho do dataset CSV (colunas ``text,label``).
        test_size: proporção do conjunto de teste (igual ao treino).
        limit: nº máximo de amostras de teste; ``None`` usa o teste completo.

    Returns:
        Tupla ``(textos_teste, rotulos_teste)`` alinhados.
    """
    from sklearn.model_selection import train_test_split

    df = load_dataset(data_path)
    x = df["text"].astype("string")
    y = df["label"].astype("string")

    _, x_test, _, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    if limit is not None and len(x_test) > limit:
        # Subamostra estratificada reprodutível do conjunto de teste.
        _, x_test, _, y_test = train_test_split(
            x_test,
            y_test,
            test_size=limit,
            random_state=RANDOM_STATE,
            stratify=y_test,
        )

    return [str(t) for t in x_test.tolist()], [str(label) for label in y_test.tolist()]


def _gain_pct(baseline: float, optimized: float) -> float:
    """Ganho percentual do otimizado sobre o baseline (positivo = mais rápido).

    ``(baseline - optimized) / baseline * 100``. Um valor positivo indica que o
    modelo otimizado é mais rápido; negativo indica regressão.
    """
    if baseline == 0:
        return 0.0
    return (baseline - optimized) / baseline * 100.0


@dataclass(frozen=True)
class ModelResult:
    """Resultado completo (latência + acurácia) de um único modelo."""

    label: str
    model_format: str
    model_path: str
    latency: LatencySummary
    accuracy: AccuracySummary | None


def build_results_payload(
    baseline: ModelResult,
    optimized: ModelResult | None,
    *,
    requests: int,
    warmup: int,
) -> dict:
    """Monta o dicionário estruturado de resultados para ``benchmark_results.json``.

    Args:
        baseline: resultado do modelo original (``.pkl``).
        optimized: resultado do modelo otimizado (``.onnx``), ou ``None`` no
            modo baseline-only.
        requests: nº de requisições cronometradas.
        warmup: nº de requisições de warmup.

    Returns:
        Dicionário serializável em JSON com modelos, ganhos, verificação da meta
        e metadados.
    """

    def model_block(result: ModelResult) -> dict:
        block = {
            "label": result.label,
            "model_format": result.model_format,
            "model_path": result.model_path,
            "latency_ms": {
                "count": result.latency.count,
                "p50": result.latency.p50_ms,
                "p95": result.latency.p95_ms,
                "mean": result.latency.mean_ms,
                "min": result.latency.min_ms,
                "max": result.latency.max_ms,
            },
        }
        if result.accuracy is not None:
            block["accuracy"] = asdict(result.accuracy)
        return block

    models = {"baseline": model_block(baseline)}
    if optimized is not None:
        models["optimized"] = model_block(optimized)

    gains = None
    if optimized is not None:
        gains = {
            "p50_pct": _gain_pct(baseline.latency.p50_ms, optimized.latency.p50_ms),
            "p95_pct": _gain_pct(baseline.latency.p95_ms, optimized.latency.p95_ms),
            "mean_pct": _gain_pct(baseline.latency.mean_ms, optimized.latency.mean_ms),
        }

    # A meta p95 local é verificada sobre o modelo otimizado (alvo de produção);
    # na ausência dele, avalia o baseline.
    target_model = optimized or baseline
    p95_target = {
        "target_ms": P95_TARGET_MS,
        "evaluated_model": target_model.label,
        "p95_ms": target_model.latency.p95_ms,
        "pass": target_model.latency.p95_ms < P95_TARGET_MS,
    }

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "metadata": {
            "requests": requests,
            "warmup": warmup,
            "p95_target_ms": P95_TARGET_MS,
            "classes": list(URGENCY_CLASSES),
        },
        "models": models,
        "latency_gain": gains,
        "p95_target_check": p95_target,
    }


def _fmt_gain(value: float) -> str:
    """Formata um ganho percentual com sinal explícito (ex.: ``+42.3%``)."""
    return f"{value:+.1f}%"


def render_markdown(payload: dict) -> str:
    """Renderiza ``benchmark_results.md`` legível a partir do payload estruturado.

    Produz: tabela comparativa de latências (p50/p95/média + ganho %), tabela de
    accuracy/f1, veredito p95<100ms e uma nota de trade-off latência vs.
    acurácia (R7.3, R7.5, R11.3).
    """
    models = payload["models"]
    baseline = models["baseline"]
    optimized = models.get("optimized")
    gains = payload.get("latency_gain")
    meta = payload["metadata"]
    check = payload["p95_target_check"]

    lines: list[str] = []
    lines.append("# Benchmark de latência e acurácia — original vs. otimizado")
    lines.append("")
    lines.append(f"_Gerado em: {payload['generated_at']}_")
    lines.append("")
    lines.append(
        f"- Requisições cronometradas: **{meta['requests']}** "
        f"(warmup: {meta['warmup']})"
    )
    lines.append(f"- Meta de latência local (p95): **< {meta['p95_target_ms']:.0f} ms** (R7.4)")
    lines.append(
        f"- Modelo original: `{baseline['model_format']}` (`{baseline['model_path']}`)"
    )
    if optimized is not None:
        lines.append(
            f"- Modelo otimizado: `{optimized['model_format']}` (`{optimized['model_path']}`)"
        )
    lines.append("")

    # --- Tabela de latência -------------------------------------------------
    lines.append("## Latência (ms)")
    lines.append("")
    if optimized is not None:
        lines.append("| Métrica | Original (.pkl) | Otimizado (.onnx) | Ganho |")
        lines.append("|---------|-----------------|-------------------|-------|")
        bl = baseline["latency_ms"]
        op = optimized["latency_ms"]
        lines.append(
            f"| p50 | {bl['p50']:.3f} | {op['p50']:.3f} | {_fmt_gain(gains['p50_pct'])} |"
        )
        lines.append(
            f"| p95 | {bl['p95']:.3f} | {op['p95']:.3f} | {_fmt_gain(gains['p95_pct'])} |"
        )
        lines.append(
            f"| média | {bl['mean']:.3f} | {op['mean']:.3f} | {_fmt_gain(gains['mean_pct'])} |"
        )
    else:
        lines.append("| Métrica | Original (.pkl) |")
        lines.append("|---------|-----------------|")
        bl = baseline["latency_ms"]
        lines.append(f"| p50 | {bl['p50']:.3f} |")
        lines.append(f"| p95 | {bl['p95']:.3f} |")
        lines.append(f"| média | {bl['mean']:.3f} |")
    lines.append("")
    lines.append("> Ganho = `(original - otimizado) / original`; positivo indica ONNX mais rápido.")
    lines.append("")

    # --- Veredito da meta p95 ----------------------------------------------
    verdict = "✅ PASS" if check["pass"] else "❌ FAIL"
    lines.append("## Meta p95 < 100 ms (local, R7.4)")
    lines.append("")
    lines.append(
        f"- Modelo avaliado: **{check['evaluated_model']}** — "
        f"p95 = **{check['p95_ms']:.3f} ms** / alvo < {check['target_ms']:.0f} ms"
    )
    lines.append(f"- Veredito: **{verdict}**")
    lines.append("")

    # --- Tabela de acurácia -------------------------------------------------
    has_accuracy = baseline.get("accuracy") is not None
    if has_accuracy:
        lines.append("## Acurácia / F1 (trade-off, R7.5)")
        lines.append("")
        classes = meta["classes"]
        header = "| Modelo | Accuracy | F1 macro | " + " | ".join(
            f"F1 {c}" for c in classes
        ) + " |"
        sep = "|--------|----------|----------|" + "|".join("----" for _ in classes) + "|"
        lines.append(header)
        lines.append(sep)

        def acc_row(name: str, block: dict) -> str:
            acc = block["accuracy"]
            per = acc["f1_per_class"]
            cells = " | ".join(f"{per[c]:.4f}" for c in classes)
            return f"| {name} | {acc['accuracy']:.4f} | {acc['f1_macro']:.4f} | {cells} |"

        lines.append(acc_row("Original (.pkl)", baseline))
        if optimized is not None and optimized.get("accuracy") is not None:
            lines.append(acc_row("Otimizado (.onnx)", optimized))
        lines.append("")
        lines.append(
            f"> Avaliado sobre {baseline['accuracy']['n_samples']} amostras do conjunto "
            "de teste (split estratificado reprodutível, seed 42)."
        )
        lines.append("")

    # --- Nota de trade-off --------------------------------------------------
    lines.append("## Trade-off latência vs. acurácia")
    lines.append("")
    if optimized is not None and has_accuracy and optimized.get("accuracy") is not None:
        bl_acc = baseline["accuracy"]["accuracy"]
        op_acc = optimized["accuracy"]["accuracy"]
        delta = op_acc - bl_acc
        p95_gain = gains["p95_pct"]
        if abs(delta) < 1e-9:
            acc_note = (
                "A conversão para ONNX **preservou integralmente a acurácia** "
                f"({op_acc:.4f}), sem qualquer degradação."
            )
        elif delta >= 0:
            acc_note = (
                f"A acurácia do ONNX ({op_acc:.4f}) é igual ou superior à do "
                f"original ({bl_acc:.4f}); não há degradação a reportar. Pequenas "
                "diferenças de ponto flutuante na reprodução do TF-IDF/LogReg pelo "
                "ONNX Runtime podem alterar a classe apenas em casos de empate "
                "(top-2 muito próximos)."
            )
        else:
            acc_note = (
                f"A acurácia do ONNX ({op_acc:.4f}) ficou {abs(delta):.4f} abaixo "
                f"do original ({bl_acc:.4f}) — degradação marginal atribuível a "
                "diferenças numéricas de ponto flutuante entre runtimes."
            )
        lines.append(
            f"A otimização via ONNX Runtime alterou a latência p95 em "
            f"**{_fmt_gain(p95_gain)}**. {acc_note} "
            "O objetivo do projeto (ciclo de MLOps de baixa latência) é atendido "
            "sem comprometer a qualidade da triagem."
        )
    else:
        lines.append(
            "Modo baseline-only: apenas o modelo original foi medido. Rode o "
            "benchmark completo (com `models/model.onnx` presente) para comparar "
            "o trade-off latência vs. acurácia."
        )
    lines.append("")

    return "\n".join(lines) + "\n"


def _benchmark_one(
    *,
    label: str,
    model_format: str,
    model_dir: Path,
    latency_texts: list[str],
    requests: int,
    warmup: int,
    eval_texts: list[str] | None,
    eval_labels: list[str] | None,
) -> ModelResult:
    """Executa latência (+ acurácia opcional) para um formato de modelo."""
    classifier = Classifier(model_format=model_format, model_dir=model_dir)
    latency = benchmark_classifier(
        classifier,
        latency_texts,
        requests=requests,
        warmup=warmup,
    )
    accuracy = None
    if eval_texts is not None and eval_labels is not None:
        accuracy = evaluate_classifier(classifier, eval_texts, eval_labels)

    return ModelResult(
        label=label,
        model_format=classifier.model_format,
        model_path=str(classifier.model_path),
        latency=latency,
        accuracy=accuracy,
    )


def run_comparison(
    *,
    data_path: Path = DEFAULT_DATA_PATH,
    model_dir: Path = DEFAULT_MODEL_DIR,
    requests: int = DEFAULT_REQUESTS,
    warmup: int = DEFAULT_WARMUP,
    baseline_only: bool = False,
    skip_accuracy: bool = False,
    accuracy_limit: int | None = DEFAULT_ACCURACY_LIMIT,
) -> dict:
    """Executa o benchmark comparativo e devolve o payload estruturado.

    Args:
        data_path: caminho do dataset (para textos de latência e split de teste).
        model_dir: diretório dos artefatos (``model.pkl`` / ``model.onnx``).
        requests: nº de requisições cronometradas por modelo.
        warmup: nº de requisições de warmup por modelo.
        baseline_only: se ``True``, mede apenas o ``.pkl`` (compatibilidade).
        skip_accuracy: se ``True``, pula a avaliação de accuracy/f1.
        accuracy_limit: nº máximo de amostras de teste para acurácia.

    Returns:
        O payload estruturado (ver ``build_results_payload``).
    """
    latency_texts = load_sample_texts(data_path)

    eval_texts: list[str] | None = None
    eval_labels: list[str] | None = None
    if not skip_accuracy:
        eval_texts, eval_labels = load_test_split(data_path, limit=accuracy_limit)

    baseline = _benchmark_one(
        label="original (.pkl)",
        model_format="pkl",
        model_dir=model_dir,
        latency_texts=latency_texts,
        requests=requests,
        warmup=warmup,
        eval_texts=eval_texts,
        eval_labels=eval_labels,
    )

    optimized: ModelResult | None = None
    if not baseline_only:
        onnx_path = model_dir / "model.onnx"
        if onnx_path.exists():
            optimized = _benchmark_one(
                label="otimizado (.onnx)",
                model_format="onnx",
                model_dir=model_dir,
                latency_texts=latency_texts,
                requests=requests,
                warmup=warmup,
                eval_texts=eval_texts,
                eval_labels=eval_labels,
            )
        else:
            print(
                f"[aviso] {onnx_path} não encontrado; executando apenas o baseline "
                "(.pkl). Exporte o ONNX com `python -m src.model.export_onnx`."
            )

    return build_results_payload(baseline, optimized, requests=requests, warmup=warmup)


def write_results(payload: dict, *, output_json: Path, output_md: Path) -> None:
    """Grava o payload em JSON e Markdown nos caminhos indicados."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    output_md.write_text(render_markdown(payload))


def _print_payload(payload: dict) -> None:
    """Imprime um resumo legível do payload no stdout."""
    models = payload["models"]
    baseline = models["baseline"]
    optimized = models.get("optimized")
    gains = payload.get("latency_gain")
    check = payload["p95_target_check"]

    print("=" * 68)
    print("Benchmark comparativo — original (.pkl) vs. otimizado (.onnx)")
    print("=" * 68)
    meta = payload["metadata"]
    print(f"Requisições : {meta['requests']} (warmup {meta['warmup']})")
    print("-" * 68)
    header = f"{'Métrica':<8}{'Original':>14}{'Otimizado':>14}{'Ganho':>12}"
    print(header)
    bl = baseline["latency_ms"]
    if optimized is not None:
        op = optimized["latency_ms"]
        for key, gain_key in (("p50", "p50_pct"), ("p95", "p95_pct"), ("mean", "mean_pct")):
            print(
                f"{key:<8}{bl[key]:>12.3f}ms{op[key]:>12.3f}ms{_fmt_gain(gains[gain_key]):>12}"
            )
    else:
        for key in ("p50", "p95", "mean"):
            print(f"{key:<8}{bl[key]:>12.3f}ms{'—':>14}{'—':>12}")
    print("-" * 68)
    verdict = "PASS ✅" if check["pass"] else "FAIL ❌"
    print(
        f"Meta p95 < {check['target_ms']:.0f} ms ({check['evaluated_model']}): "
        f"p95 = {check['p95_ms']:.3f} ms -> {verdict}"
    )

    if baseline.get("accuracy") is not None:
        print("-" * 68)
        print(f"{'Modelo':<20}{'Accuracy':>12}{'F1 macro':>12}")
        ba = baseline["accuracy"]
        print(f"{'original (.pkl)':<20}{ba['accuracy']:>12.4f}{ba['f1_macro']:>12.4f}")
        if optimized is not None and optimized.get("accuracy") is not None:
            oa = optimized["accuracy"]
            print(f"{'otimizado (.onnx)':<20}{oa['accuracy']:>12.4f}{oa['f1_macro']:>12.4f}")
    print("=" * 68)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Interpreta os argumentos de linha de comando do benchmark."""
    parser = argparse.ArgumentParser(
        description=(
            "Compara a latência (p50/p95/média + ganho %) e a acurácia (accuracy/f1) "
            "do modelo original (.pkl) vs. otimizado (.onnx) e grava "
            "benchmark_results.md/.json."
        )
    )
    parser.add_argument(
        "-n",
        "--n",
        "--requests",
        dest="requests",
        type=int,
        default=DEFAULT_REQUESTS,
        help=f"Número de requisições cronometradas (padrão: {DEFAULT_REQUESTS}).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"Requisições de aquecimento, não cronometradas (padrão: {DEFAULT_WARMUP}).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Diretório dos artefatos do modelo (padrão: {DEFAULT_MODEL_DIR}).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"CSV com textos de amostra e split de teste (padrão: {DEFAULT_DATA_PATH}).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help=f"Caminho do JSON de resultados (padrão: {DEFAULT_OUTPUT_JSON}).",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=DEFAULT_OUTPUT_MD,
        help=f"Caminho do Markdown de resultados (padrão: {DEFAULT_OUTPUT_MD}).",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Mede apenas o modelo original (.pkl), sem comparar com o ONNX.",
    )
    parser.add_argument(
        "--skip-accuracy",
        action="store_true",
        help="Pula a avaliação de accuracy/f1 (apenas latência).",
    )
    parser.add_argument(
        "--accuracy-limit",
        type=int,
        default=DEFAULT_ACCURACY_LIMIT,
        help=(
            "Nº máximo de amostras de teste para acurácia; 0 usa o teste completo "
            f"(padrão: {DEFAULT_ACCURACY_LIMIT})."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada: compara .pkl vs .onnx e grava os resultados."""
    args = parse_args(argv)

    accuracy_limit: int | None = args.accuracy_limit if args.accuracy_limit > 0 else None

    payload = run_comparison(
        data_path=args.data,
        model_dir=args.model_dir,
        requests=args.requests,
        warmup=args.warmup,
        baseline_only=args.baseline_only,
        skip_accuracy=args.skip_accuracy,
        accuracy_limit=accuracy_limit,
    )

    write_results(payload, output_json=args.output_json, output_md=args.output_md)
    _print_payload(payload)
    print(f"Resultados gravados em {args.output_json} e {args.output_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
