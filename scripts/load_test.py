"""Driver de inferências em sequência contra a API em execução (R6/R11).

Diferente de ``scripts/benchmark.py`` (que mede o modelo **em processo**,
carregando ``model.pkl``/``model.onnx`` diretamente), este script exercita a
**API HTTP** de ponta a ponta: dispara N requisições ``POST /predict`` em
sequência contra uma URL alvo e mede a latência do lado do cliente. Serve a
dois propósitos:

1. **Benchmark end-to-end** (inclui overhead de rede/HTTP/serialização), com
   p50/p95/média/min/máx e distribuição de status e de classes previstas.
2. **Gerar tráfego e logs**: como bate no endpoint real, popula as métricas
   Prometheus da API (``/metrics``) e os logs do servidor — útil para demonstrar
   os painéis de Grafana (local) ou os logs/latência na nuvem.

Usa apenas a biblioteca padrão (``urllib``), então roda sem dependências extras
e funciona tanto contra o Compose local quanto contra a URL publicada na AWS.

Uso:
    # local (API em http://localhost:8000)
    python -m scripts.load_test
    python scripts/load_test.py --url http://localhost:8000 --n 500

    # contra a API publicada (inclui o stage no path)
    python scripts/load_test.py \
        --url https://xxxx.execute-api.us-east-1.amazonaws.com/dev --n 200

    # textos customizados (um laudo por linha) e arquivos de saída
    python scripts/load_test.py --input laudos.txt \
        --log-file run.log --summary-json run.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

#: Número padrão de requisições cronometradas.
DEFAULT_REQUESTS = 200

#: Timeout por requisição, em segundos.
DEFAULT_TIMEOUT = 30.0

#: Laudos representativos embutidos (um por classe de urgência), usados em
#: rodízio quando nenhum arquivo de entrada é fornecido.
DEFAULT_TEXTS: tuple[str, ...] = (
    "Routine follow-up for stable chronic condition. Findings within normal limits.",
    "Persistent abdominal pain that warrants further evaluation and monitoring.",
    "Patient presents with acute crushing chest pain and signs of organ failure.",
)

#: Caminhos padrão dos artefatos gerados (gitignored).
DEFAULT_LOG_FILE = Path("load_test.log")
DEFAULT_SUMMARY_JSON = Path("load_test_results.json")


def _percentile(ordered: list[float], pct: float) -> float:
    """Percentil por interpolação linear sobre uma lista já ordenada."""
    if not ordered:
        raise ValueError("Sem valores para o percentil.")
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * frac


def _load_texts(input_path: Path | None) -> list[str]:
    """Carrega os laudos de entrada (um por linha) ou usa os embutidos."""
    if input_path is None:
        return list(DEFAULT_TEXTS)
    lines = [ln.strip() for ln in input_path.read_text().splitlines()]
    texts = [ln for ln in lines if ln]
    if not texts:
        raise ValueError(f"Arquivo de entrada vazio: {input_path}")
    return texts


def _post_predict(url: str, text: str, timeout: float) -> tuple[int, str | None, float]:
    """Envia um ``POST /predict`` e devolve ``(status, label, latency_ms)``.

    ``label`` é a classe prevista quando a resposta é 200 e bem-formada; caso
    contrário, ``None``. Erros de rede/HTTP são capturados e reportados como um
    status apropriado (o código HTTP quando disponível, senão 0).
    """
    endpoint = url.rstrip("/") + "/predict"
    payload = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - URL do operador
            body = resp.read()
            latency_ms = (time.perf_counter() - start) * 1000.0
            status = resp.status
            label = None
            try:
                label = json.loads(body).get("label")
            except (ValueError, AttributeError):
                label = None
            return status, label, latency_ms
    except urllib.error.HTTPError as exc:
        latency_ms = (time.perf_counter() - start) * 1000.0
        return exc.code, None, latency_ms
    except (urllib.error.URLError, TimeoutError, OSError):
        latency_ms = (time.perf_counter() - start) * 1000.0
        return 0, None, latency_ms


def run(
    *,
    url: str,
    requests: int,
    texts: list[str],
    timeout: float,
    log_file: Path,
    summary_json: Path,
) -> dict:
    """Executa a sequência de inferências, grava o log e devolve o resumo."""
    latencies_ok: list[float] = []
    status_counts: Counter[int] = Counter()
    label_counts: Counter[str] = Counter()

    print(f"Disparando {requests} requisições POST {url.rstrip('/')}/predict ...")
    with log_file.open("w", encoding="utf-8") as log:
        log.write("timestamp\tindex\tstatus\tlatency_ms\tlabel\n")
        for i in range(requests):
            text = texts[i % len(texts)]
            status, label, latency_ms = _post_predict(url, text, timeout)
            ts = datetime.now(UTC).isoformat()
            log.write(f"{ts}\t{i}\t{status}\t{latency_ms:.3f}\t{label or '-'}\n")

            status_counts[status] += 1
            if status == 200:
                latencies_ok.append(latency_ms)
                if label:
                    label_counts[label] += 1

            # Progresso a cada 10% (evita poluir o stdout em execuções grandes).
            if requests >= 10 and (i + 1) % max(1, requests // 10) == 0:
                done = i + 1
                ok = len(latencies_ok)
                print(f"  {done}/{requests}  (ok={ok}, erros={done - ok})")

    ordered = sorted(latencies_ok)
    summary: dict = {
        "url": url,
        "timestamp": datetime.now(UTC).isoformat(),
        "requests": requests,
        "successful": len(ordered),
        "failed": requests - len(ordered),
        "status_counts": {str(k): v for k, v in sorted(status_counts.items())},
        "label_distribution": dict(sorted(label_counts.items())),
        "latency_ms": None,
    }
    if ordered:
        summary["latency_ms"] = {
            "p50": _percentile(ordered, 50),
            "p95": _percentile(ordered, 95),
            "mean": statistics.fmean(ordered),
            "min": ordered[0],
            "max": ordered[-1],
        }

    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    _print_summary(summary, log_file, summary_json)
    return summary


def _print_summary(summary: dict, log_file: Path, summary_json: Path) -> None:
    """Imprime um resumo legível no stdout."""
    print("\n" + "=" * 60)
    print(f"Requisições : {summary['requests']}  "
          f"(ok={summary['successful']}, erros={summary['failed']})")
    print(f"Status      : {summary['status_counts']}")
    if summary["latency_ms"]:
        lat = summary["latency_ms"]
        print("Latência (ms, lado do cliente):")
        print(f"  p50={lat['p50']:.2f}  p95={lat['p95']:.2f}  "
              f"média={lat['mean']:.2f}  min={lat['min']:.2f}  max={lat['max']:.2f}")
    if summary["label_distribution"]:
        print(f"Classes previstas: {summary['label_distribution']}")
    print("=" * 60)
    print(f"Log por requisição : {log_file}")
    print(f"Resumo (JSON)      : {summary_json}")


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada de linha de comando."""
    parser = argparse.ArgumentParser(
        description=(
            "Dispara N inferências POST /predict em sequência contra a API em "
            "execução, mede latência do lado do cliente e grava log + resumo."
        )
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="URL base da API (inclua o stage se publicada, ex.: .../dev). "
             "Padrão: http://localhost:8000",
    )
    parser.add_argument(
        "--n", "--requests", dest="requests", type=int, default=DEFAULT_REQUESTS,
        help=f"Número de requisições a enviar (padrão: {DEFAULT_REQUESTS}).",
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Arquivo com laudos (um por linha). Se ausente, usa textos embutidos.",
    )
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT,
        help=f"Timeout por requisição em segundos (padrão: {DEFAULT_TIMEOUT}).",
    )
    parser.add_argument(
        "--log-file", type=Path, default=DEFAULT_LOG_FILE,
        help=f"Arquivo de log por requisição (padrão: {DEFAULT_LOG_FILE}).",
    )
    parser.add_argument(
        "--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON,
        help=f"Arquivo JSON do resumo (padrão: {DEFAULT_SUMMARY_JSON}).",
    )
    args = parser.parse_args(argv)

    if args.requests < 1:
        parser.error("--n deve ser >= 1")

    texts = _load_texts(args.input)
    summary = run(
        url=args.url,
        requests=args.requests,
        texts=texts,
        timeout=args.timeout,
        log_file=args.log_file,
        summary_json=args.summary_json,
    )
    # Código de saída não-zero se todas as requisições falharam (útil em CI/scripts).
    return 0 if summary["successful"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
