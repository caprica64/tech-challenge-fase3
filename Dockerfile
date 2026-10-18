# syntax=docker/dockerfile:1
# ----------------------------------------------------------------------------
# Imagem do servidor local da API de triagem de laudos médicos (R3).
#
# - Base slim para tamanho razoável (R3.5).
# - Instala apenas as dependências de runtime a partir de requirements.txt.
# - Copia o código da aplicação (src/) e o artefato do modelo (models/) para
#   que a API carregue o modelo no startup (R3.2, R3.3).
# - Expõe a porta 8000 e define um CMD claro com Uvicorn (R3.4).
#
# A imagem para AWS Lambda é construída separadamente (Dockerfile.lambda),
# onde o modelo é carregado do S3 em runtime — ver tarefa 14.
# ----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# O grafo ONNX exportado (TfidfVectorizer -> StringNormalizer) constrói o
# locale en_US.UTF-8 ao inicializar a sessão do ONNX Runtime. A imagem slim
# só traz o locale C/POSIX, então sem isto o carregamento do modelo falha
# (/predict retornaria 503). Instalamos e geramos o locale UTF-8.
RUN apt-get update \
    && apt-get install -y --no-install-recommends locales \
    && sed -i '/en_US.UTF-8/s/^# //' /etc/locale.gen \
    && locale-gen \
    && rm -rf /var/lib/apt/lists/*

# Boas práticas de runtime Python em container:
#   - não gerar arquivos .pyc;
#   - stdout/stderr sem buffer (logs imediatos);
#   - pip sem cache e sem checagem de versão (imagem menor, build mais limpo).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    MODEL_DIR=/app/models \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

WORKDIR /app

# Instala as dependências primeiro para aproveitar o cache de camadas: só é
# refeito quando requirements.txt muda, não a cada alteração de código (R3.5).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Código da aplicação e artefato do modelo (R3.2, R3.3).
COPY src/ ./src/
COPY models/ ./models/

# Executa como usuário não-root (boa prática de segurança). O usuário é dono
# apenas do que precisa; os arquivos copiados são legíveis por padrão.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

# Porta exposta da API (R3.4). O Uvicorn escuta nela via $PORT.
EXPOSE 8000

# Verificação de saúde usando o endpoint /health da própria API (R3.2).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request,sys; \
url='http://127.0.0.1:%s/health' % os.environ.get('PORT','8000'); \
sys.exit(0 if urllib.request.urlopen(url, timeout=4).status == 200 else 1)"

# Entrypoint claro: sobe a API FastAPI com Uvicorn (R3.4).
# A forma shell permite expandir $PORT (configurável por ambiente).
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
