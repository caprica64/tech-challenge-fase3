# syntax=docker/dockerfile:1
# ----------------------------------------------------------------------------
# Imagem do Airflow para a demo local da DAG train_triage_model (R5).
#
# Parte da imagem oficial do Apache Airflow e instala as dependências de ML do
# projeto (scikit-learn, pandas, onnxruntime, skl2onnx, joblib) para que os
# callables da DAG — que reutilizam `src/model` e `src/data` — rodem dentro do
# worker.
#
# Por que SEM o constraints file do Airflow nestas libs:
#   O constraints oficial pina pandas/numpy (ex.: pandas==2.1.4) para proteger o
#   CORE do Airflow. Forçar os pins de ML do projeto (pandas==2.2.3) através dele
#   gera ResolutionImpossible. Como estas libs são EXTRAS de ML (não core do
#   Airflow), deixamos o pip resolver versões compatíveis — mantendo o
#   scikit-learn/onnx do projeto e uma versão de pandas que satisfaça o Airflow.
#
# NÃO instalamos requirements.txt inteiro (ele traz fastapi, uvicorn, dvc[s3],
# etc., irrelevantes ao Airflow e potencialmente conflitantes).
# ----------------------------------------------------------------------------
FROM apache/airflow:2.10.3-python3.11

# Dependências de ML da DAG. scikit-learn/onnx ficam pinados (iguais ao projeto,
# para reproduzir o mesmo modelo); pandas fica com um piso compatível tanto com
# o scikit-learn 1.6.1 quanto com o Airflow, deixando o resolvedor escolher.
# onnx fica pinado em 1.17.0 porque o skl2onnx 1.18.0 ainda importa
# `onnx.mapping`, removido no onnx >= 1.18. Sem este teto, o pip puxa um onnx
# novo demais e o import do skl2onnx quebra (ImportError: mapping).
RUN pip install --no-cache-dir \
    "scikit-learn==1.6.1" \
    "joblib==1.4.2" \
    "onnx==1.17.0" \
    "onnxruntime==1.20.1" \
    "skl2onnx==1.18.0" \
    "pandas>=2.1.4,<2.3"
