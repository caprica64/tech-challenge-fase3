# Requirements Document

## Introdução

Este projeto implementa um sistema de **triagem automática de laudos médicos** para um hospital de referência. O sistema classifica a urgência de exames em formato de texto (laudos) em três níveis: **normal**, **atenção** e **urgente**. O coração do sistema é um classificador de texto (NLP) leve, servido via API REST em container Docker.

O foco do tech challenge (Fase 3) é o **ciclo de vida do modelo em produção**: um pipeline CI/CD (GitHub Actions), orquestração de treino/retreino (Airflow), monitoramento essencial (Prometheus + Grafana) e otimização de latência de inferência.

O objetivo primário não é maximizar a acurácia do modelo, mas sim demonstrar um ciclo de MLOps funcional, observável e de baixa latência.

### Objetivos Mensuráveis
- Latência de inferência (p95) **< 100 ms por requisição** em ambiente local (meta fixa do projeto).
- Pipeline CI/CD com no mínimo 2 automações (lint/verificação de código + testes), executando em cada push/PR.
- DAG Airflow funcional cobrindo carregamento de dados → treino → salvamento do modelo.
- Dashboard Grafana com no mínimo 3 painéis (total de requisições, latência, taxa de erro).
- Aplicação de ao menos uma técnica de otimização de performance (ex.: conversão para ONNX, quantização ou pruning).
- Comparação quantitativa da latência: modelo original vs. modelo otimizado.

### Etapas de Desenvolvimento (visão geral)
O projeto é organizado em 4 etapas, cada uma vinculada a uma disciplina e com entregáveis próprios. Os requisitos detalhados abaixo são rastreados a estas etapas.

| Etapa | Foco / Disciplina | Entregável principal |
|-------|-------------------|----------------------|
| **1** | Decisão arquitetural de deploy em nuvem + API inicial (Deploy em Nuvem) | API FastAPI em Docker + texto de decisão arquitetural (batch vs. real-time) no README + baseline de latência |
| **2** | CI/CD + pipeline de treino (CI/CD e Pipeline de Treino) | Workflow GitHub Actions (lint/test) + DAG Airflow de treino |
| **3** | Monitoramento e observabilidade (Monitoração de Performance) | Docker Compose com API + Prometheus + Grafana + dashboard |
| **4** | Otimização de latência e entrega (Latência em Modelos Não Estruturados) | Modelo otimizado (ONNX/quantização) + comparativo de latência + vídeo STAR |

### Fora de Escopo
- Diagnóstico clínico real ou uso em produção hospitalar real.
- Autenticação/autorização avançada e conformidade regulatória (LGPD/HIPAA) completas.
- Treino de modelos de linguagem de larga escala (LLMs) — o requisito é um modelo leve.

---

## Glossário
- **Laudo**: texto livre de um exame médico.
- **Classe de urgência**: rótulo em {normal, atenção, urgente}.
- **Inferência**: ato de o modelo classificar um laudo recebido pela API.
- **Latência p95**: tempo de resposta no percentil 95 das requisições.
- **Retreino**: novo ciclo de treino do modelo com dados atualizados, orquestrado pelo Airflow.
- **Artefato do modelo**: arquivo serializado do modelo treinado (ex.: `.pkl` ou `.onnx`).

---

## Requisitos

### Requisito 1 — Classificação de urgência de laudos (modelo de ML)

**User Story:** Como equipe de triagem do hospital, quero que o sistema classifique automaticamente a urgência de um laudo de texto, para que casos urgentes sejam priorizados rapidamente.

#### Critérios de Aceitação (EARS)
1. QUANDO um laudo em texto é submetido ao modelo, O SISTEMA DEVE retornar uma das classes {normal, atenção, urgente}.
2. QUANDO o modelo retorna uma classificação, O SISTEMA DEVE incluir um score/probabilidade de confiança associado à predição.
3. O SISTEMA DEVE utilizar um pipeline leve de **TF-IDF + Logistic Regression** (Scikit-Learn) como modelo base de classificação de texto.
4. QUANDO o modelo é treinado, O SISTEMA DEVE persistir o artefato do modelo em um caminho versionável/reutilizável pela API.
5. SE o texto de entrada estiver vazio ou inválido, ENTÃO O SISTEMA DEVE rejeitar a entrada e sinalizar erro de validação.
6. O SISTEMA DEVE registrar as métricas de avaliação do modelo (ex.: accuracy, f1 por classe) ao final do treino.

#### Justificativa do modelo (TF-IDF + Logistic Regression)
- **Latência**: inferência linear (produto vetor-matriz esparsa) é extremamente rápida, favorecendo a meta de p95 < 100 ms.
- **Exportação para ONNX**: Logistic Regression é totalmente suportado por `sklearn-onnx`/ONNX Runtime, inclusive com saída de probabilidades, facilitando a otimização da Etapa 4 sem perda de funcionalidade.
- **Tamanho do artefato**: modelo compacto, reduzindo o tamanho da imagem Docker e o tempo de carregamento (relevante para cold start em Lambda).
- **Simplicidade e reprodutibilidade**: menos hiperparâmetros que ensembles (ex.: Random Forest), treino rápido e determinístico com seed fixa — adequado ao foco de MLOps do projeto.
- **Probabilidades calibráveis**: fornece `predict_proba` nativamente, permitindo expor o score de confiança exigido no critério 2.

---

### Requisito 2 — API REST de inferência

**User Story:** Como sistema cliente (frontend de triagem ou integração hospitalar), quero consumir uma API REST para classificar laudos, para que a triagem seja feita de forma programática e integrável.

#### Critérios de Aceitação (EARS)
1. O SISTEMA DEVE expor um endpoint REST de predição (ex.: `POST /predict`) construído com FastAPI.
2. QUANDO uma requisição válida é recebida em `/predict`, O SISTEMA DEVE retornar a classe de urgência e o score em formato JSON.
3. O SISTEMA DEVE expor um endpoint de health check (ex.: `GET /health`) que retorna o status do serviço.
4. SE o corpo da requisição for inválido (schema incorreto), ENTÃO O SISTEMA DEVE retornar HTTP 422 com mensagem de erro.
5. SE o modelo não estiver carregado ou falhar na inferência, ENTÃO O SISTEMA DEVE retornar HTTP 503/500 com mensagem de erro apropriada.
6. O SISTEMA DEVE carregar o artefato do modelo na inicialização da API (não a cada requisição).

---

### Requisito 3 — Containerização (Docker)

**User Story:** Como engenheiro de MLOps, quero empacotar o serviço de inferência em um container Docker, para que o deploy seja reprodutível e portável.

#### Critérios de Aceitação (EARS)
1. O SISTEMA DEVE prover um Dockerfile funcional que builda a imagem do serviço de inferência.
2. QUANDO a imagem é executada, O SISTEMA DEVE iniciar a API e disponibilizar os endpoints de predição e health.
3. O SISTEMA DEVE incluir no container o artefato do modelo ou um mecanismo para carregá-lo na inicialização.
4. O SISTEMA DEVE definir uma porta exposta e um comando de entrada (entrypoint/CMD) claros.
5. A imagem DEVE ser otimizada para tamanho razoável (ex.: imagem base slim, instalação apenas de dependências necessárias).

---

### Requisito 4 — Pipeline CI/CD (GitHub Actions)

**User Story:** Como time de desenvolvimento, quero um pipeline CI/CD automatizado, para que a qualidade do código e a construção da imagem sejam validadas a cada mudança.

#### Critérios de Aceitação (EARS)
1. O SISTEMA DEVE prover um workflow do GitHub Actions com no mínimo 2 automações.
2. QUANDO há um push ou pull request, O SISTEMA DEVE executar verificação de código (lint).
3. QUANDO há um push ou pull request, O SISTEMA DEVE executar os testes automatizados.
4. SE lint ou testes falharem, ENTÃO O SISTEMA DEVE marcar o pipeline como falho e bloquear o merge.
5. O SISTEMA DEVE executar uma etapa de build da imagem Docker após lint e testes passarem.
6. O SISTEMA DEVE organizar o pipeline em etapas sequenciais/encadeadas (ex.: lint → test → build).

---

### Requisito 5 — Orquestração de treino/retreino (Airflow)

**User Story:** Como engenheiro de MLOps, quero uma DAG Airflow para orquestrar o retreino do modelo, para que o modelo possa ser atualizado de forma automatizada e reproduzível.

#### Critérios de Aceitação (EARS)
1. O SISTEMA DEVE prover uma DAG Airflow funcional para o pipeline de treino/retreino.
2. A DAG DEVE conter, no mínimo, as etapas: carregamento de dados → treino do modelo → salvamento do artefato.
3. QUANDO a DAG é executada com sucesso, O SISTEMA DEVE persistir um novo artefato de modelo.
4. SE uma etapa da DAG falhar, ENTÃO O SISTEMA DEVE sinalizar a falha e interromper as etapas dependentes.
5. O SISTEMA DEVE permitir a execução da DAG de forma manual (trigger) e/ou agendada.

---

### Requisito 6 — Monitoramento (Prometheus + Grafana)

**User Story:** Como time de operações, quero monitorar a saúde e a performance do serviço de inferência, para que problemas de latência, erros e carga sejam detectados rapidamente.

#### Critérios de Aceitação (EARS)
1. O SISTEMA DEVE instrumentar a API com `prometheus-client`, expondo métricas em um endpoint (ex.: `GET /metrics`).
2. O SISTEMA DEVE coletar, no mínimo, métricas de: total de requisições, latência/tempo de resposta e contagem/taxa de erros.
3. O SISTEMA DEVE prover uma stack de monitoramento local via Docker Compose contendo: API + Prometheus + Grafana.
4. QUANDO a stack é iniciada, O SISTEMA DEVE permitir que o Prometheus faça scraping das métricas da API.
5. O SISTEMA DEVE prover um dashboard Grafana com no mínimo 3 painéis (total de requisições, latência/tempo de resposta, taxa de erro).
6. O SISTEMA DEVE provisionar/documentar o dashboard de forma reproduzível (ex.: provisioning via arquivo ou import documentado).

---

### Requisito 7 — Otimização de latência/performance

**User Story:** Como engenheiro de MLOps, quero otimizar a inferência do modelo, para que a latência atenda ao limiar-alvo e o custo computacional seja reduzido.

#### Critérios de Aceitação (EARS)
1. O SISTEMA DEVE aplicar ao menos uma técnica de otimização de performance (ex.: conversão para ONNX, quantização básica ou pruning).
2. QUANDO a técnica de otimização é aplicada, O SISTEMA DEVE registrar a latência antes e depois para comparação (benchmark).
3. O SISTEMA DEVE documentar a técnica escolhida e o ganho de latência obtido.
4. A latência de inferência (p95) DEVE ser **< 100 ms por requisição** em ambiente local.
5. SE a otimização degradar a acurácia além de um limite aceitável, ENTÃO O SISTEMA DEVE documentar o trade-off entre latência e acurácia.

---

### Requisito 8 — Qualidade de repositório e versionamento

**User Story:** Como avaliador do tech challenge, quero um repositório organizado com histórico semântico, para que o ciclo de desenvolvimento seja auditável e profissional.

#### Critérios de Aceitação (EARS)
1. O SISTEMA DEVE manter um repositório GitHub com estrutura de pastas organizada (código, DAGs, infra, testes, docs).
2. O SISTEMA DEVE adotar um histórico de commits semântico e organizado (ex.: Conventional Commits).
3. O SISTEMA DEVE conter um README documentando: arquitetura, como rodar localmente, como executar a stack de monitoramento e como rodar o pipeline.
4. O SISTEMA DEVE incluir testes automatizados cobrindo ao menos a API e a lógica de inferência.
5. O SISTEMA DEVE documentar os entregáveis do vídeo (STAR) e os pré-requisitos de ambiente.

---

### Requisito 9 — Dados e preparação (dataset de laudos)

**User Story:** Como engenheiro de ML, quero um dataset de laudos rotulados por urgência, para que o modelo possa ser treinado e reproduzido.

#### Critérios de Aceitação (EARS)
1. O SISTEMA DEVE utilizar um dataset de laudos/textos médicos rotulados **em inglês**, com uma coluna de texto (sintoma/laudo) e uma coluna de target (classificação/urgência).
2. O SISTEMA DEVE usar um dataset público simples de classificação de textos médicos em inglês (ex.: Medical Abstracts TC Corpus, recortes do MIMIC-III open access) OU, na ausência deste, gerar um dataset sintético rotulado em inglês.
3. O dataset DEVE conter no mínimo 2.000 amostras.
4. O SISTEMA DEVE mapear/derivar os rótulos para as três classes de urgência {normal, atenção, urgente}.
5. O SISTEMA DEVE documentar a origem/natureza dos dados (público, sintético ou anonimizado).
6. QUANDO os dados são carregados para treino, O SISTEMA DEVE aplicar pré-processamento de texto (ex.: normalização, vetorização TF-IDF).
7. O SISTEMA DEVE separar os dados em treino e teste de forma reproduzível (seed fixa).
8. SE os dados contiverem informação sensível, ENTÃO O SISTEMA DEVE utilizar dados anonimizados/sintéticos (sem PII real).

---

### Requisito 10 — Decisão arquitetural de deploy em nuvem (Etapa 1)

**User Story:** Como arquiteto de soluções, quero analisar e documentar a estratégia de deploy em nuvem mais adequada ao cenário, para que a escolha entre inferência em batch e em tempo real seja justificada tecnicamente.

#### Critérios de Aceitação (EARS)
1. O SISTEMA DEVE conter, no README, uma análise textual da estratégia de deploy em nuvem na **AWS** adequada ao cenário de triagem.
2. A análise DEVE comparar os modos **batch** e **real-time** e justificar a escolha de **real-time** para o caso de triagem de laudos.
3. A arquitetura de nuvem alvo DEVE usar **AWS Lambda com runtime de container** (imagem Docker publicada no **Amazon ECR**) como unidade de execução/orquestração do container de inferência.
4. A arquitetura de nuvem alvo DEVE expor a API através do **Amazon API Gateway** à frente da função Lambda.
5. A análise DEVE justificar a escolha de **Lambda (container runtime) + API Gateway** frente a alternativas (ex.: ECS/Fargate, EKS), considerando latência, cold start, escala sob demanda e simplicidade operacional.
6. A arquitetura de nuvem alvo DEVE prever **Amazon S3** para armazenamento/versionamento do artefato do modelo.
7. A arquitetura de observabilidade em nuvem DEVE utilizar os serviços gerenciados **Amazon Managed Service for Prometheus (AMP)** e **Amazon Managed Grafana (AMG)**, como equivalente gerenciado da stack local de Prometheus + Grafana.
8. A decisão arquitetural DEVE ser consistente com a implementação real-time via API REST em container (imagem Docker) adotada no projeto.
9. A análise DEVE registrar que a stack de monitoramento **local** (Docker Compose com Prometheus + Grafana, Requisito 6) é usada para desenvolvimento/demonstração, enquanto **AMP + AMG** são o alvo em produção na nuvem.

---

### Requisito 11 — Benchmark de latência e entrega final (Etapa 4)

**User Story:** Como avaliador do tech challenge, quero evidências comparativas de latência e um vídeo demonstrativo, para que a otimização e o funcionamento do pipeline sejam comprovados.

#### Critérios de Aceitação (EARS)
1. O SISTEMA DEVE medir e registrar a latência de inferência do modelo **original** (baseline) em ambiente local.
2. O SISTEMA DEVE medir e registrar a latência de inferência do modelo **otimizado** (ex.: ONNX Runtime).
3. O SISTEMA DEVE apresentar um comparativo quantitativo entre o modelo original e o otimizado (ex.: tabela de latências p50/p95 e ganho percentual).
4. O SISTEMA DEVE entregar um vídeo de até 5 minutos no método STAR (Situation, Task, Action, Result) demonstrando o pipeline funcionando.
5. O SISTEMA DEVE incluir o link do vídeo no README.
6. O vídeo DEVE demonstrar: o pipeline em execução, a latência alcançada e as lições aprendidas.

---

## Decisões já tomadas
- **Modelo base**: **TF-IDF + Logistic Regression** (Scikit-Learn) — justificativa no Requisito 1.
- **Latência-alvo**: **p95 < 100 ms** por requisição em ambiente local (meta fixa) — Requisitos 7 e 11.
- **Otimização**: conversão para **ONNX Runtime** (quantização como alternativa) — Etapa 4 / Requisito 7.
- **Dataset**: textos médicos **em inglês**, público (ex.: Medical Abstracts TC Corpus / MIMIC-III open access) ou sintético, com **≥ 2.000 amostras** — Etapa 4 / Requisito 9.
- **Arquitetura de deploy (nuvem alvo)**: **AWS**, modo **real-time** — **AWS Lambda com runtime de container** (imagem no **Amazon ECR**) atrás do **Amazon API Gateway**, com **Amazon S3** para o artefato do modelo. Análise batch vs. real-time documentada no README — Etapa 1 / Requisito 10.
- **Monitoramento (local)**: stack via Docker Compose (API + Prometheus + Grafana) para desenvolvimento/demonstração — Etapa 3 / Requisito 6.
- **Monitoramento (nuvem alvo)**: serviços gerenciados **Amazon Managed Service for Prometheus (AMP)** + **Amazon Managed Grafana (AMG)** — Requisito 10.
- **Entrega**: vídeo STAR (≤ 5 min) com link no README — Etapa 4 / Requisito 11.
