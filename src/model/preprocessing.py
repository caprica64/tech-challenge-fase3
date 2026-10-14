"""Pré-processamento de texto para a triagem de laudos médicos.

Este módulo centraliza a **normalização de texto** e a **construção do
vetorizador TF-IDF** usados pelo pipeline de treino (``train.py``) e,
indiretamente, pela inferência. Mantê-los juntos aqui garante que o
pré-processamento aplicado no treino seja exatamente o mesmo aplicado na
inferência — evitando divergência treino/inferência (R9.6).

------------------------------------------------------------------------------
Estratégia (R9.6, decisão de design)
------------------------------------------------------------------------------
O grosso do pré-processamento é delegado ao próprio ``TfidfVectorizer`` do
scikit-learn, que já faz, de forma coesa e serializável junto ao modelo:

    - ``lowercase=True`` — normalização de caixa;
    - ``stop_words="english"`` — remoção de stopwords básicas do inglês;
    - ``ngram_range=(1, 2)`` — unigramas + bigramas, capturando contexto curto;
    - tokenização por expressão regular padrão (``token_pattern``), que já
      descarta pontuação ao extrair apenas tokens alfanuméricos.

Como o vetorizador é parte do ``Pipeline`` sklearn (serializado em
``model.pkl`` e exportável para ONNX), todo esse pré-processamento viaja junto
com o artefato — não há etapa manual a reproduzir na inferência.

A função ``normalize_text`` fica disponível como uma normalização leve e
determinística (colapso de espaços em branco) que pode ser aplicada à entrada
antes da vetorização, mantendo a lógica num único lugar.
"""

from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer

#: Idioma das stopwords usadas pelo vetorizador (dataset em inglês, R9.1).
STOP_WORDS = "english"

#: Faixa de n-gramas do TF-IDF: unigramas + bigramas.
NGRAM_RANGE: tuple[int, int] = (1, 2)

#: Frequência mínima de documento para um termo entrar no vocabulário.
#: Descarta termos raríssimos (ruído/typos) sem perder sinal relevante.
MIN_DF = 2

#: Regex para colapsar sequências de espaços em branco em um único espaço.
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normaliza um laudo de texto de forma leve e determinística.

    Aplica normalização de caixa (minúsculas) e colapso de espaços em branco
    (incluindo quebras de linha e tabs) em um único espaço, removendo espaços
    nas pontas. A remoção de pontuação e stopwords é responsabilidade do
    ``TfidfVectorizer`` (ver docstring do módulo), mantendo o pré-processamento
    coeso com o vetorizador serializado.

    Args:
        text: o texto do laudo. Qualquer valor não-string é convertido via
            ``str`` para robustez.

    Returns:
        O texto normalizado (minúsculas, espaços colapsados, sem bordas).
    """
    if not isinstance(text, str):
        text = str(text)
    return _WHITESPACE_RE.sub(" ", text).strip().lower()


def build_vectorizer(
    *,
    stop_words: str = STOP_WORDS,
    ngram_range: tuple[int, int] = NGRAM_RANGE,
    min_df: int = MIN_DF,
) -> TfidfVectorizer:
    """Constrói o ``TfidfVectorizer`` padrão do projeto.

    Centraliza a configuração do vetorizador para que treino e inferência
    usem exatamente a mesma transformação (R9.6). O ``lowercase=True`` do
    vetorizador complementa ``normalize_text``, garantindo normalização de
    caixa mesmo que o texto não passe por ``normalize_text``.

    Args:
        stop_words: lista/idioma de stopwords (padrão: inglês).
        ngram_range: faixa de n-gramas (padrão: unigramas + bigramas).
        min_df: frequência mínima de documento para incluir um termo.

    Returns:
        Uma instância de ``TfidfVectorizer`` configurada.
    """
    return TfidfVectorizer(
        lowercase=True,
        stop_words=stop_words,
        ngram_range=ngram_range,
        min_df=min_df,
    )
