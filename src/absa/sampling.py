"""Seleção do conjunto de referência (gabarito) para anotação manual.

Estratifica por nota e por situação de entrega para não deixar as classes
raras de fora: comentários de 2 e 3 estrelas e de pedidos atrasados são
sobre-representados em relação à base.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ANNOTATED, LABEL_COLUMNS, POLARITIES, SEED

# Quantos comentários sortear por nota (total 800)
QUOTA_POR_NOTA = {1: 200, 2: 130, 3: 150, 4: 140, 5: 180}
FRACAO_ATRASADOS = 0.30  # dentro de cada nota, ao menos 30% de pedidos atrasados


def sample_gold(df: pd.DataFrame, quota=QUOTA_POR_NOTA, frac_late=FRACAO_ATRASADOS,
                min_palavras=3, seed=SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pool = df[df.n_palavras >= min_palavras].drop_duplicates("texto")
    partes = []
    for nota, n in quota.items():
        grupo = pool[pool.nota == nota]
        atrasados, pontuais = grupo[grupo.atrasado], grupo[~grupo.atrasado]
        n_late = min(len(atrasados), int(round(n * frac_late)))
        partes.append(atrasados.sample(n_late, random_state=rng.integers(1e9)))
        partes.append(pontuais.sample(n - n_late, random_state=rng.integers(1e9)))
    gold = pd.concat(partes).sample(frac=1, random_state=seed).reset_index(drop=True)
    gold["ordem_anotacao"] = np.arange(len(gold))
    return gold


def load_gold(arquivo="gabarito_anotado.csv"):
    """Gabarito anotado + metadados (nota, atraso, UF, categoria) e divisão validação/teste.

    Falha se houver célula de aspecto vazia ou com valor fora de POLARITIES.
    """
    anot = pd.read_csv(ANNOTATED / arquivo, keep_default_na=False)
    for col in LABEL_COLUMNS:
        anot[col] = anot[col].str.strip().str.lower()
        invalidos = anot.loc[~anot[col].isin(POLARITIES), ["ordem_anotacao", col]]
        if len(invalidos):
            raise ValueError(f"{col}: {len(invalidos)} valores inválidos, ex.:\n{invalidos.head()}")
    meta = pd.read_csv(ANNOTATED / "gabarito_metadados.csv").drop(columns=["texto", "ordem_anotacao"])
    gold = anot.merge(meta, on="review_id", how="left", validate="one_to_one")
    return split_gold(gold)


def split_gold(gold: pd.DataFrame, n_dev=300, seed=SEED):
    """Divide em validação (ajuste de prompts e limiares) e teste (congelado)."""
    dev, test = train_test_split(gold, train_size=n_dev, stratify=gold.nota, random_state=seed)
    return dev.reset_index(drop=True), test.reset_index(drop=True)


def double_annotation_subset(gold: pd.DataFrame, n=200, seed=SEED) -> pd.DataFrame:
    """Subamostra para o segundo anotador (cálculo do kappa de Cohen)."""
    return gold.sample(n, random_state=seed).sort_values("ordem_anotacao")
