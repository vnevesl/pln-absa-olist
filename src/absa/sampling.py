"""Amostra opcional da Olist para anotação manual (teste de mudança de domínio).

O gabarito principal é o RePro (ver repro.py). Estes 100 comentários da Olist
servem para medir se um modelo avaliado na Americanas/B2W funciona na Olist e
para cobrir avaliações de polaridade mista ("bom, mas atrasou"), que o RePro
não permite usar. A amostra é estratificada por nota e sobre-representa
pedidos atrasados.
"""
import numpy as np
import pandas as pd

from .config import ANNOTATED, LABEL_COLUMNS, POLARITIES, SEED

QUOTA_POR_NOTA = {1: 25, 2: 15, 3: 20, 4: 15, 5: 25}  # total 100
FRACAO_ATRASADOS = 0.30


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


def load_olist_gold(arquivo="olist_anotado.csv") -> pd.DataFrame:
    """Anotação manual da Olist + metadados (nota, atraso, UF, categoria).

    Falha se houver célula de aspecto vazia ou com valor fora de POLARITIES.
    """
    anot = pd.read_csv(ANNOTATED / arquivo, keep_default_na=False)
    for col in LABEL_COLUMNS:
        anot[col] = anot[col].str.strip().str.lower()
        invalidos = anot.loc[~anot[col].isin(POLARITIES), ["ordem_anotacao", col]]
        if len(invalidos):
            raise ValueError(f"{col}: {len(invalidos)} valores inválidos, ex.:\n{invalidos.head()}")
    meta = pd.read_csv(ANNOTATED / "olist_metadados.csv").drop(columns=["texto", "ordem_anotacao"])
    return anot.merge(meta, on="review_id", how="left", validate="one_to_one")
