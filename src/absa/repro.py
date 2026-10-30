"""Gabarito humano: corpus RePro (Silva et al., PROPOR 2024).

10.003 avaliações da Americanas/B2W anotadas por humanos com tópicos
(multirrótulo) e polaridade. A polaridade do RePro é da avaliação inteira,
não de cada tópico, então a conversão para "polaridade por aspecto" segue:

- polaridade única (ex.: só NEGATIVO): vale para todos os tópicos marcados;
- polaridade mista com um único tópico: o tópico é "misto" — descartado,
  porque o esquema do projeto não tem essa classe;
- polaridade mista com vários tópicos: não dá para saber qual tópico é
  positivo e qual é negativo — descartado.

Os descartes ficam registrados na coluna `motivo_descarte` para o relatório.
"""
import ast
import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ASPECTS, INTERIM, LABEL_COLUMNS, RAW, SEED

REPRO_URL = "https://huggingface.co/datasets/lucasnil/repro/resolve/main/RePro.csv"

TOPIC2ASPECT = {
    "PRODUTO": "produto",
    "ENTREGA": "entrega",
    "CONDICOESDERECEBIMENTO": "recebimento",
    "ANUNCIO": "anuncio",
    "OUTROS": "outros",
}
POLARITY_MAP = {"POSITIVO": "positivo", "NEGATIVO": "negativo", "NEUTRO": "neutro"}


def download_repro(dest=RAW, force=False):
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "RePro.csv"
    if force or not path.exists():
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(REPRO_URL, timeout=60) as resp, tmp.open("wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
        tmp.rename(path)
    return path


def convert(raw: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por avaliação, colunas `asp_*` no esquema do projeto."""
    df = pd.DataFrame({
        "review_id": [f"repro_{i}" for i in range(len(raw))],
        "titulo": raw.review_title,
        "texto": raw.review_text.fillna("").str.strip(),
        "nota": raw.overall_rating,
        "categoria": raw.site_category_lv1,
        "uf": raw.reviewer_state,
        "genero": raw.reviewer_gender,
        "ano_nascimento": raw.reviewer_birth_year,
    })
    polaridades = raw.polarity.map(ast.literal_eval).map(lambda p: sorted({POLARITY_MAP[x] for x in p}))
    n_topicos = raw[list(TOPIC2ASPECT)].sum(axis=1)

    motivo = pd.Series("", index=raw.index)
    motivo[df.texto == ""] = "texto vazio"
    mista = polaridades.map(len) > 1
    motivo[(motivo == "") & mista & (n_topicos == 1)] = "polaridade mista, um tópico"
    motivo[(motivo == "") & mista & (n_topicos > 1)] = "polaridade mista, vários tópicos"
    df["motivo_descarte"] = motivo
    df["inadequada"] = raw.INADEQUADA.astype(bool)

    pol_unica = polaridades.map(lambda p: p[0] if len(p) == 1 else None)
    for topico, asp in TOPIC2ASPECT.items():
        df[f"asp_{asp}"] = [pol if marcado else "ausente" for marcado, pol in zip(raw[topico].astype(bool), pol_unica)]
    return df


def load_repro(test_size=0.2, val_size=0.1, seed=SEED):
    """Baixa, converte e divide em treino/validação/teste (estratificado pela nota).

    Devolve (treino, validacao, teste, descartados). A divisão é determinística.
    """
    raw = pd.read_csv(download_repro())
    df = convert(raw)
    descartados = df[df.motivo_descarte != ""]
    ok = df[df.motivo_descarte == ""].reset_index(drop=True)
    resto, teste = train_test_split(ok, test_size=test_size, stratify=ok.nota, random_state=seed)
    treino, val = train_test_split(resto, test_size=val_size / (1 - test_size), stratify=resto.nota, random_state=seed)
    INTERIM.mkdir(parents=True, exist_ok=True)
    return (treino.reset_index(drop=True), val.reset_index(drop=True),
            teste.reset_index(drop=True), descartados.reset_index(drop=True))


def label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Contagem de cada polaridade por aspecto (para o relatório)."""
    return df[LABEL_COLUMNS].apply(lambda s: s.value_counts()).fillna(0).astype(int).T.rename(
        index=lambda c: c.removeprefix("asp_"))


__all__ = ["ASPECTS", "convert", "download_repro", "label_distribution", "load_repro"]
