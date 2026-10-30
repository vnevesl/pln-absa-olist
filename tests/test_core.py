import numpy as np
import pandas as pd

from absa.config import ASPECTS, LABEL_COLUMNS
from absa.heuristics import heuristic_labels
from absa.llm_labeler import build_prompt, parse_response
from absa.metrics import macro_f1, mcnemar, per_aspect_report, tuple_f1
from absa.models.baseline import TfidfAspectClassifier
from absa.preprocess import clean_classic, clean_light


def _labels(rows):
    return pd.DataFrame([{f"asp_{a}": r.get(a, "ausente") for a in ASPECTS} for r in rows])


def test_clean_light_keeps_case_and_masks_store():
    assert clean_light("Comprei na Stark, MUITOOOO bom!! http://x.com") == "Comprei na <LOJA>, MUITOO bom!!"


def test_clean_classic_keeps_negation():
    out = clean_classic("O produto NÃO chegou e não gostei")
    assert "nao" in out.split() and "o" not in out.split()


def test_parse_response_normalizes_and_rejects():
    raw = ('Claro: {"produto": "Positiva", "entrega": "negativo", "recebimento": "ausente", '
           '"anuncio": "ausente", "outros": "neutro"}')
    assert parse_response(raw) == {"produto": "positivo", "entrega": "negativo", "recebimento": "ausente",
                                   "anuncio": "ausente", "outros": "neutro"}
    assert parse_response('{"produto": "talvez"}') is None
    assert parse_response("sem json") is None


def test_prompt_contains_examples_and_text():
    ex = {"texto": "chegou rápido", **{a: "ausente" for a in ASPECTS}, "entrega": "positivo"}
    p = build_prompt("produto quebrado", [ex])
    assert "chegou rápido" in p and p.rstrip().endswith("Comentário: produto quebrado\nJSON:")


def test_heuristics_examples():
    lab = heuristic_labels("Produto ótimo, mas chegou atrasado e a caixa veio amassada")
    assert lab["entrega"] == "negativo" and lab["recebimento"] == "negativo"


def test_metrics_perfect_and_tuple():
    y = _labels([{"produto": "positivo", "entrega": "negativo"}, {"entrega": "positivo"}, {}])
    assert macro_f1(y, y) == 1.0 and tuple_f1(y, y) == 1.0
    pred = y.copy()
    pred.loc[0, "asp_entrega"] = "ausente"
    assert tuple_f1(y, pred) < 1.0
    rep = per_aspect_report(y, pred)
    assert rep.iloc[-1]["aspecto"] == "MÉDIA"


def test_mcnemar_counts():
    y = _labels([{"produto": "positivo"}] * 4)
    a = y.copy()
    b = y.copy()
    b.loc[:1, "asp_produto"] = "negativo"
    r = mcnemar(y, a, b)
    assert r["so_a_acerta"] == 2 and r["so_b_acerta"] == 0


def test_baseline_learns_toy_data():
    textos = ["produto otimo", "chegou atrasado", "produto ruim", "entrega rapida"] * 10
    rows = [{"produto": "positivo"}, {"entrega": "negativo"}, {"produto": "negativo"}, {"entrega": "positivo"}] * 10
    y = _labels(rows)
    clf = TfidfAspectClassifier(min_df=1).fit(textos, y)
    pred = clf.predict(textos)
    assert macro_f1(y, pred) > 0.9
    proba = clf.predict_proba(textos)
    assert proba.shape == (40, len(ASPECTS), 4) and np.allclose(proba.sum(axis=2), 1)
    assert list(pred.columns) == LABEL_COLUMNS


def test_repro_conversion_rules():
    from absa.repro import convert
    raw = pd.DataFrame({
        "review_title": ["", "", "", ""],
        "review_text": ["Chegou rápido, produto ótimo", "Bom mas demorou", "Produto ruim e caro", "Normal"],
        "overall_rating": [5, 3, 1, 3],
        "site_category_lv1": ["x"] * 4, "reviewer_state": ["SP"] * 4,
        "reviewer_gender": ["F"] * 4, "reviewer_birth_year": [1990] * 4,
        "polarity": ["['POSITIVO']", "['NEGATIVO', 'POSITIVO']", "['NEGATIVO']", "['NEGATIVO', 'POSITIVO']"],
        "ENTREGA": [1, 1, 0, 0], "PRODUTO": [1, 1, 1, 1], "CONDICOESDERECEBIMENTO": [0] * 4,
        "ANUNCIO": [0] * 4, "OUTROS": [0, 0, 1, 0], "INADEQUADA": [0] * 4,
    })
    df = convert(raw)
    assert df.loc[0, "asp_entrega"] == "positivo" and df.loc[0, "asp_produto"] == "positivo"
    assert df.loc[1, "motivo_descarte"] == "polaridade mista, vários tópicos"
    assert df.loc[2, "asp_outros"] == "negativo" and df.loc[2, "asp_anuncio"] == "ausente"
    assert df.loc[3, "motivo_descarte"] == "polaridade mista, um tópico"
