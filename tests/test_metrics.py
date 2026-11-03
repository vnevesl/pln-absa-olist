import numpy as np
import pandas as pd

from absa.config import ASPECTS
from absa.metrics import _polarity_f1, bootstrap_ci, holm, macro_f1, paired_bootstrap, tuple_f1


def _labels(rows):
    return pd.DataFrame([{f"asp_{a}": r.get(a, "ausente") for a in ASPECTS} for r in rows])


GOLD = _labels([{"produto": "positivo", "entrega": "negativo"}, {"produto": "negativo"},
                {"entrega": "positivo", "recebimento": "negativo"}, {"produto": "positivo"}] * 5)


def test_spurious_absent_class_does_not_move_macro():
    pred = GOLD.copy()
    pred.loc[0, "asp_anuncio"] = "neutro"  # classe que não existe no gabarito de anúncio
    assert abs(macro_f1(GOLD, pred) - macro_f1(GOLD, GOLD)) < 1e-3


def test_polarity_f1_ignores_detection_misses():
    t = np.array(["positivo", "negativo", "positivo", "ausente"])
    p_ok = np.array(["positivo", "negativo", "positivo", "ausente"])
    p_miss = np.array(["positivo", "negativo", "ausente", "ausente"])  # perdeu a detecção de um positivo
    assert _polarity_f1(t, p_ok) == 1.0
    assert 0.5 < _polarity_f1(t, p_miss) < 1.0  # erra o positivo, mas "ausente" não vira classe com F1 = 0


def test_tuple_f1_bounds():
    assert tuple_f1(GOLD, GOLD) == 1.0
    tudo_ausente = GOLD.copy()
    tudo_ausente[:] = "ausente"
    assert tuple_f1(GOLD, tudo_ausente) == 0.0


def test_holm_matches_hand_example():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert np.isclose(adj["a"], 0.03) and np.isclose(adj["c"], 0.06) and np.isclose(adj["b"], 0.06)


def test_paired_bootstrap_identical_models():
    r = paired_bootstrap(GOLD, GOLD, GOLD, n=200)
    assert r["diferenca"] == 0 and r["p_valor"] == 1.0


def test_bootstrap_ci_fixes_labels_with_macro():
    f1, lo, hi = bootstrap_ci(GOLD, GOLD, metric=macro_f1, n=100)
    assert f1 == lo == hi == 1.0
