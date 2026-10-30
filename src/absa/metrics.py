"""Métricas da ABSA por categoria e testes estatísticos.

Rótulos são DataFrames com uma coluna por aspecto (`asp_<aspecto>`) e valores
em POLARITIES ("ausente", "positivo", "negativo", "neutro").
"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, roc_auc_score

from .config import ASPECTS, LABEL_COLUMNS, SEED


def per_aspect_report(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
    """F1 macro (4 classes), F1 de detecção (mencionado ou não) e suporte por aspecto."""
    rows = []
    for asp, col in zip(ASPECTS, LABEL_COLUMNS, strict=True):
        t, p = y_true[col].to_numpy(), y_pred[col].to_numpy()
        rows.append({
            "aspecto": asp,
            "f1_macro": f1_score(t, p, average="macro", zero_division=0),
            "f1_deteccao": f1_score(t != "ausente", p != "ausente", zero_division=0),
            "f1_polaridade": _polarity_f1(t, p),
            "mencoes": int((t != "ausente").sum()),
        })
    df = pd.DataFrame(rows)
    df.loc[len(df)] = {"aspecto": "MÉDIA", **df.drop(columns="aspecto").mean().to_dict()}
    return df


def _polarity_f1(t, p):
    """F1 macro da polaridade, só onde o aspecto foi de fato mencionado."""
    mask = t != "ausente"
    if not mask.any():
        return np.nan
    return f1_score(t[mask], p[mask], average="macro", zero_division=0)


def tuple_f1(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    """F1 micro sobre os pares (aspecto, polaridade) mencionados — métrica de ACSA."""
    tp = fp = fn = 0
    for col in LABEL_COLUMNS:
        t, p = y_true[col].to_numpy(), y_pred[col].to_numpy()
        tp += int(((t == p) & (t != "ausente")).sum())
        fp += int(((p != "ausente") & (t != p)).sum())
        fn += int(((t != "ausente") & (t != p)).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def macro_f1(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    """Métrica principal: média do F1 macro dos aspectos (classes presentes no gabarito ou na predição)."""
    return float(np.mean([
        f1_score(y_true[c], y_pred[c], average="macro", zero_division=0)
        for c in LABEL_COLUMNS
    ]))


def bootstrap_ci(y_true, y_pred, metric=macro_f1, n=1000, alpha=0.05, seed=SEED):
    """Intervalo de confiança percentil por reamostragem das avaliações."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = y_true.reset_index(drop=True), y_pred.reset_index(drop=True)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        stats.append(metric(y_true.iloc[idx], y_pred.iloc[idx]))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return metric(y_true, y_pred), float(lo), float(hi)


def paired_bootstrap(y_true, pred_a, pred_b, metric=macro_f1, n=1000, seed=SEED):
    """Diferença A − B com IC e p-valor unilateral (fração de reamostras com A ≤ B)."""
    rng = np.random.default_rng(seed)
    y_true, pred_a, pred_b = (d.reset_index(drop=True) for d in (y_true, pred_a, pred_b))
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        diffs.append(metric(y_true.iloc[idx], pred_a.iloc[idx]) - metric(y_true.iloc[idx], pred_b.iloc[idx]))
    diffs = np.array(diffs)
    return {
        "diferenca": metric(y_true, pred_a) - metric(y_true, pred_b),
        "ic95": tuple(np.quantile(diffs, [0.025, 0.975])),
        "p_valor": float((diffs <= 0).mean()),
    }


def mcnemar(y_true, pred_a, pred_b):
    """Teste de McNemar exato sobre acerto da avaliação inteira (todos os aspectos)."""
    ok_a = (pred_a[LABEL_COLUMNS].to_numpy() == y_true[LABEL_COLUMNS].to_numpy()).all(axis=1)
    ok_b = (pred_b[LABEL_COLUMNS].to_numpy() == y_true[LABEL_COLUMNS].to_numpy()).all(axis=1)
    b = int((ok_a & ~ok_b).sum())  # só A acerta
    c = int((~ok_a & ok_b).sum())  # só B acerta
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"so_a_acerta": b, "so_b_acerta": c, "p_valor": float(p)}


def delivery_validation(atrasado, prob_entrega_negativa):
    """Validação extrínseca: AUC entre P(entrega negativa) e atraso real."""
    atrasado = np.asarray(atrasado, dtype=bool)
    return float(roc_auc_score(atrasado, np.asarray(prob_entrega_negativa)))
