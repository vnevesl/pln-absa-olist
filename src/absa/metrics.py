"""Métricas da ABSA por categoria e testes estatísticos.

Rótulos são DataFrames com uma coluna por aspecto (`asp_<aspecto>`) e valores
em POLARITIES ("ausente", "positivo", "negativo", "neutro").

Métrica principal: F1 micro de pares (aspecto, polaridade) — `tuple_f1`, padrão
da formulação por categoria (SemEval-2016). O F1 macro usa um conjunto de
classes FIXADO PELO GABARITO: uma predição espúria de uma classe que não existe
no teste não pode mudar a média (ver pesquisa/melhorias/04 §2.1).
"""
import inspect
from functools import partial

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, roc_auc_score

from .config import ASPECTS, LABEL_COLUMNS, POLARITIES, SEED

OPINIOES = ["positivo", "negativo", "neutro"]


def labels_from_gold(y_true: pd.DataFrame, include_absent=False) -> dict:
    """Classes com suporte no gabarito, por coluna. Calcular UMA vez no teste e reutilizar."""
    out = {}
    for col in LABEL_COLUMNS:
        presentes = [p for p in POLARITIES if (y_true[col] == p).any()]
        out[col] = presentes if include_absent else [p for p in presentes if p != "ausente"]
    return out


def tuple_f1(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    """F1 micro sobre os pares (aspecto, polaridade) mencionados — métrica principal."""
    tp = fp = fn = 0
    for col in LABEL_COLUMNS:
        t, p = y_true[col].to_numpy(), y_pred[col].to_numpy()
        tp += int(((t == p) & (t != "ausente")).sum())
        fp += int(((p != "ausente") & (t != p)).sum())
        fn += int(((t != "ausente") & (t != p)).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


PRIMARY_METRIC = tuple_f1


def macro_f1(y_true: pd.DataFrame, y_pred: pd.DataFrame, labels=None, include_absent=False) -> float:
    """Média entre aspectos do F1 macro com classes fixadas pelo gabarito (sem "ausente" por padrão)."""
    labels = labels or labels_from_gold(y_true, include_absent)
    return float(np.mean([
        f1_score(y_true[c], y_pred[c], labels=labels[c], average="macro", zero_division=0)
        for c in LABEL_COLUMNS if labels[c]
    ]))


def detection_f1(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    """Média entre aspectos do F1 binário "mencionado ou não"."""
    return float(np.mean([
        f1_score(y_true[c] != "ausente", y_pred[c] != "ausente", zero_division=0) for c in LABEL_COLUMNS
    ]))


def _polarity_f1(t, p):
    """F1 macro da polaridade nas posições em que o gabarito menciona o aspecto.

    Só as polaridades do gabarito entram na média; uma falha de detecção conta
    como erro dessas classes, sem criar uma classe "ausente" com F1 zero.
    """
    mask = t != "ausente"
    labs = [c for c in OPINIOES if (t[mask] == c).any()]
    if not labs:
        return np.nan
    return f1_score(t[mask], p[mask], labels=labs, average="macro", zero_division=0)


def per_aspect_report(y_true: pd.DataFrame, y_pred: pd.DataFrame, labels=None) -> pd.DataFrame:
    """Por aspecto: F1 de pares, F1 macro (classes fixas), detecção, polaridade e menções."""
    labels = labels or labels_from_gold(y_true)
    rows = []
    for asp, col in zip(ASPECTS, LABEL_COLUMNS, strict=True):
        t, p = y_true[col].to_numpy(), y_pred[col].to_numpy()
        so_este = {c: "ausente" for c in LABEL_COLUMNS if c != col}
        rows.append({
            "aspecto": asp,
            "f1_pares": tuple_f1(y_true.assign(**so_este), y_pred.assign(**so_este)),
            "f1_macro": f1_score(t, p, labels=labels[col], average="macro", zero_division=0) if labels[col] else np.nan,
            "f1_deteccao": f1_score(t != "ausente", p != "ausente", zero_division=0),
            "f1_polaridade": _polarity_f1(t, p),
            "mencoes": int((t != "ausente").sum()),
        })
    df = pd.DataFrame(rows)
    df.loc[len(df)] = {"aspecto": "MÉDIA", **df.drop(columns="aspecto").mean().to_dict()}
    return df


def _fix_labels(metric, y_true):
    """Se a métrica aceita `labels`, fixa-os no gabarito completo antes de reamostrar."""
    if "labels" in inspect.signature(metric).parameters:
        return partial(metric, labels=labels_from_gold(y_true))
    return metric


def bootstrap_ci(y_true, y_pred, metric=PRIMARY_METRIC, n=1000, alpha=0.05, seed=SEED):
    """Intervalo de confiança percentil por reamostragem das avaliações."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = y_true.reset_index(drop=True), y_pred.reset_index(drop=True)
    metric = _fix_labels(metric, y_true)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        stats.append(metric(y_true.iloc[idx], y_pred.iloc[idx]))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return metric(y_true, y_pred), float(lo), float(hi)


def paired_bootstrap(y_true, pred_a, pred_b, metric=PRIMARY_METRIC, n=10_000, seed=SEED):
    """Diferença A − B com IC 95%, p-valor BILATERAL e erro padrão (reamostragem pareada)."""
    rng = np.random.default_rng(seed)
    y_true, pred_a, pred_b = (d.reset_index(drop=True) for d in (y_true, pred_a, pred_b))
    metric = _fix_labels(metric, y_true)
    diffs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        diffs[i] = metric(y_true.iloc[idx], pred_a.iloc[idx]) - metric(y_true.iloc[idx], pred_b.iloc[idx])
    p_two = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "diferenca": metric(y_true, pred_a) - metric(y_true, pred_b),
        "ic95": tuple(float(x) for x in np.quantile(diffs, [0.025, 0.975])),
        "p_valor": float(min(1.0, p_two)),
        "ep": float(diffs.std(ddof=1)),
    }


def holm(pvals: dict) -> dict:
    """Correção de Holm (1979) para uma família de p-valores {nome: p}."""
    itens = sorted(pvals.items(), key=lambda kv: kv[1])
    m, ajustados, corrente = len(itens), {}, 0.0
    for i, (nome, p) in enumerate(itens):
        corrente = max(corrente, min(1.0, (m - i) * p))
        ajustados[nome] = corrente
    return ajustados


def mcnemar(y_true, pred_a, pred_b):
    """Teste de McNemar exato sobre acerto da avaliação inteira (todos os aspectos). Secundário."""
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
