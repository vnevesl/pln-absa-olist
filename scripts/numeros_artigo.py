"""Reproduz os números das Tabelas 3, 4 e 5 do artigo (CPU, ~2 min).

Uso: PYTHONPATH=src python scripts/numeros_artigo.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from absa.config import ASPECTS, LABEL_COLUMNS, POLARITIES, SEED
from absa.data import load_reviews
from absa.heuristics import heuristic_labels
from absa.metrics import tuple_f1
from absa.models.baseline import TfidfAspectClassifier
from absa.preprocess import clean_classic
from absa.repro import load_repro

treino, val, teste, _ = load_repro()
# Classes fixadas pelo gabarito, sem "ausente" (ver pesquisa/melhorias/04 §3.3)
LABELS = {c: [p for p in POLARITIES if p != "ausente" and (teste[c] == p).any()] for c in LABEL_COLUMNS}


def macro_fixo(y, p):
    return float(np.mean([f1_score(y[c], p[c], labels=LABELS[c], average="macro", zero_division=0)
                          for c in LABEL_COLUMNS]))


def deteccao(y, p):
    return float(np.mean([f1_score(y[c] != "ausente", p[c] != "ausente", zero_division=0) for c in LABEL_COLUMNS]))


RNG = np.random.default_rng(SEED)


def ic(y, p, metrica, n=1000, rng=RNG):
    y, p = y.reset_index(drop=True), p.reset_index(drop=True)
    amostras = [metrica(y.iloc[i], p.iloc[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return metrica(y, p), *np.quantile(amostras, [0.025, 0.975])


regras = pd.DataFrame([heuristic_labels(t) for t in teste.texto])
regras.columns = LABEL_COLUMNS
preds = {"Regras": regras}
X_treino, X_teste = treino.texto.map(clean_classic), teste.texto.map(clean_classic)
modelos = {nome: TfidfAspectClassifier(nome).fit(X_treino, treino) for nome in ["rf", "logreg"]}
preds["TF-IDF + random forest"] = modelos["rf"].predict(X_teste)
preds["TF-IDF + regressão logística"] = modelos["logreg"].predict(X_teste)

print("Tabela 3 — teste humano do RePro")
for nome, p in preds.items():
    f, lo, hi = ic(teste, p, tuple_f1)
    print(f"  {nome:30s} pares {f:.3f} ({lo:.3f}–{hi:.3f}) | macro {macro_fixo(teste, p):.3f} | "
          f"detecção {deteccao(teste, p):.3f}")

print("Tabela 4 — regressão logística por aspecto")
p = preds["TF-IDF + regressão logística"]
for c in LABEL_COLUMNS:
    so = [x for x in LABEL_COLUMNS if x != c]
    pares = tuple_f1(teste.assign(**{x: "ausente" for x in so}), p.assign(**{x: "ausente" for x in so}))
    macro = f1_score(teste[c], p[c], labels=LABELS[c], average="macro", zero_division=0)
    print(f"  {c:16s} pares {pares:.3f} | macro {macro:.3f}")

print("Tabela 5 — atraso real na Olist")
olist = load_reviews()
olist = olist[olist.entregue].reset_index(drop=True)
proba = modelos["logreg"].predict_proba(olist.texto.map(clean_classic))
neg = POLARITIES.index("negativo")
p_entrega = proba[:, ASPECTS.index("entrega"), neg]
p_produto = proba[:, ASPECTS.index("produto"), neg]
y = olist.atrasado.to_numpy()
print(f"  n = {len(olist)}, atrasados = {y.sum()}")
print(f"  nota invertida {roc_auc_score(y, -olist.nota):.3f} | P(entrega neg) {roc_auc_score(y, p_entrega):.3f} | "
      f"placebo P(produto neg) {roc_auc_score(y, p_produto):.3f}")
X_nota = pd.get_dummies(olist.nota.astype(str))
X_ambos = X_nota.assign(p_entrega=p_entrega)
for nome, X in [("nota", X_nota), ("nota + P(entrega neg)", X_ambos)]:
    score = LogisticRegression(max_iter=1000).fit(X, y).predict_proba(X)[:, 1]  # mesma amostra (ver artigo)
    print(f"  regressão logística com {nome}: AUC {roc_auc_score(y, score):.3f}")
por_nota = {n: roc_auc_score(y[olist.nota == n], p_entrega[olist.nota == n]) for n in range(1, 6)}
print("  dentro de cada nota:", " / ".join(f"{n}★ {a:.3f}" for n, a in por_nota.items()))
