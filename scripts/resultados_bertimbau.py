"""Tabelas do artigo a partir dos resultados do Colab (results/encoders/).

H1: BERTimbau (3 sementes) × TF-IDF + regressão logística no teste humano do RePro.
H2: BERTimbau com rótulos humanos (H) × rótulos do Qwen2.5-7B (L, results/h2/).
H3: P(entrega negativa) × atraso real na metade "relato" da Olist.

Uso: PYTHONPATH=src python scripts/resultados_bertimbau.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from absa.config import ASPECTS, LABEL_COLUMNS, POLARITIES, ROOT, SEED
from absa.data import load_reviews
from absa.metrics import bootstrap_ci, detection_f1, holm, macro_f1, paired_bootstrap, per_aspect_report, tuple_f1
from absa.models.baseline import TfidfAspectClassifier
from absa.preprocess import clean_classic
from absa.repro import load_repro

RES = ROOT / "results" / "encoders"
RES_H2 = ROOT / "results" / "h2"
treino, val, teste, _ = load_repro()


def ler(arq: Path) -> pd.DataFrame:
    return pd.read_csv(arq).set_index("review_id").loc[teste.review_id, LABEL_COLUMNS].reset_index(drop=True)


tfidf = TfidfAspectClassifier("logreg").fit(treino.texto_modelo.map(clean_classic), treino)
pred_tfidf = tfidf.predict(teste.texto_modelo.map(clean_classic))
modelos = {"TF-IDF + RL": [pred_tfidf]}
for cond in ["H", "V", "L"]:
    arqs = sorted(RES.glob(f"pred_teste_bertimbau_{cond}_s*.csv")) or sorted(RES_H2.glob(f"pred_teste_bertimbau_{cond}_s*.csv"))
    if arqs:
        modelos[f"BERTimbau {cond}"] = [ler(a) for a in arqs]

resumo = {}
print("H1 — teste humano do RePro (n = %d)" % len(teste))
for nome, preds in modelos.items():
    pares = [tuple_f1(teste, p) for p in preds]
    macro = [macro_f1(teste, p) for p in preds]
    det = [detection_f1(teste, p) for p in preds]
    mediana = preds[int(np.argsort(pares)[len(pares) // 2])]
    f1, lo, hi = bootstrap_ci(teste, mediana)
    resumo[nome] = {"sementes": len(preds), "f1_pares_media": np.mean(pares), "f1_pares_dp": np.std(pares, ddof=1)
                    if len(preds) > 1 else None, "ic_semente_mediana": [lo, hi],
                    "f1_macro_media": np.mean(macro), "f1_deteccao_media": np.mean(det)}
    dp = f" ± {np.std(pares, ddof=1):.3f}" if len(preds) > 1 else ""
    print(f"  {nome:15s} pares {np.mean(pares):.3f}{dp} (IC da semente mediana {lo:.3f}–{hi:.3f}) | "
          f"macro {np.mean(macro):.3f} | detecção {np.mean(det):.3f}")

print("\nComparações pareadas (bootstrap bilateral, semente 42), com Holm")
h42 = ler(RES / "pred_teste_bertimbau_H_s42.csv")
comparacoes = {"BERTimbau H − TF-IDF": paired_bootstrap(teste, h42, pred_tfidf)}
if (RES / "pred_teste_bertimbau_V_s42.csv").exists():
    comparacoes["BERTimbau H − BERTimbau V"] = paired_bootstrap(teste, h42, ler(RES / "pred_teste_bertimbau_V_s42.csv"))
if (RES_H2 / "pred_teste_bertimbau_L_s42.csv").exists():
    comparacoes["BERTimbau H − BERTimbau L"] = paired_bootstrap(teste, h42, ler(RES_H2 / "pred_teste_bertimbau_L_s42.csv"))
ajust = holm({k: v["p_valor"] for k, v in comparacoes.items()})
for k, v in comparacoes.items():
    print(f"  {k:28s} Δ = {v['diferenca']:+.3f} (IC95% {v['ic95'][0]:+.3f} a {v['ic95'][1]:+.3f}) "
          f"p = {v['p_valor']:.4f}, p-Holm = {ajust[k]:.4f}")
resumo["comparacoes"] = {k: {**v, "p_holm": ajust[k]} for k, v in comparacoes.items()}

if "BERTimbau L" in resumo:
    print("\nH2 — qualidade dos rótulos do LLM e recuperação do gap")
    qual = json.loads((RES_H2 / "qualidade_rotulos_qwen7b.json").read_text())
    valdev = pd.read_csv(RES_H2 / "repro_valdev_qwen7b.csv")
    m = {c: resumo[f"BERTimbau {c}"]["f1_pares_media"] for c in "HLV"}
    gap = (m["L"] - m["V"]) / (m["H"] - m["V"])
    h2 = {**qual, "validos_valdev": int(valdev.valido.sum()), "n_valdev": len(valdev), "n_treino": len(treino),
          "recuperacao_gap": gap}
    print(f"  concordância LLM × humano: val_dev {qual['f1_pares_valdev']:.3f} "
          f"(IC {qual['ic_valdev'][0]:.3f}–{qual['ic_valdev'][1]:.3f}), treino {qual['f1_pares_treino']:.3f}")
    print(f"  respostas válidas: val_dev {h2['validos_valdev']}/{len(valdev)}, treino {qual['validos_treino']}/{len(treino)}"
          f" | {qual['seg_por_texto']:.2f} s por texto na T4")
    print(f"  recuperação do gap (L − V)/(H − V) = {gap:.1%}")
    resumo["h2"] = h2

print("\nPor aspecto — BERTimbau H (média das sementes) × TF-IDF")
relatorios_h = [per_aspect_report(teste, p).set_index("aspecto") for p in modelos["BERTimbau H"]]
rep_h = sum(relatorios_h) / len(relatorios_h)
rep_t = per_aspect_report(teste, pred_tfidf).set_index("aspecto")
tabela = pd.DataFrame({"pares_bert": rep_h.f1_pares, "pares_tfidf": rep_t.f1_pares,
                       "macro_bert": rep_h.f1_macro, "macro_tfidf": rep_t.f1_macro}).round(3)
print(tabela.to_string())
resumo["por_aspecto"] = tabela.to_dict()
if "BERTimbau L" in modelos:
    rel_l = [per_aspect_report(teste, p).set_index("aspecto") for p in modelos["BERTimbau L"]]
    rep_l = sum(rel_l) / len(rel_l)
    tab_l = pd.DataFrame({"pares_H": rep_h.f1_pares, "pares_L": rep_l.f1_pares,
                          "macro_H": rep_h.f1_macro, "macro_L": rep_l.f1_macro}).round(3)
    print("\nPor aspecto — H × L (média das sementes)")
    print(tab_l.to_string())
    resumo["por_aspecto_L"] = tab_l.to_dict()

print("\nH3 — Olist, metade 'relato' (BERTimbau H s42 treinado só no RePro)")
olist = pd.read_csv(RES / "olist_relato_probs_bertimbau_H_s42.csv")
y = olist.atrasado.to_numpy()
rng = np.random.default_rng(SEED)


def auc_ic(y, s, n=1000):
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(roc_auc_score(y[i], s[i]))
    return roc_auc_score(y, s), *np.quantile(vals, [0.025, 0.975])


base = load_reviews().set_index("review_id")
txt = base.loc[olist.review_id, "texto_modelo"].map(clean_classic)
p_tfidf = tfidf.predict_proba(txt)[:, ASPECTS.index("entrega"), POLARITIES.index("negativo")]
escores = {"nota (invertida)": -olist.nota.to_numpy(), "BERTimbau P(entrega neg)": olist.p_entrega_negativo.to_numpy(),
           "TF-IDF P(entrega neg)": p_tfidf, "placebo BERTimbau P(produto neg)": olist.p_produto_negativo.to_numpy()}
h3 = {"n": len(olist), "atrasados": int(y.sum())}
print(f"  n = {len(olist)}, atrasados = {y.sum()}")
for nome, s in escores.items():
    a, lo, hi = auc_ic(y, s)
    h3[nome] = [a, lo, hi]
    print(f"  {nome:34s} AUC {a:.3f} (IC95% {lo:.3f}–{hi:.3f})")
X_nota = pd.get_dummies(olist.nota.astype(str)).to_numpy(float)
for nome, X in [("nota", X_nota),
                ("nota + BERTimbau P(entrega neg)", np.column_stack([X_nota, olist.p_entrega_negativo])),
                ("nota + TF-IDF P(entrega neg)", np.column_stack([X_nota, p_tfidf]))]:
    s = cross_val_predict(LogisticRegression(max_iter=1000), X, y, cv=5, method="predict_proba")[:, 1]
    h3[f"cv5 {nome}"] = roc_auc_score(y, s)
    print(f"  regressão logística (validação cruzada 5x) com {nome}: AUC {roc_auc_score(y, s):.3f}")
for nome, s in [("BERTimbau", olist.p_entrega_negativo.to_numpy()), ("TF-IDF", p_tfidf)]:
    por_nota = {int(n): roc_auc_score(y[olist.nota == n], s[olist.nota == n]) for n in range(1, 6)}
    h3[f"por_nota {nome}"] = por_nota
    print(f"  dentro de cada nota ({nome}):", " / ".join(f"{n}★ {a:.3f}" for n, a in por_nota.items()))
resumo["h3"] = h3

(RES / "resumo_artigo.json").write_text(json.dumps(resumo, indent=1, ensure_ascii=False, default=float))
print("\n→ results/encoders/resumo_artigo.json")
