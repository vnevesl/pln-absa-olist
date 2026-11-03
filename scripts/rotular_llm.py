"""Rotula o RePro com um LLM para o experimento humano × LLM (hipótese H2).

Rotula, com os MESMOS textos dos rótulos humanos:
  - val_dev (599): concordância LLM × humano, antes de gastar com o resto;
  - teste (1.597): o LLM avaliado como modelo;
  - treino (5.589): rótulos de LLM para treinar o BERTimbau.

Os exemplos do prompt vêm de val_demo (200 da validação, separados de val_dev).
Tudo fica em cache: se cair, rode de novo que continua de onde parou.

Uso (API da Maritaca; a chave fica só no terminal, nunca no código):
    export MARITACA_API_KEY=...
    PYTHONPATH=src python scripts/rotular_llm.py --etapa val     # ~10 min, confira a concordância
    PYTHONPATH=src python scripts/rotular_llm.py --etapa tudo    # teste + treino
"""
import argparse
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from absa.config import ASPECTS, INTERIM, LABEL_COLUMNS, ROOT, SEED
from absa.llm_labeler import label_texts, openai_compatible_generator, select_fixed_examples
from absa.metrics import bootstrap_ci, per_aspect_report, tuple_f1
from absa.repro import load_repro

RESULTS_LLM = ROOT / "results" / "llm"


def dividir_validacao(val):
    demo, dev = train_test_split(val, train_size=200, stratify=val.nota, random_state=SEED)
    return demo.reset_index(drop=True), dev.reset_index(drop=True)


def rotular(df, nome, generate, model_id, exemplos):
    rotulos, _ = label_texts(df.texto_modelo.tolist(), generate, model_id, exemplos, cache_name=f"{model_id}_{nome}")
    validos = [r is not None for r in rotulos]
    out = df[["review_id", "texto", "texto_modelo", "nota"]].copy()
    for a in ASPECTS:
        out[f"asp_{a}"] = [r[a] if r else None for r in rotulos]
    out["valido"] = validos
    out.to_csv(INTERIM / f"repro_{nome}_llm.csv", index=False)
    # Versão sem texto (só review_id + rótulos) vai para o git: o Colab junta com o RePro pelo review_id
    RESULTS_LLM.mkdir(parents=True, exist_ok=True)
    out[["review_id", *LABEL_COLUMNS, "valido"]].to_csv(RESULTS_LLM / f"repro_{nome}_{model_id}.csv", index=False)
    print(f"{nome}: {sum(validos)}/{len(df)} respostas válidas → results/llm/repro_{nome}_{model_id}.csv")
    return out


def avaliar(gold, pred, nome):
    ok = pred.valido.to_numpy()
    # Inválidos contam como erro na avaliação (tudo "ausente"); no treino eles são descartados
    p = pred[LABEL_COLUMNS].where(pd.Series(ok, index=pred.index), "ausente").fillna("ausente")
    f1, lo, hi = bootstrap_ci(gold.reset_index(drop=True), p.reset_index(drop=True))
    print(f"\n{nome}: F1 de pares LLM × humano = {f1:.3f} (IC95% {lo:.3f}–{hi:.3f})")
    print(per_aspect_report(gold.reset_index(drop=True), p.reset_index(drop=True)).round(3).to_string(index=False))
    return {"f1_pares": f1, "ic": [lo, hi], "validos": int(ok.sum()), "n": len(ok),
            "f1_pares_so_validos": tuple_f1(gold[ok].reset_index(drop=True), p[ok].reset_index(drop=True))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etapa", choices=["val", "tudo"], default="val")
    ap.add_argument("--modelo", default="sabiazinho-4")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    treino, val, teste, _ = load_repro()
    demo, dev = dividir_validacao(val)
    exemplos = select_fixed_examples(demo, k=10)
    generate = openai_compatible_generator(model=args.modelo, workers=args.workers)
    INTERIM.mkdir(parents=True, exist_ok=True)

    resumo = {"modelo": args.modelo, "n_exemplos_prompt": len(exemplos)}
    resumo["val_dev"] = avaliar(dev, rotular(dev, "valdev", generate, args.modelo, exemplos), "val_dev")
    if args.etapa == "tudo":
        resumo["teste"] = avaliar(teste, rotular(teste, "teste", generate, args.modelo, exemplos), "teste")
        rot_treino = rotular(treino, "treino", generate, args.modelo, exemplos)
        resumo["treino"] = avaliar(treino, rot_treino, "treino (qualidade dos rótulos de LLM)")
    destino = RESULTS_LLM / f"resumo_{args.modelo}_{args.etapa}.json"
    destino.write_text(json.dumps(resumo, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
