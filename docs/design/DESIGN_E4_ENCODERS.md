# DESIGN E4 — Encoders: condições de rótulo, sementes e treino misto

**Status: Pronto para execução** · aprovado em 30/10/2026 · D4: 3 sementes

## Objetivo

Deixar `models/encoder.py` e o notebook 05 prontos para rodar no Colab o experimento humano × LLM e o treino
misto, de forma retomável. Treinos de verdade **não** rodam aqui; os testes usam um modelo minúsculo criado
por `from_config` (sem download).

## Referências obrigatórias

- `pesquisa/melhorias/03_llm_anotador.md` §6.1–6.4 e §7.4
- `pesquisa/melhorias/01_adaptacao_de_dominio.md` §5 Passo 2
- `pesquisa/melhorias/04_avaliacao_estatistica_xai.md` §4.2

## E4-S1 · Condições, sementes e retomada

`src/absa/models/encoder.py`:

1. `EncoderAspectClassifier(config, model=None, tokenizer=None)`: aceita modelo e tokenizador já
   construídos (para testes sem rede).
2. `fit(texts, labels, ..., sample_weight=None, soft_targets=None)`:
   - `sample_weight` (n,) pondera a perda por exemplo (`reduction="none"`);
   - `soft_targets` (n, 5, 4) usa entropia cruzada suave no lugar do rótulo duro;
   - `TrainConfig.label_smoothing` (padrão 0.0) é repassado à CE.
3. A seleção de época usa a métrica principal (E1-S1) no dev **humano**.
4. Registra no `history` o tempo por época e o pico de memória da GPU, quando houver.

Novo `src/absa/experiments.py`:

5. `SEEDS = [13, 42, 87]`.
6. `CONDICOES`, cada condição um dicionário `{id, fonte, frac, peso, soft, filtro}`:
   - `H-25`, `H-50`, `H-100`: humanos, subamostras **aninhadas** e estratificadas por nota;
   - `V-only`: treina só no `val_dev`, com época escolhida por validação cruzada interna de 3 dobras;
   - `L`: rótulo duro do LLM;
   - `L-filt`: só exemplos com confiança mínima entre aspectos ≥ τ, com τ escolhido no `val_dev` e gravado;
   - `L-soft`: alvos = probabilidades do LLM.
7. `build_training_set(cond, treino_humano, rotulos_llm, val_dev, seed) -> (textos, rotulos, pesos, soft)`.
8. `run_condition(cond, encoder_name, seed, ...)`: pula se `models/{encoder}_{cond}_s{seed}.pt` **e**
   `data/interim/pred_repro_teste_{encoder}_{cond}_s{seed}.csv` já existirem (retomada pelo Drive). Grava as
   predições com probabilidades (`p_*`).
9. `human_equivalent(curva: dict[n, f1], f1_alvo) -> float`: interpolação log-linear (03 §6.4); fora da faixa,
   devolve `nan` e avisa.

`notebooks/05_encoders.ipynb` (reescrever):

- Laço `for cond in CONDICOES for seed in SEEDS` com retomada.
- Ordem por prioridade: `H-100`, `L`, `L-filt` (BERTimbau, 3 sementes) → `H-25`, `H-50`, `V-only`, `L-soft` (1
  semente) → Albertina `H-100` (1 semente).
- Célula de estimativa do tempo restante antes de cada bloco.
- Célula final com a curva de aprendizado e o equivalente-humano.

Testes (`tests/test_encoder_tiny.py`): com um BERT minúsculo (`BertConfig` com 2 camadas e hidden 32) e um
tokenizador criado de um vocabulário local temporário, `fit` roda 1 época com `sample_weight`, `soft_targets`
e `label_smoothing`; salvar e carregar dá as mesmas probabilidades; os subconjuntos humanos são aninhados;
`human_equivalent` acerta um caso sintético; `run_condition` pula quando os arquivos existem.

## E4-S3 · Treino misto RePro (humano) + Olist (LLM)

1. Condições `X-1.0` e `X-0.5`: treino humano completo do RePro + rótulos LLM da Olist (só `dev`, sem o
   gabarito), com peso 1,0 ou 0,5 nos exemplos da Olist.
2. `build_training_set` suporta essas condições; uma verificação **falha** se algum `review_id` ou texto do
   gabarito Olist estiver no conjunto (usa `gold_exclusion` do E1-S4).
3. Notebook 05: bloco opcional "Treino misto" (3 sementes) após o bloco principal.
4. Testes: pesos corretos por origem; a guarda de vazamento dispara com um exemplo plantado.

## Fora do escopo

Perda de rótulo parcial (E4-S2, noite 2), TAPT (cortado), execução na GPU, avaliação agregada (E5).
